"""
semantic_rerank.py
------------------
Re-Ranking in Semantic Space para retrieval explicavel.

A primeira fase (DINO+Aligner) traz um top-K denso. Esse modulo re-pontua
os candidatos usando os scene graphs gerados pelo RelTR: cada tripla
(s, p, o) e codificada como texto via Qwen3 e comparada com a query no
mesmo espaco semantico do retrieval (4096-d, normalizado).

Score final por imagem:
    final = alpha * score_dense + (1 - alpha) * score_sg

onde `score_sg` e um pooling das similaridades query-tripla
(max ou top-N mean).

Alem do re-rank, o modulo retorna as triplas mais alinhadas com a query
para fins de explicabilidade (quais relacoes "puxaram" a imagem para cima).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.reltr_wrapper import SceneGraph, SceneGraphTriple


@dataclass
class ReRankResult:
    """Resultado de re-ranking para uma unica imagem."""
    path: str
    sg: SceneGraph
    score_dense: float
    score_sg: float
    score_final: float
    rank_before: int
    rank_after: int = -1
    top_triples: List[Tuple[SceneGraphTriple, float]] = field(default_factory=list)


def triple_to_text(triple: SceneGraphTriple) -> str:
    """Converte uma tripla em sentenca natural para encoding textual."""
    return f"{triple.subject} {triple.predicate} {triple.object}"


class SemanticReRanker:
    """
    Re-ranqueia candidatos do retrieval inicial usando similaridade
    semantica entre a query e as triplas de cada scene graph.

    Parameters
    ----------
    qwen:
        Embedder textual ja carregado (mesmo usado para a query do retrieval).
    alpha:
        Peso do score denso original. `final = alpha*dense + (1-alpha)*sg`.
        Default 0.5.
    pooling:
        Como agregar similaridades query-tripla por imagem:
        - "max": maior similaridade (foco na melhor tripla).
        - "topn_mean": media das top-N (mais robusto).
    top_n_pool:
        N para pooling "topn_mean". Default 3.
    explain_top_k:
        Numero de triplas a guardar como explicacao por imagem. Default 3.
    """

    def __init__(
        self,
        qwen: QwenSceneEmbedder,
        alpha: float = 0.5,
        pooling: str = "topn_mean",
        top_n_pool: int = 3,
        explain_top_k: int = 3,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha deve estar em [0,1], recebido {alpha}")
        if pooling not in {"max", "topn_mean"}:
            raise ValueError(f"pooling deve ser 'max' ou 'topn_mean', recebido {pooling}")

        self.qwen = qwen
        self.alpha = alpha
        self.pooling = pooling
        self.top_n_pool = top_n_pool
        self.explain_top_k = explain_top_k

    @torch.no_grad()
    def _encode_triples(self, triples: List[SceneGraphTriple]) -> torch.Tensor:
        """
        Encode triplas como texto -> [T, 4096] normalizado (CPU float32).
        Usa o mesmo pipeline da query (embed_components com normalize=False
        + F.normalize), garantindo consistencia de espaco.
        """
        if not triples:
            return torch.empty(0, 4096)

        texts = [triple_to_text(t) for t in triples]
        # embed_components espera list[list[str]] = [[t1, t2, ...]] -> [1, T, 4096]
        emb = self.qwen.embed_components([texts], normalize=False)
        emb = emb.squeeze(0)                                # [T, 4096]
        emb = F.normalize(emb, p=2, dim=-1).float().cpu()
        return emb

    def _pool(self, sims: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """
        Reduz similaridades [T] -> escalar.

        Returns
        -------
        score:
            Score agregado.
        order:
            Indices das triplas em ordem decrescente de similaridade
            (usado para explicabilidade).
        """
        order = torch.argsort(sims, descending=True)
        if sims.numel() == 0:
            return 0.0, order

        if self.pooling == "max":
            return sims.max().item(), order

        # topn_mean
        n = min(self.top_n_pool, sims.numel())
        return sims[order[:n]].mean().item(), order

    @torch.no_grad()
    def rerank(
        self,
        query_emb: torch.Tensor,
        candidates: List[Tuple[str, float, SceneGraph]],
    ) -> List[ReRankResult]:
        """
        Re-ranqueia candidatos.

        Parameters
        ----------
        query_emb:
            Embedding da query `[4096]` ja normalizado (mesmo do retrieval).
        candidates:
            Lista de `(path, dense_score, scene_graph)` na ordem do top-K
            denso (rank 1 primeiro).

        Returns
        -------
        Lista de `ReRankResult` ordenada pelo score final (melhor primeiro).
        Cada resultado preserva `rank_before` (posicao no top-K denso, 1-based)
        e recebe `rank_after` (posicao apos re-rank).
        """
        q = query_emb.float().cpu().squeeze()  # [4096]

        # Normaliza dense_scores para [0,1] dentro do conjunto. Sem isso,
        # a soma ponderada vira-se irrelevante quando os scores tem escalas
        # diferentes (cosine ja em [-1,1] pode dominar ou ser dominado).
        dense_scores = torch.tensor([c[1] for c in candidates], dtype=torch.float32)
        dense_norm = self._minmax_norm(dense_scores)

        results: List[ReRankResult] = []

        sg_scores_raw: List[float] = []
        per_item_pack: List[Tuple[List[SceneGraphTriple], torch.Tensor, torch.Tensor]] = []

        for path, _, sg in candidates:
            triples = sg.triples
            if not triples:
                sg_scores_raw.append(0.0)
                per_item_pack.append((triples, torch.empty(0), torch.empty(0, dtype=torch.long)))
                continue

            t_emb = self._encode_triples(triples)            # [T, 4096]
            sims = t_emb @ q                                  # [T]
            score, order = self._pool(sims)
            sg_scores_raw.append(score)
            per_item_pack.append((triples, sims, order))

        sg_scores = torch.tensor(sg_scores_raw, dtype=torch.float32)
        sg_norm = self._minmax_norm(sg_scores)

        for i, (path, dense, sg) in enumerate(candidates):
            triples, sims, order = per_item_pack[i]
            final = self.alpha * dense_norm[i].item() + (1.0 - self.alpha) * sg_norm[i].item()

            top_triples: List[Tuple[SceneGraphTriple, float]] = []
            if triples and sims.numel() > 0:
                top_idx = order[: self.explain_top_k].tolist()
                top_triples = [(triples[j], sims[j].item()) for j in top_idx]

            results.append(ReRankResult(
                path=path,
                sg=sg,
                score_dense=float(dense),
                score_sg=float(sg_scores_raw[i]),
                score_final=float(final),
                rank_before=i + 1,
                top_triples=top_triples,
            ))

        results.sort(key=lambda r: r.score_final, reverse=True)
        for new_rank, r in enumerate(results, 1):
            r.rank_after = new_rank
        return results

    @staticmethod
    def _minmax_norm(x: torch.Tensor) -> torch.Tensor:
        """Min-max em [0,1]. Se constante, retorna 0.5 para todos (neutro)."""
        if x.numel() == 0:
            return x
        lo, hi = x.min(), x.max()
        if (hi - lo).abs() < 1e-8:
            return torch.full_like(x, 0.5)
        return (x - lo) / (hi - lo)
