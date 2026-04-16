"""
relation_predictor.py
---------------------
Stage 3 do pipeline SGG: prediz predicados para pares de objetos detectados.

Entrada: lista de deteccoes (com bbox_grid) + DINOv3 visual grid.
Saida: lista de edges {source, target, relation, confidence}.

Implementacao independente de generation.py — reimplementa roi_mean_pool
e compute_union_bbox como metodos desta classe.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.SG.relation_head import RelationHead


class HasBboxGrid(Protocol):
    """Protocolo minimo para deteccoes aceitas pelo RelationPredictor."""
    bbox_grid: torch.Tensor  # [4] coords em grade 32x32


@dataclass
class Detection:
    """Deteccao de objeto com bbox em coordenadas de grade 32x32."""
    label: str
    bbox_grid: torch.Tensor  # [4] (x1, y1, x2, y2) em coords 0..32
    score: float = 0.0
    node_id: int = -1


class RelationPredictor:
    """
    Stage 3 do pipeline SGG: prediz predicados para pares de objetos detectados.

    Entrada: lista de Detection (com bbox_grid) + DINOv3 visual grid.
    Saida: lista de edges {source, target, relation, confidence}.

    Parameters
    ----------
    relation_head : RelationHead
        Head treinada para classificacao de predicados.
    aligner : LoRACrossAttentionAligner
        Aligner congelado (usa visual_proj para projetar 768 -> 4096).
    pred_vocab : list[str]
        Vocabulario de predicados (mesma ordem do treino).
    edge_threshold : float
        Confianca minima para incluir uma edge (default 0.0).
    max_edges_per_pair : int
        Numero maximo de predicados por par ordenado (default 1).
    """

    def __init__(
        self,
        relation_head: RelationHead,
        aligner: LoRACrossAttentionAligner,
        pred_vocab: list[str],
        edge_threshold: float = 0.0,
        max_edges_per_pair: int = 1,
    ):
        self.relation_head = relation_head
        self.aligner = aligner
        self.pred_vocab = pred_vocab
        self.edge_threshold = edge_threshold
        self.max_edges_per_pair = max_edges_per_pair

    @staticmethod
    def _roi_mean_pool(
        grid: torch.Tensor,
        bboxes: torch.Tensor,
        img_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mean-pool de patches dentro de cada bbox.

        Paridade 1:1 com train_relation_head.roi_mean_pool.

        Shapes
        ------
        grid    : [B, 32, 32, 768]
        bboxes  : [M, 4] (x1, y1, x2, y2) em coords de grade
        img_idx : [M] indice da imagem no batch

        Returns
        -------
        feats : [M, 768]
        """
        M = bboxes.size(0)
        out = grid.new_zeros(M, grid.size(-1))
        # quantiza bbox -> indices de patch
        x1 = bboxes[:, 0].floor().clamp(0, 31).long()
        y1 = bboxes[:, 1].floor().clamp(0, 31).long()
        x2 = bboxes[:, 2].ceil().clamp(1, 32).long()
        y2 = bboxes[:, 3].ceil().clamp(1, 32).long()
        # garante x2 > x1, y2 > y1
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        for i in range(M):
            bi = img_idx[i].item()
            region = grid[bi, y1[i]:y2[i], x1[i]:x2[i]]  # [h, w, 768]
            out[i] = region.reshape(-1, region.size(-1)).mean(dim=0)
        return out

    @staticmethod
    def compute_union_bbox(
        sub_bboxes: torch.Tensor,
        obj_bboxes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Retorna bbox uniao de cada par (sub, obj).

        Shapes
        ------
        sub_bboxes : [P, 4] (x1, y1, x2, y2) em coords de grade
        obj_bboxes : [P, 4] (x1, y1, x2, y2) em coords de grade

        Returns
        -------
        union_bboxes : [P, 4] = [min(x1), min(y1), max(x2), max(y2)]
        """
        x1 = torch.min(sub_bboxes[:, 0], obj_bboxes[:, 0])
        y1 = torch.min(sub_bboxes[:, 1], obj_bboxes[:, 1])
        x2 = torch.max(sub_bboxes[:, 2], obj_bboxes[:, 2])
        y2 = torch.max(sub_bboxes[:, 3], obj_bboxes[:, 3])
        return torch.stack([x1, y1, x2, y2], dim=-1)  # [P, 4]

    @torch.no_grad()
    def predict(
        self,
        detections: list[Detection],
        visual_grid: torch.Tensor,
    ) -> list[dict]:
        """
        Prediz relacoes entre todos os pares ordenados de deteccoes.

        Pipeline (paridade com train_relation_head.forward_pairs):
          1. Empilha bboxes dos detections: [N, 4]
          2. Pares ordenados (i, j) com i != j: P = N*(N-1)
          3. sub_bbox = bboxes[i_idx], obj_bbox = bboxes[j_idx]
          4. union_bbox = compute_union_bbox(sub_bbox, obj_bbox)
          5. roi_mean_pool para sub/obj/union sobre grid: [P, 768] cada
          6. aligner.visual_proj: [P, 4096] cada
          7. RelationHead(sub_feat, obj_feat, union_feat) -> logits [P, vocab_pred]
          8. softmax, topk(max_edges_per_pair), threshold
          9. Retorna edges: [{source, target, relation, confidence}]

        Parameters
        ----------
        detections : list[Detection]
            Lista de deteccoes com .bbox_grid [4] em coords de grade 32x32.
        visual_grid : torch.Tensor
            [1, 32, 32, 768] — grid de features visuais DINOv3.

        Returns
        -------
        list[dict]
            Edges preditas. Cada dict tem:
              source     : int — indice do sujeito em detections
              target     : int — indice do objeto em detections
              relation   : str — predicado do pred_vocab
              confidence : float — probabilidade softmax

        Retorna lista vazia se < 2 deteccoes.
        """
        if len(detections) < 2:
            return []

        device = visual_grid.device
        head_dtype = next(self.relation_head.parameters()).dtype
        head_device = next(self.relation_head.parameters()).device

        n = len(detections)

        # 1. Empilha bboxes: [N, 4]
        bboxes = torch.stack([d.bbox_grid for d in detections]).to(device)
        img_idx = torch.zeros(n, dtype=torch.long, device=device)  # B=1

        # 2. Pares ordenados (i, j) com i != j
        pair_idx = [(i, j) for i in range(n) for j in range(n) if i != j]
        if not pair_idx:
            return []

        i_idx = torch.tensor([p[0] for p in pair_idx], device=device)
        j_idx = torch.tensor([p[1] for p in pair_idx], device=device)

        # 3. Sub/obj bboxes para cada par
        sub_bboxes = bboxes[i_idx]  # [P, 4]
        obj_bboxes = bboxes[j_idx]  # [P, 4]

        # 4. Union bbox
        union_bboxes = self.compute_union_bbox(sub_bboxes, obj_bboxes)  # [P, 4]

        # 5. ROI mean-pool: precisa de img_idx por roi
        p_count = len(pair_idx)
        pair_img_idx = torch.zeros(p_count, dtype=torch.long, device=device)

        sub_feat_768 = self._roi_mean_pool(visual_grid, sub_bboxes, pair_img_idx)    # [P, 768]
        obj_feat_768 = self._roi_mean_pool(visual_grid, obj_bboxes, pair_img_idx)    # [P, 768]
        union_feat_768 = self._roi_mean_pool(visual_grid, union_bboxes, pair_img_idx)  # [P, 768]

        # 6. Projecao para espaco texto (4096) via aligner congelado
        aligner_dtype = next(self.aligner.parameters()).dtype
        sub_feat = self.aligner.visual_proj(sub_feat_768.to(aligner_dtype))      # [P, 4096]
        obj_feat = self.aligner.visual_proj(obj_feat_768.to(aligner_dtype))      # [P, 4096]
        union_feat = self.aligner.visual_proj(union_feat_768.to(aligner_dtype))  # [P, 4096]

        # Move para dtype/device da head
        sub_feat = sub_feat.to(head_device, head_dtype)
        obj_feat = obj_feat.to(head_device, head_dtype)
        union_feat = union_feat.to(head_device, head_dtype)

        # 7. RelationHead
        logits = self.relation_head(sub_feat, obj_feat, union_feat)  # [P, vocab_pred]

        # 8. Softmax + topk + threshold
        probs = torch.softmax(logits.float(), dim=-1)  # [P, vocab_pred]
        vocab_pred = probs.size(-1)
        k = min(self.max_edges_per_pair, vocab_pred)
        top_conf, top_cls = probs.topk(k, dim=-1)  # [P, k]

        # 9. Montar edges
        edges: list[dict] = []
        for p_idx, (i, j) in enumerate(pair_idx):
            for r in range(k):
                conf = top_conf[p_idx, r].item()
                cls = top_cls[p_idx, r].item()
                if conf <= self.edge_threshold:
                    continue
                if cls >= len(self.pred_vocab):
                    continue
                edges.append({
                    "source": i,
                    "target": j,
                    "relation": self.pred_vocab[cls],
                    "confidence": float(conf),
                })

        return edges
