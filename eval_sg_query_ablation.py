"""
eval_sg_query_ablation.py
--------------------------
Avaliacao de retrieval + scene graph + re-ranking para uma unica query
em 3 granularidades (completa, metade, termo dominante) sobre todos os
aligners treinados.

Executa sem parar: termina um aligner, passa para o proximo.

Aligners avaliados
------------------
  1. noup       — best_aligner_no_up.pth  (DINO sem AnyUp, 196 patches)
  2. anyup      — best_aligner.pth        (DINO + AnyUp,  1024 patches)
  3. anyup_ats  — best_ats_aligner.pth    (AnyUp + ATS → 196 patches selecionados)
  4. anyup_pca  — best_pca_aligner.pth    (AnyUp + PCA → 196 patches selecionados)

Query
-----
  full     : "Pit bulls & other bully-types welcome at 780's dog motel."
  half     : primeiros len//2 caracteres
  dominant : "Dogs"

Fluxo otimizado de VRAM
------------------------
  Phase 1 — DINO index (2 cargas: noup + anyup)
  Phase 2 — Qwen encode (1 carga: 3 queries)
  Phase 3 — Retrieval + SGs (1 carga de RelTR, 4 trocas de aligner)
  Phase 4 — Re-ranking semantico (1 carga de Qwen)
  Phase 5 — Sumario + vizs

Uso:
    python eval_sg_query_ablation.py
    python eval_sg_query_ablation.py --top_k 10 --image_dir F:/COYO/coyo/extracted/00000
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset_withSSD
from eval_retrieval_with_h5 import calcular_recall_bidirecional
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.rerank.semantic_rerank import ReRankResult, SemanticReRanker
from models.SG.reltr_wrapper import RelTRWrapper, SceneGraph
from train_aligner_ats_h5 import AdaptiveTokenSampler
from train_aligner_pca_h5 import PCAVarianceSampler
from utils.io_utils import ensure_dir
from viz.scene_graph_viz import visualize_rerank_result, visualize_retrieval_result

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ── Query definitions ────────────────────────────────────────────────────────

QUERY_FULL = "Pit bulls & other bully-types welcome at 780’s dog motel."
QUERY_HALF = QUERY_FULL[: len(QUERY_FULL) // 2]
QUERY_DOMINANT = "Dogs"

QUERIES: List[Tuple[str, str]] = [
    ("full", QUERY_FULL),
    ("half", QUERY_HALF),
    ("dominant", QUERY_DOMINANT),
]


# ── Aligner configurations ───────────────────────────────────────────────────

ALIGNER_CONFIGS = [
    {
        "name": "noup",
        "ckpt": "checkpoints/best_aligner_no_up.pth",
        "upsampler": "none",
        "sampler_type": None,
        "ckpt_format": "plain",
    },
    {
        "name": "anyup",
        "ckpt": "checkpoints/best_aligner.pth",
        "upsampler": "anyup",
        "sampler_type": None,
        "ckpt_format": "plain",
    },
    {
        "name": "anyup_ats",
        "ckpt": "checkpoints/best_ats_aligner.pth",
        "upsampler": "anyup",
        "sampler_type": "ats",
        "ckpt_format": "module_dict",
    },
    {
        "name": "anyup_pca",
        "ckpt": "checkpoints/best_pca_aligner.pth",
        "upsampler": "anyup",
        "sampler_type": "pca",
        "ckpt_format": "plain",
    },
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _offload(model) -> None:
    target = getattr(model, "torch_hub", model)
    if hasattr(target, "to"):
        target.to("cpu")
    torch.cuda.empty_cache()


def _timer(label: str):
    """Context manager para cronometrar fases."""
    class _T:
        def __enter__(self):
            self.t0 = time.time()
            print(f"\n{'='*60}")
            print(f"  {label}")
            print(f"{'='*60}")
            return self
        def __exit__(self, *_):
            elapsed = time.time() - self.t0
            print(f"  [{label}] concluido em {elapsed:.0f}s")
    return _T()


_image_cache: Dict[str, Image.Image] = {}

def _get_image(path: str) -> Image.Image:
    if path not in _image_cache:
        _image_cache[path] = Image.open(path).convert("RGB")
    return _image_cache[path]


# ── Ground truth ──────────────────────────────────────────────────────────────

def _normalize_quotes(s: str) -> str:
    """Normaliza aspas curvas (Unicode) para ASCII reto para comparacao."""
    return s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')


def find_ground_truth(
    image_dir: str,
    query: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> str | None:
    """Busca no diretorio a imagem cuja caption (.txt) corresponde exatamente a query."""
    q_norm = _normalize_quotes(query.strip())
    for txt_path in Path(image_dir).rglob("*.txt"):
        try:
            caption = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
            if _normalize_quotes(caption) == q_norm:
                for ext in extensions:
                    img_path = txt_path.with_suffix(ext)
                    if img_path.exists():
                        return str(img_path)
        except Exception:
            continue
    return None


# ── DINO indexing ─────────────────────────────────────────────────────────────

@torch.no_grad()
def index_images(
    image_dir: str,
    device: str,
    upsampler: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Tuple[List[str], torch.Tensor]:
    """
    Indexa imagens com DINO (com ou sem AnyUp).

    Returns
    -------
    paths : list[str]
    Vraw  : [N, P, 768]  (P=196 se noup, P=1024 se anyup)
    """
    paths = sorted(
        str(p) for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in extensions
    )
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {image_dir}")

    dino = DinoSceneEncoder(device=device, upsampler=upsampler)
    all_v: List[torch.Tensor] = []

    for path in tqdm(paths, desc=f"Indexando [{upsampler}]"):
        img = _get_image(path)
        img_t = _IMG_TRANSFORM(img).unsqueeze(0).to(device)

        _, feat = dino.extract_features(img_t)

        if upsampler != "none":
            feat = F.adaptive_avg_pool2d(feat, (32, 32))

        B, C, H, W = feat.shape
        vis = feat.view(B, C, -1).transpose(1, 2).to(torch.bfloat16).cpu()
        all_v.append(vis)
        torch.cuda.empty_cache()

    _offload(dino.model)
    if hasattr(dino, "upsampler") and dino.upsampler is not None:
        _offload(dino.upsampler)
    del dino
    torch.cuda.empty_cache()

    Vraw = torch.cat(all_v, dim=0)
    print(f"  Vraw shape: {Vraw.shape}  ({upsampler})")
    return paths, Vraw


# ── Query encoding ────────────────────────────────────────────────────────────

@torch.no_grad()
def encode_queries(
    queries: List[Tuple[str, str]],
    device: str,
) -> Dict[str, torch.Tensor]:
    """Codifica todas as variantes da query via Qwen3. Retorna {name: [1,1,4096]}."""
    qwen = QwenSceneEmbedder(device=device)
    q_embs: Dict[str, torch.Tensor] = {}

    for name, text in queries:
        emb = qwen.embed_components([[text]], normalize=False)  # [1, 1, 4096]
        q_embs[name] = emb.cpu()
        print(f"  query '{name}' ({len(text)} chars): \"{text}\"")

    _offload(qwen.model)
    del qwen
    torch.cuda.empty_cache()
    return q_embs


# ── Aligner loading ──────────────────────────────────────────────────────────

def load_aligner(
    config: dict,
    device: str,
) -> Tuple[LoRACrossAttentionAligner, nn.Module | None]:
    """Carrega aligner (+sampler se necessario). Retorna (aligner, sampler)."""
    dtype = torch.bfloat16

    if config["ckpt_format"] == "module_dict":
        sampler = AdaptiveTokenSampler(visual_dim=768, text_dim=4096, k=196)
        aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
        combined = nn.ModuleDict({"sampler": sampler, "aligner": aligner})
        combined.load_state_dict(torch.load(config["ckpt"], map_location=device))
        combined.to(device).to(dtype).eval()
        print(f"  [{config['name']}] ModuleDict carregado de {config['ckpt']}")
        return aligner, sampler

    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    aligner.load_state_dict(torch.load(config["ckpt"], map_location=device))
    aligner.to(device).to(dtype).eval()
    print(f"  [{config['name']}] Aligner carregado de {config['ckpt']}")

    sampler = None
    if config["sampler_type"] == "pca":
        sampler = PCAVarianceSampler(k=196, n_components=64)
        print(f"  [{config['name']}] PCAVarianceSampler instanciado (param-free)")

    return aligner, sampler


# ── Retrieval ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def retrieve_top_k(
    query_emb: torch.Tensor,
    Vraw: torch.Tensor,
    paths: List[str],
    aligner: LoRACrossAttentionAligner,
    device: str,
    top_k: int,
    sampler: nn.Module | None = None,
    sampler_type: str | None = None,
    chunk_size: int = 32,
) -> List[Tuple[str, float]]:
    """
    Retrieval com aligner (+sampler opcional).

    Para cada chunk de imagens, aplica:
      - nenhum sampler:  aligner(patches, query)
      - ATS:             sampler(patches, query) → aligner(selected, query)
      - PCA:             sampler(patches) → aligner(selected, query)
    """
    dtype = torch.bfloat16
    N = Vraw.shape[0]
    scores = torch.zeros(N)

    q = query_emb.to(device).to(dtype)
    q_norm = F.normalize(q.squeeze(1), p=2, dim=-1).float().cpu()

    for start in tqdm(range(0, N, chunk_size), desc="Scoring", leave=False):
        end = min(start + chunk_size, N)
        B = end - start

        v_batch = Vraw[start:end].to(device).to(dtype)
        q_batch = q.expand(B, -1, -1)

        if sampler_type == "ats":
            selected, _, _ = sampler(v_batch, q_batch)
            attn_out, _, _ = aligner(selected, q_batch)
        elif sampler_type == "pca":
            selected, _, _ = sampler(v_batch)
            attn_out, _, _ = aligner(selected.to(dtype), q_batch)
        else:
            attn_out, _, _ = aligner(v_batch, q_batch)

        v_norm = F.normalize(attn_out.squeeze(1), p=2, dim=-1).float().cpu()
        scores[start:end] = (v_norm * q_norm).sum(-1)

        del v_batch, attn_out, v_norm
        torch.cuda.empty_cache()

    k = min(top_k, len(paths))
    top_scores, top_idx = scores.topk(k)
    return [(paths[i], top_scores[j].item()) for j, i in enumerate(top_idx.tolist())]


# ── Rank lookup ───────────────────────────────────────────────────────────────

def _find_rank(results_or_rr, gt_path: str) -> int | None:
    """Encontra rank (1-based) do ground truth. None se ausente."""
    if not gt_path:
        return None
    for i, item in enumerate(results_or_rr):
        path = item[0] if isinstance(item, tuple) else item.path
        if os.path.abspath(path) == os.path.abspath(gt_path):
            return i + 1
    return None


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(summary: dict, top_k: int) -> None:
    """Imprime tabela de resultados."""
    hdr = f"{'Aligner':<14}"
    for q_name, _ in QUERIES:
        hdr += f" | {q_name:^20}"
    print(f"\n{'='*78}")
    print(hdr)
    print(f"{'='*78}")

    for cfg in ALIGNER_CONFIGS:
        name = cfg["name"]
        if name not in summary:
            continue
        row = f"{name:<14}"
        for q_name, _ in QUERIES:
            m = summary[name].get(q_name, {})
            rd = m.get("rank_dense")
            rr = m.get("rank_reranked")
            sd = f"D:{rd}" if rd else f"D:>{top_k}"
            sr = f"R:{rr}" if rr else f"R:>{top_k}"
            if rd and rr:
                delta = rd - rr
                sd += f" {sr} Δ{'+' if delta > 0 else ''}{delta}"
            else:
                sd += f" {sr}"
            row += f" | {sd:^20}"
        print(row)
    print(f"{'='*78}")
    print("D = rank denso, R = rank re-ranked, Δ = melhoria de posicao\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    device = args.device
    top_k = args.top_k

    print(f"[config] device={device}, top_k={top_k}")
    print(f"[config] image_dir={args.image_dir}")
    print(f"\n[queries]")
    for name, text in QUERIES:
        print(f"  {name:>10} ({len(text):2d} chars): \"{text}\"")

    # ── Ground truth ──────────────────────────────────────────────────────
    gt_path = find_ground_truth(args.image_dir, QUERY_FULL)
    if gt_path:
        print(f"\n[ground truth] {gt_path}")
    else:
        print("\n[WARN] Ground truth nao encontrado no diretorio — ranks serao None")

    # ── Phase 1: DINO indexing ────────────────────────────────────────────
    vraw_cache: Dict[str, Tuple[List[str], torch.Tensor]] = {}

    with _timer("Phase 1a: DINO index (noup)"):
        paths, Vraw_noup = index_images(args.image_dir, device, upsampler="none")
        vraw_cache["none"] = (paths, Vraw_noup)

    with _timer("Phase 1b: DINO index (anyup)"):
        _, Vraw_anyup = index_images(args.image_dir, device, upsampler="anyup")
        vraw_cache["anyup"] = (paths, Vraw_anyup)

    # ── Phase 2: Qwen encode queries ─────────────────────────────────────
    with _timer("Phase 2: Qwen encode queries"):
        q_embs = encode_queries(QUERIES, device)

    # ── Phase 3: Retrieval + Scene Graphs ─────────────────────────────────
    # RelTR fica carregado durante toda esta fase (cabe com aligner na VRAM)
    all_data: Dict[str, Dict[str, dict]] = {}

    with _timer("Phase 3: Retrieval + Scene Graphs"):
        reltr = RelTRWrapper(
            checkpoint=args.reltr_ckpt,
            reltr_repo=args.reltr_repo,
            device=device,
            num_queries=args.num_queries,
            threshold=args.threshold,
        )

        for cfg in ALIGNER_CONFIGS:
            name = cfg["name"]
            print(f"\n--- Aligner: {name} ---")

            aligner, sampler = load_aligner(cfg, device)
            cur_paths, cur_Vraw = vraw_cache[cfg["upsampler"]]
            all_data[name] = {}

            for q_name, q_text in QUERIES:
                print(f"  Query: {q_name}")
                results = retrieve_top_k(
                    q_embs[q_name], cur_Vraw, cur_paths, aligner, device,
                    top_k, sampler=sampler, sampler_type=cfg["sampler_type"],
                )

                sgs: List[SceneGraph] = []
                for img_path, _ in results:
                    sgs.append(reltr.predict(_get_image(img_path)))

                all_data[name][q_name] = {
                    "results": results,
                    "sgs": sgs,
                    "query_text": q_text,
                }

                # Viz pre-rerank
                out_dir = os.path.join(args.output_dir, name, q_name)
                for rank, ((img_path, score), sg) in enumerate(zip(results, sgs), 1):
                    save_path = os.path.join(out_dir, f"rank{rank:02d}_pre_{os.path.basename(img_path)}.png")
                    visualize_retrieval_result(
                        image=_get_image(img_path), sg=sg,
                        query_text=q_text, retrieval_score=score,
                        save_path=save_path,
                        highlight_nodes=q_text.lower().split(),
                    )

            _offload(aligner)
            if sampler is not None and hasattr(sampler, "to"):
                _offload(sampler)
            del aligner, sampler
            torch.cuda.empty_cache()

        _offload(reltr.model)
        del reltr
        torch.cuda.empty_cache()

    # ── Phase 4: Re-ranking semantico ─────────────────────────────────────
    summary: Dict[str, Dict[str, dict]] = {}

    with _timer("Phase 4: Re-ranking semantico (Qwen)"):
        qwen = QwenSceneEmbedder(device=device)
        reranker = SemanticReRanker(
            qwen=qwen,
            alpha=args.rerank_alpha,
            pooling=args.rerank_pool,
            top_n_pool=args.rerank_topn,
            explain_top_k=args.explain_top_k,
        )

        for cfg in ALIGNER_CONFIGS:
            name = cfg["name"]
            summary[name] = {}

            for q_name, q_text in QUERIES:
                data = all_data[name][q_name]
                candidates = [
                    (p, s, sg)
                    for (p, s), sg in zip(data["results"], data["sgs"])
                ]
                rr_results = reranker.rerank(q_embs[q_name], candidates)

                # Viz pos-rerank
                out_dir = os.path.join(args.output_dir, name, q_name)
                for r in rr_results:
                    save_path = os.path.join(
                        out_dir,
                        f"rerank{r.rank_after:02d}_{os.path.basename(r.path)}.png",
                    )
                    visualize_rerank_result(
                        image=_get_image(r.path),
                        result=r,
                        query_text=q_text,
                        save_path=save_path,
                        alpha=args.rerank_alpha,
                    )

                rank_dense = _find_rank(data["results"], gt_path)
                rank_rr = _find_rank(rr_results, gt_path)

                summary[name][q_name] = {
                    "rank_dense": rank_dense,
                    "rank_reranked": rank_rr,
                    "top_k_paths_dense": [os.path.basename(p) for p, _ in data["results"]],
                    "top_k_paths_reranked": [os.path.basename(r.path) for r in rr_results],
                    "scores_dense": [s for _, s in data["results"]],
                    "scores_reranked": [
                        {"dense": r.score_dense, "sg": r.score_sg, "final": r.score_final}
                        for r in rr_results
                    ],
                    "top_triples": [
                        [
                            {"triple": f"{t.subject} {t.predicate} {t.object}", "sim": sim}
                            for t, sim in r.top_triples
                        ]
                        for r in rr_results
                    ],
                }

                print(
                    f"  [{name}/{q_name}] "
                    f"rank_dense={rank_dense or f'>{top_k}'}  "
                    f"rank_reranked={rank_rr or f'>{top_k}'}"
                )

        _offload(qwen.model)
        del qwen, reranker
        torch.cuda.empty_cache()

    # ── Phase 5: Summary ──────────────────────────────────────────────────
    print_summary(summary, top_k)

    full_summary = {
        "queries": {name: text for name, text in QUERIES},
        "gt_image": os.path.basename(gt_path) if gt_path else None,
        "top_k": top_k,
        "rerank_alpha": args.rerank_alpha,
        "results": summary,
    }
    out_json = os.path.join(args.output_dir, "summary.json")
    ensure_dir(args.output_dir)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_summary, f, indent=2, ensure_ascii=False)
    print(f"[done] Summary salvo em {out_json}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Retrieval + SG + Re-ranking — ablation de query × aligner"
    )
    parser.add_argument("--image_dir", default="F:/COYO/coyo/extracted/00000")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", default="results/sg_query_ablation")
    parser.add_argument("--threshold", type=float, default=0.3, help="Score minimo RelTR")
    parser.add_argument("--num_queries", type=int, default=20, help="Triplas RelTR por imagem")
    parser.add_argument("--reltr_ckpt", default="checkpoints/reltr/reltr_vg.pth")
    parser.add_argument("--reltr_repo", default="reltr_repo")
    parser.add_argument("--rerank_alpha", type=float, default=0.5)
    parser.add_argument("--rerank_pool", default="max", choices=["max", "topn_mean"])
    parser.add_argument("--rerank_topn", type=int, default=3)
    parser.add_argument("--explain_top_k", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)
