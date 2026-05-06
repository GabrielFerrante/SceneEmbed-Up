"""
eval_retrieval_sg.py
--------------------
Retrieval COYO com geração de scene graphs (RelTR) para as top-K imagens
recuperadas e visualização com explicabilidade.

Fluxo:
  1. Carrega aligner COYO pré-treinado
  2. Recebe uma query textual
  3. Recupera top-K imagens mais similares
  4. Para cada imagem recuperada: gera scene graph via RelTR
  5. Salva visualização (imagem + grafo) em results/sg_viz/

Uso:
    python eval_retrieval_sg.py --query "Pit bulls & other bully-types welcome at 780’s dog motel." --top_k 5
    python eval_retrieval_sg.py --query "person riding bike" --top_k 3 --threshold 0.2
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.reltr_wrapper import RelTRWrapper, SceneGraph
from models.rerank.semantic_rerank import SemanticReRanker
from viz.scene_graph_viz import visualize_retrieval_result, visualize_rerank_result


def encode_query(
    query: str,
    qwen: QwenSceneEmbedder,
) -> torch.Tensor:
    """
    Codifica a query — formato igual ao text_queries do treino: [1, 1, 4096],
    NÃO normalizado (a normalização acontece no retrieve_top_k).
    """
    t_emb = qwen.embed_components([[query]], normalize=False)  # [1, 1, 4096]
    return t_emb.cpu()


def index_images(
    image_dir: str,
    dino: DinoSceneEncoder,
    device: str,
    extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp"),
) -> Tuple[List[str], torch.Tensor]:
    """
    Extrai patches visuais brutos do DINO para todas as imagens.

    Returns
    -------
    paths:
        Lista de caminhos de imagem.
    Vraw:
        Tensor `[N, 1024, 768]` de patches visuais (CPU, bfloat16) — entrada bruta
        para o aligner. NÃO passa pelo aligner aqui (será feito no retrieval).
    """
    paths = [
        str(p) for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in extensions
    ]
    if not paths:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {image_dir}")

    print(f"[index] {len(paths)} imagens encontradas")
    all_v: List[torch.Tensor] = []

    for path in tqdm(paths, desc="Indexando imagens"):
        img = Image.open(path).convert("RGB")
        img_t = _IMG_TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            _, hr_feat = dino.extract_features(img_t)
            hr_feat = F.adaptive_avg_pool2d(hr_feat, (32, 32))
            B, C, H, W = hr_feat.shape  # noqa: F841
            vis = hr_feat.view(B, C, -1).transpose(1, 2).to(torch.bfloat16).cpu()  # [1, 1024, 768]
        all_v.append(vis)
        torch.cuda.empty_cache()

    Vraw = torch.cat(all_v, dim=0)  # [N, 1024, 768]
    return paths, Vraw


def retrieve_top_k(
    query_emb: torch.Tensor,        # [1, 1, 4096] no espaço Qwen (não normalizado)
    Vraw: torch.Tensor,             # [N, 1024, 768] patches visuais
    paths: List[str],
    aligner: LoRACrossAttentionAligner,
    device: str,
    top_k: int,
    chunk_size: int = 32,
) -> List[Tuple[str, float]]:
    """
    Para cada imagem, roda o aligner usando a query como text_queries:
        attn_output = aligner(visual_patches, query_text)

    Compara attn_output (visual refinado pela query) contra a query normalizada.
    Igual ao protocolo do treino e do eval_retrieval_with_h5.py.
    """
    aligner.eval()
    N = Vraw.shape[0]
    scores = torch.zeros(N)

    q = query_emb.to(device).to(torch.bfloat16)                  # [1, 1, 4096]
    q_norm = F.normalize(q.squeeze(1), p=2, dim=-1).float().cpu()  # [1, 4096]

    with torch.no_grad():
        for start in tqdm(range(0, N, chunk_size), desc="Scoring"):
            end = min(start + chunk_size, N)
            B = end - start

            v_batch = Vraw[start:end].to(device)                 # [B, 1024, 768]
            q_batch = q.expand(B, -1, -1)                        # [B, 1, 4096]

            attn_out, _, _ = aligner(v_batch, q_batch)           # [B, 1, 4096]
            v_norm = F.normalize(attn_out.squeeze(1), p=2, dim=-1).float().cpu()  # [B, 4096]

            scores[start:end] = (v_norm * q_norm).sum(-1)

            del v_batch, attn_out, v_norm
            torch.cuda.empty_cache()

    k = min(top_k, len(paths))
    top_scores, top_idx = scores.topk(k)
    return [(paths[i], top_scores[j].item()) for j, i in enumerate(top_idx.tolist())]


def _offload(model) -> None:
    """Move modelo para CPU e libera cache CUDA."""
    # AnyUpModel não é nn.Module — o modelo real está em .torch_hub
    target = getattr(model, "torch_hub", model)
    if hasattr(target, "to"):
        target.to("cpu")
    torch.cuda.empty_cache()


def main(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[device] {device}")

    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    if os.path.exists(args.aligner_ckpt):
        aligner.load_state_dict(
            torch.load(args.aligner_ckpt, map_location="cpu"), strict=False
        )
        print(f"  Aligner: {args.aligner_ckpt}")
    aligner.to(torch.bfloat16)

    # --- Fase 1: indexar patches visuais (só DINO na GPU) ---
    print("[load] DinoSceneEncoder...")
    dino = DinoSceneEncoder(device=device)

    paths, Vraw = index_images(args.image_dir, dino, device)  # [N, 1024, 768] CPU

    _offload(dino.model)
    _offload(dino.upsampler)
    del dino

    # --- Fase 2: codificar query (só Qwen) ---
    print("\n[load] QwenSceneEmbedder...")
    qwen = QwenSceneEmbedder(device=device)

    print(f'[query] "{args.query}"')
    with torch.no_grad():
        q_emb = encode_query(args.query, qwen)  # [1, 1, 4096]

    _offload(qwen.model)
    del qwen

    # --- Fase 3: retrieval com aligner (Aligner na GPU) ---
    aligner.to(device).eval()
    results = retrieve_top_k(q_emb, Vraw, paths, aligner, device, args.top_k)
    _offload(aligner)
    del aligner
    print(f"\n[retrieval] Top-{args.top_k} resultados:")
    for rank, (path, score) in enumerate(results, 1):
        print(f"  {rank}. {os.path.basename(path)}  score={score:.4f}")

    # --- Fase 4: scene graph (só RelTR na GPU) ---
    print("\n[load] RelTR...")
    reltr = RelTRWrapper(
        checkpoint=args.reltr_ckpt,
        reltr_repo=args.reltr_repo,
        device=device,
        num_queries=args.num_queries,
        threshold=args.threshold,
    )

    sgs: List[SceneGraph] = []
    images_cache = []  # cache pra evitar reler do disco na visualizacao
    for img_path, _ in tqdm(results, desc="Gerando scene graphs"):
        image = Image.open(img_path).convert("RGB")
        sgs.append(reltr.predict(image))
        images_cache.append(image)

    _offload(reltr.model)
    del reltr

    out_dir = os.path.join(args.output_dir, args.query.replace(" ", "_")[:40])

    if args.rerank:
        # --- Fase 5: Re-Ranking in Semantic Space ---
        print("\n[rerank] Re-ranking semantico via Qwen3 + scene graphs...")
        qwen = QwenSceneEmbedder(device=device)
        reranker = SemanticReRanker(
            qwen=qwen,
            alpha=args.rerank_alpha,
            pooling=args.rerank_pool,
            top_n_pool=args.rerank_topn,
            explain_top_k=args.explain_top_k,
        )

        candidates = [(p, s, sg) for (p, s), sg in zip(results, sgs)]
        rr_results = reranker.rerank(q_emb, candidates)

        _offload(qwen.model)
        del qwen

        print(f"\n[rerank] Top-{args.top_k} apos re-rank (alpha={args.rerank_alpha}):")
        for r in rr_results:
            delta = r.rank_before - r.rank_after
            arrow = "=" if delta == 0 else (f"+{delta}" if delta > 0 else f"{delta}")
            print(
                f"  #{r.rank_after}  (was #{r.rank_before}, {arrow})  "
                f"final={r.score_final:.4f}  dense={r.score_dense:.4f}  sg={r.score_sg:.4f}  "
                f"{os.path.basename(r.path)}"
            )

        path_to_image = {p: img for (p, _), img in zip(results, images_cache)}
        for r in rr_results:
            save_path = os.path.join(
                out_dir,
                f"rerank{r.rank_after:02d}_{os.path.basename(r.path)}.png",
            )
            visualize_rerank_result(
                image=path_to_image[r.path],
                result=r,
                query_text=args.query,
                save_path=save_path,
                alpha=args.rerank_alpha,
            )
    else:
        for rank, ((img_path, score), sg, image) in enumerate(zip(results, sgs, images_cache), 1):
            save_path = os.path.join(out_dir, f"rank{rank:02d}_{os.path.basename(img_path)}.png")
            visualize_retrieval_result(
                image=image,
                sg=sg,
                query_text=args.query,
                retrieval_score=score,
                save_path=save_path,
                highlight_nodes=args.query.lower().split(),
            )

    print(f"\n[done] Visualizações em {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval COYO + Scene Graph (RelTR)")
    parser.add_argument("--query",       required=True,  help="Query textual de retrieval")
    parser.add_argument("--image_dir",   default="F:/COYO/coyo/extracted/00000", help="Diretório com imagens COYO")
    parser.add_argument("--top_k",       type=int, default=5,  help="Número de imagens a recuperar")
    parser.add_argument("--threshold",   type=float, default=0.3, help="Score mínimo RelTR")
    parser.add_argument("--num_queries", type=int, default=20, help="Triplas RelTR por imagem")
    parser.add_argument("--aligner_ckpt", default="checkpoints/best_aligner.pth")
    parser.add_argument("--reltr_ckpt",   default="checkpoints/reltr/reltr_vg.pth")
    parser.add_argument("--reltr_repo",   default="reltr_repo")
    parser.add_argument("--output_dir",   default="results/sg_viz")
    parser.add_argument("--device",   default="cuda")
    # --- Re-Ranking in Semantic Space ---
    parser.add_argument("--rerank",       action="store_true",
                        help="Ativa re-ranking semantico via scene graphs + Qwen")
    parser.add_argument("--rerank_alpha", type=float, default=0.5,
                        help="Peso do score denso. final = alpha*dense + (1-alpha)*sg")
    parser.add_argument("--rerank_pool",  default="topn_mean", choices=["max", "topn_mean"],
                        help="Pooling das similaridades query-tripla")
    parser.add_argument("--rerank_topn",  type=int, default=3,
                        help="N para pooling topn_mean")
    parser.add_argument("--explain_top_k", type=int, default=3,
                        help="Triplas explicativas por imagem na visualizacao")
    args = parser.parse_args()
    main(args)
