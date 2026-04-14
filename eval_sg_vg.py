"""
eval_sg_vg.py
--------------
SGGen Recall@K benchmark no Visual Genome (VG-150).

Metricas padrao da literatura de Scene Graph Generation:
  - Recall@K (R@20, R@50, R@100)
  - Mean Recall@K (mR@20, mR@50, mR@100)

Referencia:
  Xu et al., "Scene Graph Generation by Iterative Message Passing", CVPR 2017
  Zellers et al., "Neural Motifs", CVPR 2018
  Tang et al., "Unbiased Scene Graph Generation", CVPR 2020

Uso:
    python eval_sg_vg.py --vg-dir G:/vg --checkpoint checkpoints/best_aligner.pth
    python eval_sg_vg.py --vg-dir G:/vg --checkpoint checkpoints/best_aligner.pth --max-samples 50
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime

import torch
from PIL import Image
from tqdm import tqdm

from data.data_utils_pytorch import get_transforms
from data.vg_dataset import (
    build_vg150_vocab,
    get_image_path,
    load_scene_graphs,
)
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.classifier_head import SGClassifierHead
from models.SG.generation import SceneGraphGenerator


# ── GT Extraction ───────────────────────────────────────────────────────


def extract_gt_triples(
    sg: dict,
    obj_vocab: set[str],
    pred_vocab: set[str],
) -> set[tuple[str, str, str]]:
    """
    Extrai tripletas GT (sub, pred, obj) filtradas pelo vocabulario VG-150.

    Tripletas cujos labels nao estao no vocabulario sao descartadas,
    mantendo consistencia com o que o modelo pode prever.
    """
    # Mapeia object_id -> nome (apenas labels no vocabulario)
    id_to_name: dict[int, str] = {}
    for obj in sg.get("objects", []):
        names = obj.get("names") or [obj.get("name", "")]
        name = names[0].lower().strip() if names else ""
        if name in obj_vocab:
            id_to_name[obj["object_id"]] = name

    triples: set[tuple[str, str, str]] = set()
    for rel in sg.get("relationships", []):
        sub_id = rel.get("subject_id")
        obj_id = rel.get("object_id")
        pred = rel.get("predicate", "").lower().strip()

        if sub_id in id_to_name and obj_id in id_to_name and pred in pred_vocab:
            triples.add((id_to_name[sub_id], pred, id_to_name[obj_id]))

    return triples


# ── Prediction Extraction ──────────────────────────────────────────────


def extract_pred_triples(
    sg_result: dict,
) -> list[tuple[tuple[str, str, str], float]]:
    """
    Extrai tripletas preditas com score de confianca combinado.

    Confianca = score_sub * confidence_edge * score_obj
    (padrao em SGGen: combinar confiancas de deteccao e classificacao)

    Retorna lista ordenada por confianca decrescente.
    """
    nodes = {n["id"]: n for n in sg_result.get("nodes", [])}

    triples: list[tuple[tuple[str, str, str], float]] = []
    for edge in sg_result.get("edges", []):
        src = nodes.get(edge["source"])
        tgt = nodes.get(edge["target"])
        if src is None or tgt is None:
            continue

        conf = src["score"] * edge["confidence"] * tgt["score"]
        triple = (
            src["label"].lower().strip(),
            edge["relation"].lower().strip(),
            tgt["label"].lower().strip(),
        )
        triples.append((triple, conf))

    triples.sort(key=lambda x: x[1], reverse=True)
    return triples


# ── Recall@K ────────────────────────────────────────────────────────────


def recall_at_k(
    pred_triples: list[tuple[tuple[str, str, str], float]],
    gt_triples: set[tuple[str, str, str]],
    k: int,
) -> float:
    """
    Recall@K: fracao das tripletas GT encontradas nas top-K predicoes.

    Parameters
    ----------
    pred_triples:
        Predicoes ordenadas por confianca decrescente.
    gt_triples:
        Conjunto de tripletas ground-truth.
    k:
        Numero de predicoes a considerar.
    """
    if not gt_triples:
        return 0.0
    top_k = {t for t, _ in pred_triples[:k]}
    return len(gt_triples.intersection(top_k)) / len(gt_triples)


def per_predicate_recall(
    pred_triples: list[tuple[tuple[str, str, str], float]],
    gt_triples: set[tuple[str, str, str]],
    k: int,
) -> dict[str, float | None]:
    """
    Recall@K por classe de predicado (para calculo de mR@K).

    Retorna dict predicado -> recall (None se predicado ausente no GT).
    """
    # Agrupa GT por predicado
    gt_by_pred: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
    for sub, pred, obj in gt_triples:
        gt_by_pred[pred].add((sub, pred, obj))

    top_k = {t for t, _ in pred_triples[:k]}

    results: dict[str, float | None] = {}
    for pred, pred_gt in gt_by_pred.items():
        matched = pred_gt.intersection(top_k)
        results[pred] = len(matched) / len(pred_gt)

    return results


# ── Main ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SGGen Recall@K — Visual Genome")
    parser.add_argument("--vg-dir", type=str, required=True, help="Diretorio com dados do VG")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_aligner.pth")
    parser.add_argument("--max-samples", type=int, default=1000, help="Limite de amostras de teste")
    parser.add_argument("--node-threshold", type=float, default=0.01, help="Threshold baixo para manter candidatos")
    parser.add_argument("--edge-threshold", type=float, default=0.0, help="Threshold de arestas (0.0 = manter todas para ranking)")
    parser.add_argument("--seed", type=int, default=42, help="Seed para split train/test")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fracao de teste")
    parser.add_argument("--sg-head-checkpoint", type=str, default=None,
                        help="Caminho para checkpoint da SGClassifierHead. Se fornecido, substitui o scoring por retrieval no passo de deteccao de nos.")
    parser.add_argument("--min-patches", type=int, default=2,
                        help="Tamanho minimo (em patches 32x32) para um componente virar um no")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"  VRAM: {free:.1f}/{total:.1f} GB livre")

    # ── 1. Carregar dados VG ────────────────────────────────────────────
    all_sgs = load_scene_graphs(args.vg_dir)
    obj_list, pred_list, obj_set, pred_set = build_vg150_vocab(all_sgs)

    # ── 2. Split deterministico ─────────────────────────────────────────
    import random
    rng = random.Random(args.seed)
    indices = list(range(len(all_sgs)))
    rng.shuffle(indices)

    test_start = int(len(indices) * (1 - args.test_ratio))
    test_indices = indices[test_start:]
    print(f"\nSplit: {test_start} train/val, {len(test_indices)} test")

    # Filtra imagens de teste que tem GT valido e imagem no disco
    test_samples: list[dict] = []
    for idx in tqdm(test_indices, desc="Filtrando teste"):
        sg = all_sgs[idx]
        gt = extract_gt_triples(sg, obj_set, pred_set)
        if not gt:
            continue
        img_path = get_image_path(args.vg_dir, sg["image_id"])
        if img_path is None:
            continue
        test_samples.append({"sg": sg, "gt": gt, "img_path": img_path})
        if len(test_samples) >= args.max_samples:
            break

    print(f"Amostras de teste validas: {len(test_samples)}")
    if not test_samples:
        print("Nenhuma amostra valida. Verifique se as imagens foram baixadas.")
        return

    # ── 3. Carregar modelos ─────────────────────────────────────────────
    print("\nCarregando modelos...")
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    if os.path.exists(args.checkpoint):
        aligner.load_state_dict(
            torch.load(args.checkpoint, map_location=device), strict=False
        )
        print(f"  Checkpoint: {args.checkpoint}")
    else:
        print(f"  [AVISO] Checkpoint nao encontrado: {args.checkpoint}")
    aligner.to(device).to(torch.bfloat16).eval()

    dino = DinoSceneEncoder(device=device)
    qwen = QwenSceneEmbedder(device=device)

    sg_head = None
    if args.sg_head_checkpoint:
        if not os.path.exists(args.sg_head_checkpoint):
            raise FileNotFoundError(f"SG head checkpoint nao encontrado: {args.sg_head_checkpoint}")
        sg_head = SGClassifierHead(
            visual_dim=768,
            vocab_size=len(obj_list),
        )
        state = torch.load(args.sg_head_checkpoint, map_location=device)
        sg_head.load_state_dict(state, strict=True)
        sg_head.to(device).to(torch.bfloat16).eval()
        print(f"  SG head carregado: {args.sg_head_checkpoint}")

    generator = SceneGraphGenerator(
        dino_encoder=dino,
        qwen_embedder=qwen,
        aligner=aligner,
        threshold=args.node_threshold,
        edge_threshold=args.edge_threshold,
        sg_classifier_head=sg_head,
        min_patches=args.min_patches,
    )

    torch.cuda.empty_cache()

    # ── 4. Transform de imagem ──────────────────────────────────────────
    transform = get_transforms(256)

    # ── 5. Avaliacao SGGen ──────────────────────────────────────────────
    K_VALUES = [20, 50, 100]

    all_recalls: dict[int, list[float]] = {k: [] for k in K_VALUES}
    # Para mR@K: acumula recalls por predicado, por K
    per_pred_accum: dict[int, dict[str, list[float]]] = {
        k: defaultdict(list) for k in K_VALUES
    }
    n_gt_triples_total = 0
    n_pred_triples_total = 0

    print(f"\nIniciando SGGen Recall@K ({len(test_samples)} amostras)...")
    print(f"  Candidatos: {len(obj_list)} objetos, {len(pred_list)} predicados")
    print(f"  Node threshold: {args.node_threshold}, Edge threshold: {args.edge_threshold}")

    try:
        for i, sample in enumerate(
            tqdm(test_samples, desc="SGGen Eval")
        ):
            img = Image.open(sample["img_path"]).convert("RGB")
            img_tensor = transform(img)

            sg_result = generator.generate(img_tensor, obj_list, pred_list)
            pred_triples = extract_pred_triples(sg_result)
            gt = sample["gt"]

            n_gt_triples_total += len(gt)
            n_pred_triples_total += len(pred_triples)

            for k in K_VALUES:
                r = recall_at_k(pred_triples, gt, k)
                all_recalls[k].append(r)

                # Per-predicate recall para mR@K
                pp = per_predicate_recall(pred_triples, gt, k)
                for pred, val in pp.items():
                    if val is not None:
                        per_pred_accum[k][pred].append(val)

            # Log periodico
            if (i + 1) % 50 == 0:
                r50_so_far = sum(all_recalls[50]) / len(all_recalls[50])
                nodes_avg = sum(len(s["sg"].get("nodes", [])) for s in test_samples[:i+1]) / (i + 1)
                print(
                    f"  [{i+1}/{len(test_samples)}] "
                    f"R@50={r50_so_far:.4f}  "
                    f"GT triples/img={n_gt_triples_total/(i+1):.1f}  "
                    f"Pred triples/img={n_pred_triples_total/(i+1):.1f}"
                )

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise

    # ── 6. Resultados ───────────────────────────────────────────────────
    def safe_mean(vals: list[float]) -> float:
        return sum(vals) / len(vals) if vals else 0.0

    # Recall@K
    recall_results: dict[str, float] = {}
    for k in K_VALUES:
        recall_results[f"R@{k}"] = safe_mean(all_recalls[k])

    # Mean Recall@K (mR@K) — media das medias por predicado
    mean_recall_results: dict[str, float] = {}
    for k in K_VALUES:
        pred_means = []
        for pred, vals in per_pred_accum[k].items():
            if vals:
                pred_means.append(safe_mean(vals))
        mean_recall_results[f"mR@{k}"] = safe_mean(pred_means)

    print("\n" + "=" * 60)
    print("SGGen Recall@K — Visual Genome (VG-150)")
    print("=" * 60)
    print(f"  Amostras avaliadas : {len(test_samples)}")
    print(f"  GT triples/img    : {n_gt_triples_total / len(test_samples):.1f}")
    print(f"  Pred triples/img  : {n_pred_triples_total / len(test_samples):.1f}")
    print(f"  Predicados c/ GT  : {len(per_pred_accum[50])}")
    print()
    print("  Recall@K:")
    for k in K_VALUES:
        print(f"    R@{k:<3d} = {recall_results[f'R@{k}']:.4f}")
    print()
    print("  Mean Recall@K:")
    for k in K_VALUES:
        print(f"    mR@{k:<3d} = {mean_recall_results[f'mR@{k}']:.4f}")
    print("=" * 60)

    # ── 7. Salvar resultados ────────────────────────────────────────────
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, f"sggen_vg150_results_{timestamp_tag}.json")

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark": "Visual Genome VG-150",
            "num_samples": len(test_samples),
            "node_threshold": args.node_threshold,
            "edge_threshold": args.edge_threshold,
            "checkpoint": args.checkpoint,
            "sg_head_checkpoint": args.sg_head_checkpoint,
            "vocab_objects": len(obj_list),
            "vocab_predicates": len(pred_list),
            "avg_gt_triples": n_gt_triples_total / len(test_samples),
            "avg_pred_triples": n_pred_triples_total / len(test_samples),
        },
        "recall": recall_results,
        "mean_recall": mean_recall_results,
        "per_predicate_recall_at_50": {
            pred: safe_mean(vals)
            for pred, vals in sorted(per_pred_accum[50].items())
            if vals
        },
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"\nResultados salvos em: {output_path}")


if __name__ == "__main__":
    main()
