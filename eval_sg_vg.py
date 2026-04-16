"""
eval_sg_vg.py
--------------
SGGen Recall@K benchmark no Visual Genome (VG-150).

Pipeline de 3 stages independentes:
  Stage 1: DetrDetector    — deteccao de objetos via DETR-R50 fine-tuned
  Stage 2: AttributeClassifier — classificacao de atributos (opcional)
  Stage 3: RelationPredictor   — predicao de relacoes entre pares

Metricas padrao da literatura de Scene Graph Generation:
  - Recall@K (R@20, R@50, R@100)
  - Mean Recall@K (mR@20, mR@50, mR@100)

Referencia:
  Xu et al., "Scene Graph Generation by Iterative Message Passing", CVPR 2017
  Zellers et al., "Neural Motifs", CVPR 2018
  Tang et al., "Unbiased Scene Graph Generation", CVPR 2020

Uso:
    python eval_sg_vg.py --vg-dir G:/vg \\
        --aligner checkpoints/best_aligner.pth \\
        --detr-checkpoint checkpoints/best_detr_vg150.pth \\
        --rel-head-checkpoint checkpoints/best_relation_head.pth

    # Com atributos (stage 2):
    python eval_sg_vg.py --vg-dir G:/vg \\
        --aligner checkpoints/best_aligner.pth \\
        --detr-checkpoint checkpoints/best_detr_vg150.pth \\
        --rel-head-checkpoint checkpoints/best_relation_head.pth \\
        --attr-head-checkpoint checkpoints/best_attribute_head.pth
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from datetime import datetime

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from data.data_utils_pytorch import get_transforms
from data.vg_dataset import (
    build_attribute_vocab,
    build_vg150_vocab,
    get_image_path,
    load_scene_graphs,
)
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.detectors.detr_detector import DetrDetector
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.SG.attribute_classifier import AttributeClassifier
from models.SG.attribute_head import AttributeHead
from models.SG.relation_head import RelationHead
from models.SG.relation_predictor import RelationPredictor


# ── Visual Feature Extraction ─────────────────────────────────────────


@torch.no_grad()
def extract_visual_grid(
    dino: DinoSceneEncoder,
    image_tensor: torch.Tensor,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Extrai grid de features visuais DINOv3 HR para uma imagem.

    Pipeline: DINO extract -> HR feat -> pool 32x32 -> reshape [1, 32, 32, 768].

    Parameters
    ----------
    dino : DinoSceneEncoder
    image_tensor : [3, H, W] normalizado (output de get_transforms)
    device : str
    dtype : torch.dtype

    Returns
    -------
    visual_grid : [1, 32, 32, 768]
    """
    images = image_tensor.unsqueeze(0).to(device)       # [1, 3, H, W]
    _, hr_feat = dino.extract_features(images)
    hr_feat_small = F.adaptive_avg_pool2d(hr_feat, (32, 32))  # [1, C, 32, 32]
    B, C, H, W = hr_feat_small.shape
    # [1, C, 32, 32] -> [1, 1024, 768] -> [1, 32, 32, 768]
    visual_input = hr_feat_small.reshape(B, C, H * W).transpose(1, 2)
    visual_grid = visual_input.reshape(1, 32, 32, C).to(dtype).contiguous()
    return visual_grid


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
    parser = argparse.ArgumentParser(
        description="SGGen Recall@K — Visual Genome (3-stage pipeline)"
    )
    # Data
    parser.add_argument("--vg-dir", type=str, required=True,
                        help="Diretorio com dados do VG")
    parser.add_argument("--max-samples", type=int, default=1000,
                        help="Limite de amostras de teste")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed para split train/test")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                        help="Fracao de teste")
    parser.add_argument("--image-size", type=int, default=256)

    # Vocab sizes (devem corresponder ao treino)
    parser.add_argument("--n-objects", type=int, default=150)
    parser.add_argument("--n-preds", type=int, default=50)
    parser.add_argument("--n-attrs", type=int, default=200)

    # Checkpoints
    parser.add_argument("--aligner", type=str, default="checkpoints/best_aligner.pth",
                        help="Checkpoint do LoRACrossAttentionAligner (congelado)")
    parser.add_argument("--detr-checkpoint", type=str, required=True,
                        help="Checkpoint do DetrDetector fine-tuned VG-150")
    parser.add_argument("--rel-head-checkpoint", type=str, required=True,
                        help="Checkpoint da RelationHead")
    parser.add_argument("--attr-head-checkpoint", type=str, default=None,
                        help="Checkpoint da AttributeHead (opcional; omitir pula stage 2)")

    # DETR
    parser.add_argument("--detr-score-threshold", type=float, default=0.5,
                        help="Confianca minima para deteccoes DETR")

    # RelationHead
    parser.add_argument("--edge-threshold", type=float, default=0.0,
                        help="Threshold de arestas (0.0 = manter todas para ranking)")
    parser.add_argument("--max-edges-per-pair", type=int, default=1,
                        help="Predicados por par (top-k) da RelationHead")
    parser.add_argument("--rel-head-proj-dim", type=int, default=512)
    parser.add_argument("--rel-head-hidden", type=int, default=1024)

    # AttributeClassifier
    parser.add_argument("--attr-threshold", type=float, default=0.3)
    parser.add_argument("--attr-top-k", type=int, default=5)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {autocast_dtype}")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"  VRAM: {free:.1f}/{total:.1f} GB livre")

    # ── 1. Carregar dados VG ────────────────────────────────────────────
    all_sgs = load_scene_graphs(args.vg_dir)
    obj_list, pred_list, obj_set, pred_set = build_vg150_vocab(
        all_sgs, n_objects=args.n_objects, n_predicates=args.n_preds
    )

    # ── 2. Split deterministico ─────────────────────────────────────────
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

    # Stage 1: DETR detector
    if not os.path.exists(args.detr_checkpoint):
        raise FileNotFoundError(
            f"DETR checkpoint nao encontrado: {args.detr_checkpoint}"
        )
    detector = DetrDetector(
        checkpoint_path=args.detr_checkpoint,
        vg150_labels=obj_list,
        score_threshold=args.detr_score_threshold,
        device=device,
    )
    print(f"  DetrDetector: {args.detr_checkpoint}")

    # DINO (congelado, para feature grid)
    dino = DinoSceneEncoder(device=device)
    for p in dino.model.parameters():
        p.requires_grad_(False)

    # Aligner (congelado, usa apenas visual_proj 768->4096)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    if os.path.exists(args.aligner):
        aligner.load_state_dict(
            torch.load(args.aligner, map_location=device), strict=False
        )
        print(f"  Aligner: {args.aligner}")
    else:
        print(f"  [AVISO] Aligner nao encontrado: {args.aligner} — pesos aleatorios")
    aligner.to(device).to(autocast_dtype).eval()
    for p in aligner.parameters():
        p.requires_grad_(False)

    # Stage 3: RelationPredictor
    if not os.path.exists(args.rel_head_checkpoint):
        raise FileNotFoundError(
            f"Relation head checkpoint nao encontrado: {args.rel_head_checkpoint}"
        )
    relation_head = RelationHead(
        text_dim=4096,
        vocab_size=len(pred_list),
        proj_dim=args.rel_head_proj_dim,
        hidden=args.rel_head_hidden,
        use_ctx=True,
    )
    state = torch.load(args.rel_head_checkpoint, map_location=device)
    relation_head.load_state_dict(state, strict=True)
    relation_head.to(device).to(autocast_dtype).eval()
    print(f"  RelationHead: {args.rel_head_checkpoint}")

    rel_pred = RelationPredictor(
        relation_head=relation_head,
        aligner=aligner,
        pred_vocab=pred_list,
        edge_threshold=args.edge_threshold,
        max_edges_per_pair=args.max_edges_per_pair,
    )

    # Stage 2: AttributeClassifier (opcional)
    attr_clf = None
    if args.attr_head_checkpoint and os.path.exists(args.attr_head_checkpoint):
        attr_list, _ = build_attribute_vocab(all_sgs, n_attrs=args.n_attrs)
        attribute_head = AttributeHead(
            feat_dim=4096,
            vocab_size=len(attr_list),
        )
        state = torch.load(args.attr_head_checkpoint, map_location=device)
        attribute_head.load_state_dict(state, strict=True)
        attribute_head.to(device).to(autocast_dtype).eval()
        print(f"  AttributeHead: {args.attr_head_checkpoint}")

        attr_clf = AttributeClassifier(
            attribute_head=attribute_head,
            aligner=aligner,
            attr_vocab=attr_list,
            score_threshold=args.attr_threshold,
            top_k=args.attr_top_k,
        )
    else:
        print("  AttributeClassifier: desabilitado (sem checkpoint)")

    if device == "cuda":
        torch.cuda.empty_cache()

    # ── 4. Transform de imagem (para DINO) ──────────────────────────────
    transform = get_transforms(args.image_size)

    # ── 5. Avaliacao SGGen ──────────────────────────────────────────────
    K_VALUES = [20, 50, 100]

    all_recalls: dict[int, list[float]] = {k: [] for k in K_VALUES}
    per_pred_accum: dict[int, dict[str, list[float]]] = {
        k: defaultdict(list) for k in K_VALUES
    }
    n_gt_triples_total = 0
    n_pred_triples_total = 0
    n_detections_total = 0

    print(f"\nIniciando SGGen Recall@K ({len(test_samples)} amostras)...")
    print(f"  Vocabulario: {len(obj_list)} objetos, {len(pred_list)} predicados")
    print(f"  DETR threshold: {args.detr_score_threshold}")
    print(f"  Edge threshold: {args.edge_threshold}")

    try:
        for i, sample in enumerate(tqdm(test_samples, desc="SGGen Eval")):
            # Abre imagem
            img_pil = Image.open(sample["img_path"]).convert("RGB")
            img_pil_resized = img_pil.resize(
                (args.image_size, args.image_size), Image.BILINEAR
            )

            # Stage 1: DETR — deteccao de objetos
            detections = detector.detect(
                img_pil_resized,
                image_size=args.image_size,
                image_size_grid=32,
            )
            n_detections_total += len(detections)

            gt = sample["gt"]
            n_gt_triples_total += len(gt)

            if len(detections) == 0:
                for k in K_VALUES:
                    all_recalls[k].append(0.0)
                continue

            # DINO: extrair visual grid [1, 32, 32, 768]
            img_tensor = transform(img_pil)                    # [3, H, W]
            visual_grid = extract_visual_grid(
                dino, img_tensor, device, autocast_dtype
            )                                                  # [1, 32, 32, 768]

            # Stage 2: atributos (opcional, enriquece detections)
            if attr_clf is not None:
                detections = attr_clf.predict(detections, visual_grid)

            # Stage 3: relacoes entre pares
            edges = rel_pred.predict(detections, visual_grid)

            # Compor scene graph dict para metricas
            nodes = []
            for idx, det in enumerate(detections):
                node = det.to_node_dict()
                node["id"] = idx
                nodes.append(node)
            sg_result = {"nodes": nodes, "edges": edges}

            # Extrair tripletas e computar metricas
            pred_triples = extract_pred_triples(sg_result)
            n_pred_triples_total += len(pred_triples)

            for k in K_VALUES:
                r = recall_at_k(pred_triples, gt, k)
                all_recalls[k].append(r)

                pp = per_predicate_recall(pred_triples, gt, k)
                for pred, val in pp.items():
                    if val is not None:
                        per_pred_accum[k][pred].append(val)

            # Log periodico
            if (i + 1) % 50 == 0:
                r50_so_far = sum(all_recalls[50]) / len(all_recalls[50])
                avg_dets = n_detections_total / (i + 1)
                print(
                    f"  [{i+1}/{len(test_samples)}] "
                    f"R@50={r50_so_far:.4f}  "
                    f"Dets/img={avg_dets:.1f}  "
                    f"GT/img={n_gt_triples_total/(i+1):.1f}  "
                    f"Pred/img={n_pred_triples_total/(i+1):.1f}"
                )

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise

    # ── 6. Resultados ───────────────────────────────────────────────────
    n_eval = max(len(test_samples), 1)

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
    print(f"  Pipeline           : DETR + Attr + Rel (3-stage)")
    print(f"  Amostras avaliadas : {len(test_samples)}")
    print(f"  Dets/img           : {n_detections_total / n_eval:.1f}")
    print(f"  GT triples/img     : {n_gt_triples_total / n_eval:.1f}")
    print(f"  Pred triples/img   : {n_pred_triples_total / n_eval:.1f}")
    print(f"  Predicados c/ GT   : {len(per_pred_accum[50])}")
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
    output_path = os.path.join(
        results_dir, f"sggen_vg150_results_{timestamp_tag}.json"
    )

    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "benchmark": "Visual Genome VG-150",
            "pipeline": "3-stage (DetrDetector + AttributeClassifier + RelationPredictor)",
            "num_samples": len(test_samples),
            "image_size": args.image_size,
            "detr_checkpoint": args.detr_checkpoint,
            "detr_score_threshold": args.detr_score_threshold,
            "aligner_checkpoint": args.aligner,
            "rel_head_checkpoint": args.rel_head_checkpoint,
            "attr_head_checkpoint": args.attr_head_checkpoint,
            "edge_threshold": args.edge_threshold,
            "max_edges_per_pair": args.max_edges_per_pair,
            "vocab_objects": len(obj_list),
            "vocab_predicates": len(pred_list),
            "avg_detections": n_detections_total / n_eval,
            "avg_gt_triples": n_gt_triples_total / n_eval,
            "avg_pred_triples": n_pred_triples_total / n_eval,
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
