"""
train_relation_head.py
----------------------
Treina a RelationHead sobre pares GT do Visual Genome (VG-150).

Aligner (checkpoint de retrieval) e DINO ficam congelados.
Supervisao: tripletas (sub, pred, obj) do VG, usando bboxes GT no treino
(setup PredCls-like — isola o aprendizado de relacoes de erros de deteccao).

Uso:
    python train_relation_head.py --vg-dir G:/vg --aligner checkpoints/best_aligner.pth
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_utils_pytorch import get_transforms
from data.vg_dataset import (
    VisualGenomeRelationDataset,
    build_vg150_vocab,
    compute_predicate_class_weight,
    deterministic_split,
    load_scene_graphs,
    relation_collate,
)
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.SG.relation_head import RelationHead
from models.SG.relation_predictor import RelationPredictor
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping


def extract_visual_input(
    dino: DinoSceneEncoder,
    images: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """DINO -> HR feats -> pool 32x32 -> [B, 1024, 768] em dtype alvo."""
    _, hr_feat = dino.extract_features(images)
    hr_feat_small = F.adaptive_avg_pool2d(hr_feat, (32, 32))
    B, C, H, W = hr_feat_small.shape
    visual_input = hr_feat_small.reshape(B, C, H * W).transpose(1, 2)
    return visual_input.to(dtype).contiguous()


def roi_mean_pool(
    grid: torch.Tensor,
    bboxes: torch.Tensor,
    img_idx: torch.Tensor,
) -> torch.Tensor:
    """
    Mean-pool de patches dentro de cada bbox.

    Shapes:
      grid    : [B, 32, 32, 768]
      bboxes  : [M, 4] (x1, y1, x2, y2) em coords de grade
      img_idx : [M] indice da imagem no batch

    Returns:
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


def forward_pairs(
    visual_input: torch.Tensor,
    sub_bboxes: torch.Tensor,
    obj_bboxes: torch.Tensor,
    img_idx: torch.Tensor,
    aligner: LoRACrossAttentionAligner,
    head: RelationHead,
) -> torch.Tensor:
    """
    Computa logits de predicados usando ROI mean-pool do retangulo uniao.

    Shapes
    ------
    visual_input : [B, 1024, 768]
    sub_bboxes   : [M, 4] (x1, y1, x2, y2) em coords de grade
    obj_bboxes   : [M, 4]
    img_idx      : [M]

    Returns
    -------
    logits : [M, vocab_pred]
    """
    B, N, C = visual_input.shape
    grid = visual_input.reshape(B, 32, 32, C)                        # [B, 32, 32, 768]

    sub_feat_768 = roi_mean_pool(grid, sub_bboxes, img_idx)          # [M, 768]
    obj_feat_768 = roi_mean_pool(grid, obj_bboxes, img_idx)          # [M, 768]

    # union bbox: retangulo minimo que contem sub + obj
    union_bbox = RelationPredictor.compute_union_bbox(sub_bboxes, obj_bboxes)  # [M, 4]
    union_feat_768 = roi_mean_pool(grid, union_bbox, img_idx)        # [M, 768]

    # projecao para espaco texto (4096) via aligner congelado
    sub_feat = aligner.visual_proj(sub_feat_768)                     # [M, 4096]
    obj_feat = aligner.visual_proj(obj_feat_768)                     # [M, 4096]
    union_feat = aligner.visual_proj(union_feat_768)                 # [M, 4096]

    return head(sub_feat, obj_feat, union_feat)                      # [M, vocab_pred]


@torch.no_grad()
def evaluate(
    head: RelationHead,
    dino: DinoSceneEncoder,
    aligner: LoRACrossAttentionAligner,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    autocast_dtype: torch.dtype,
) -> tuple[float, float, float]:
    """Retorna (val_loss, top1_acc, macro_recall)."""
    head.eval()
    total_loss = 0.0
    n_samples = 0
    correct = 0
    vocab_size = head.mlp[-1].out_features
    per_cls_tp = torch.zeros(vocab_size, device=device)
    per_cls_total = torch.zeros(vocab_size, device=device)

    for images, sub_b, obj_b, preds, img_idx in tqdm(loader, desc="Val", leave=False):
        if preds.numel() == 0:
            continue
        images = images.to(device, non_blocking=True)
        sub_b = sub_b.to(device, non_blocking=True)
        obj_b = obj_b.to(device, non_blocking=True)
        preds = preds.to(device, non_blocking=True)
        img_idx = img_idx.to(device, non_blocking=True)

        visual_input = extract_visual_input(dino, images, autocast_dtype)
        logits = forward_pairs(visual_input, sub_b, obj_b, img_idx, aligner, head)
        loss = criterion(logits, preds)

        bsz = preds.size(0)
        total_loss += loss.item() * bsz
        n_samples += bsz

        pred_cls = logits.argmax(dim=-1)
        correct += (pred_cls == preds).sum().item()

        for c in range(vocab_size):
            mask = (preds == c)
            if mask.any():
                per_cls_total[c] += mask.sum()
                per_cls_tp[c] += ((pred_cls == c) & mask).sum()

    val_loss = total_loss / max(n_samples, 1)
    top1 = correct / max(n_samples, 1)
    valid = per_cls_total > 0
    macro_recall = (per_cls_tp[valid] / per_cls_total[valid]).mean().item() if valid.any() else 0.0
    return val_loss, top1, macro_recall


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino RelationHead sobre VG-150")
    parser.add_argument("--vg-dir", type=str, required=True)
    parser.add_argument("--aligner", type=str, default="checkpoints/best_aligner.pth",
                        help="Checkpoint do LoRACrossAttentionAligner (congelado)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Imagens por batch (pares sao ~10x mais por imagem)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--n-objects", type=int, default=150)
    parser.add_argument("--n-predicates", type=int, default=50)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--class-weight-clip", type=float, default=50.0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/relation_head")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {autocast_dtype}")

    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"  VRAM: {free:.1f}/{total:.1f} GB livre")
        torch.cuda.empty_cache()

    # ── 1. Dados VG ────────────────────────────────────────────────────
    all_sgs = load_scene_graphs(args.vg_dir)
    obj_list, pred_list, _, _ = build_vg150_vocab(
        all_sgs, n_objects=args.n_objects, n_predicates=args.n_predicates
    )
    train_all_idx, _test_idx = deterministic_split(
        len(all_sgs), args.test_ratio, args.seed
    )

    transform = get_transforms(args.image_size)
    full_train_ds = VisualGenomeRelationDataset(
        vg_dir=args.vg_dir,
        scene_graphs=all_sgs,
        obj_list=obj_list,
        pred_list=pred_list,
        indices=train_all_idx,
        transform=transform,
    )
    print(f"Imagens train+val com pares validos: {len(full_train_ds)}")

    n_val = int(len(full_train_ds) * args.val_ratio)
    n_train = len(full_train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_ds = Subset(full_train_ds, perm[:n_train])
    val_ds = Subset(full_train_ds, perm[n_train:])
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    class_weight = compute_predicate_class_weight(
        full_train_ds, clip_max=args.class_weight_clip
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=relation_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=relation_collate,
    )

    # ── 2. Modelos ─────────────────────────────────────────────────────
    dino = DinoSceneEncoder(device=device)
    for p in dino.model.parameters():
        p.requires_grad_(False)

    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    if os.path.exists(args.aligner):
        state = torch.load(args.aligner, map_location=device)
        aligner.load_state_dict(state, strict=False)
        print(f"  Aligner carregado: {args.aligner}")
    else:
        print(f"  [AVISO] Aligner nao encontrado: {args.aligner} — usando pesos aleatorios")
    aligner.to(device).to(autocast_dtype).eval()
    for p in aligner.parameters():
        p.requires_grad_(False)

    head = RelationHead(
        text_dim=4096,
        vocab_size=len(pred_list),
        proj_dim=args.proj_dim,
        hidden=args.hidden,
        dropout=args.dropout,
        use_ctx=True,
    ).to(device).to(autocast_dtype)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"RelationHead params treinaveis: {n_params:,}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight.to(autocast_dtype))
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # ── 3. Logging + early stopping ────────────────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, run_tag))
    early = EarlyStopping(
        save_dir=args.ckpt_dir,
        filename="best_relation_head.pth",
        patience=args.patience,
        min_delta=1e-3,
    )

    # ── 4. Loop ────────────────────────────────────────────────────────
    global_step = 0
    try:
        for epoch in range(args.epochs):
            head.train()
            running_loss = 0.0
            n_seen = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for images, sub_b, obj_b, preds, img_idx in pbar:
                if preds.numel() == 0:
                    continue
                images = images.to(device, non_blocking=True)
                sub_b = sub_b.to(device, non_blocking=True)
                obj_b = obj_b.to(device, non_blocking=True)
                preds = preds.to(device, non_blocking=True)
                img_idx = img_idx.to(device, non_blocking=True)

                visual_input = extract_visual_input(dino, images, autocast_dtype)

                logits = forward_pairs(
                    visual_input, sub_b, obj_b, img_idx, aligner, head
                )                                                        # [M, vocab_pred] bf16
                loss = criterion(logits, preds)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()

                m = preds.size(0)
                running_loss += loss.item() * m
                n_seen += m
                global_step += 1

                if global_step % 20 == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                pbar.set_postfix(loss=f"{running_loss / max(n_seen, 1):.4f}",
                                 pairs=m)

            train_loss = running_loss / max(n_seen, 1)
            val_loss, top1, macro_r = evaluate(
                head, dino, aligner, val_loader, criterion, device, autocast_dtype
            )
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/top1", top1, epoch + 1)
            writer.add_scalar("val/macro_recall", macro_r, epoch + 1)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  top1={top1:.4f}  macro_R={macro_r:.4f}"
            )

            save_epoch_checkpoint(
                head, epoch + 1, directory=args.ckpt_dir, name="relation_head_epoch"
            )

            if early(val_loss, head.state_dict(), epoch):
                break

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise
    finally:
        writer.close()

    # Salva vocab de predicados usado
    import json
    vocab_path = os.path.join(args.ckpt_dir, "relation_head_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"pred_list": pred_list}, f, ensure_ascii=False, indent=2)
    print(f"Vocab salvo em: {vocab_path}")


if __name__ == "__main__":
    main()
