"""
train_sg_head.py
----------------
Treina apenas a SGClassifierHead sobre features DINO congeladas.
NAO toca no pipeline de retrieval (aligner LoRA) — head independente para SGGen.

Uso:
    python train_sg_head.py --vg-dir G:/vg
    python train_sg_head.py --vg-dir G:/vg --epochs 20 --batch-size 32
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
    VisualGenomeMultiLabelDataset,
    build_vg150_vocab,
    compute_pos_weight,
    deterministic_split,
    load_scene_graphs,
)
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.SG.classifier_head import SGClassifierHead
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping


def extract_visual_input(
    dino: DinoSceneEncoder,
    images: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    DINO -> HR feats -> pool 32x32 -> [B, 1024, 768] em dtype alvo.

    Shapes:
      images:       [B, 3, H, W]
      hr_feat:      [B, 768, H_hr, W_hr]
      hr_feat_small:[B, 768, 32, 32]
      out:          [B, 1024, 768]
    """
    _, hr_feat = dino.extract_features(images)
    hr_feat_small = F.adaptive_avg_pool2d(hr_feat, (32, 32))
    B, C, H, W = hr_feat_small.shape
    visual_input = hr_feat_small.reshape(B, C, H * W).transpose(1, 2)
    return visual_input.to(dtype).contiguous()


@torch.no_grad()
def evaluate(
    head: SGClassifierHead,
    dino: DinoSceneEncoder,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    autocast_dtype: torch.dtype,
) -> tuple[float, float]:
    """Retorna (val_loss, macro_f1 @ threshold=0.5)."""
    head.eval()
    total_loss = 0.0
    n_samples = 0
    tp = fp = fn = None
    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        visual_input = extract_visual_input(dino, images, autocast_dtype)
        patch_logits = head(visual_input)                       # [B, N, vocab]
        logits = patch_logits.amax(dim=1)                       # MIL max-pool -> [B, vocab]
        loss = criterion(logits.float(), targets.float())

        bsz = images.size(0)
        total_loss += loss.item() * bsz
        n_samples += bsz

        preds = (torch.sigmoid(logits.float()) > 0.5).float()
        if tp is None:
            tp = torch.zeros(preds.size(1), device=device)
            fp = torch.zeros_like(tp)
            fn = torch.zeros_like(tp)
        tp += (preds * targets).sum(dim=0)
        fp += (preds * (1 - targets)).sum(dim=0)
        fn += ((1 - preds) * targets).sum(dim=0)

    val_loss = total_loss / max(n_samples, 1)
    if tp is None:
        return val_loss, 0.0
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    # Macro F1 apenas em classes com ao menos 1 positivo no val
    mask = (tp + fn) > 0
    macro_f1 = f1[mask].mean().item() if mask.any() else 0.0
    return val_loss, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino SGClassifierHead sobre VG-150")
    parser.add_argument("--vg-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-objects", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="Fracao do conjunto de treino usada para validacao")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos-weight-clip", type=float, default=50.0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/sg_head")
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
    obj_list, _, _, _ = build_vg150_vocab(all_sgs, n_objects=args.n_objects)

    train_all_idx, _test_idx = deterministic_split(
        len(all_sgs), args.test_ratio, args.seed
    )

    transform = get_transforms(args.image_size)
    full_train_ds = VisualGenomeMultiLabelDataset(
        vg_dir=args.vg_dir,
        scene_graphs=all_sgs,
        obj_list=obj_list,
        indices=train_all_idx,
        transform=transform,
    )
    print(f"Amostras train+val validas: {len(full_train_ds)}")

    # split train/val deterministico
    n_val = int(len(full_train_ds) * args.val_ratio)
    n_train = len(full_train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_ds = Subset(full_train_ds, perm[:n_train])
    val_ds = Subset(full_train_ds, perm[n_train:])
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    pos_weight = compute_pos_weight(full_train_ds, clip_max=args.pos_weight_clip).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    # ── 2. Modelos ─────────────────────────────────────────────────────
    dino = DinoSceneEncoder(device=device)
    for p in dino.model.parameters():
        p.requires_grad_(False)

    head = SGClassifierHead(
        visual_dim=768,
        vocab_size=len(obj_list),
    ).to(device)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"SGClassifierHead params treinaveis: {n_params:,}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ── 3. Logging + early stopping ────────────────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, run_tag))
    early = EarlyStopping(
        save_dir=args.ckpt_dir,
        filename="best_sg_head.pth",
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
            for images, targets in pbar:
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                visual_input = extract_visual_input(dino, images, autocast_dtype)

                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    patch_logits = head(visual_input)                   # [B, N, vocab]
                    logits = patch_logits.amax(dim=1)                   # MIL max-pool -> [B, vocab]
                loss = criterion(logits.float(), targets.float())

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()

                bsz = images.size(0)
                running_loss += loss.item() * bsz
                n_seen += bsz
                global_step += 1

                if global_step % 20 == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                pbar.set_postfix(loss=f"{running_loss / n_seen:.4f}")

            train_loss = running_loss / max(n_seen, 1)
            val_loss, macro_f1 = evaluate(
                head, dino, val_loader, criterion, device, autocast_dtype
            )
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/macro_f1", macro_f1, epoch + 1)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  macro_f1={macro_f1:.4f}"
            )

            save_epoch_checkpoint(head, epoch + 1, directory=args.ckpt_dir, name="sg_head_epoch")

            if early(val_loss, head.state_dict(), epoch):
                break

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise
    finally:
        writer.close()

    # Salva o vocabulario usado (necessario para inference/eval)
    vocab_path = os.path.join(args.ckpt_dir, "sg_head_vocab.json")
    import json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"obj_list": obj_list}, f, ensure_ascii=False, indent=2)
    print(f"Vocab salvo em: {vocab_path}")


if __name__ == "__main__":
    main()
