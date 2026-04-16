"""
train_attribute_head.py
-----------------------
Treina a AttributeHead sobre atributos GT do Visual Genome.

Aligner (checkpoint de retrieval) e DINO ficam congelados.
Supervisao: atributos multi-label por objeto com bboxes GT
(setup PredCls-like — isola o aprendizado de atributos de erros de deteccao).

Uso:
    python train_attribute_head.py --vg-dir G:/vg --aligner checkpoints/best_aligner.pth
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.data_utils_pytorch import get_transforms
from data.vg_dataset import (
    VisualGenomeAttributeDataset,
    attribute_collate,
    build_attribute_vocab,
    build_vg150_vocab,
    compute_attribute_pos_weight,
    deterministic_split,
    load_scene_graphs,
)
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.SG.attribute_head import AttributeHead
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


def forward_attrs(
    visual_input: torch.Tensor,  # [B, 1024, 768]
    bboxes: torch.Tensor,        # [M, 4] grid coords
    img_idx: torch.Tensor,       # [M]
    aligner: LoRACrossAttentionAligner,
    head: AttributeHead,
) -> torch.Tensor:
    """
    Forward pass para atributos: ROI pool -> visual_proj -> AttributeHead.

    Retorna logits [M, vocab_attrs].
    """
    B, N, C = visual_input.shape
    grid = visual_input.reshape(B, 32, 32, C)       # [B, 32, 32, 768]
    feat_768 = roi_mean_pool(grid, bboxes, img_idx)  # [M, 768]
    feat_4096 = aligner.visual_proj(feat_768)         # [M, 4096]
    return head(feat_4096)                            # [M, vocab_attrs]


@torch.no_grad()
def evaluate(
    head: AttributeHead,
    dino: DinoSceneEncoder,
    aligner: LoRACrossAttentionAligner,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: str,
    autocast_dtype: torch.dtype,
) -> tuple[float, float]:
    """
    Avalia a AttributeHead no conjunto de validacao.

    Retorna (val_loss, macro_f1).
    Calcula F1 per-class via TP/FP/FN com threshold=0.5.
    """
    head.eval()
    total_loss = 0.0
    n_samples = 0
    vocab_size = head.mlp[-1].out_features

    # Acumuladores per-class
    tp = torch.zeros(vocab_size, device=device)
    fp = torch.zeros(vocab_size, device=device)
    fn = torch.zeros(vocab_size, device=device)

    for images, bboxes_batch, attrs_batch, img_idx in tqdm(loader, desc="Val", leave=False):
        if bboxes_batch.numel() == 0:
            continue
        images = images.to(device, non_blocking=True)
        bboxes_batch = bboxes_batch.to(device, non_blocking=True)
        attrs_batch = attrs_batch.to(device, non_blocking=True)
        img_idx = img_idx.to(device, non_blocking=True)

        # TODO: otimizacao futura — agrupar objetos por imagem para evitar
        # DINO duplicado. Aceitar ineficiencia neste primeiro checkpoint.
        visual_input = extract_visual_input(dino, images, autocast_dtype)
        logits = forward_attrs(visual_input, bboxes_batch, img_idx, aligner, head)
        loss = criterion(logits, attrs_batch.to(logits.dtype))

        bsz = bboxes_batch.size(0)
        total_loss += loss.item() * bsz
        n_samples += bsz

        # Predicoes binarias com threshold=0.5
        preds = (torch.sigmoid(logits) >= 0.5).float()
        targets = attrs_batch.to(preds.dtype)

        tp += (preds * targets).sum(dim=0)
        fp += (preds * (1.0 - targets)).sum(dim=0)
        fn += ((1.0 - preds) * targets).sum(dim=0)

    val_loss = total_loss / max(n_samples, 1)

    # Macro F1: media dos F1 per-class (ignora classes sem suporte)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1_per_class = 2 * precision * recall / (precision + recall + 1e-8)
    # Apenas classes com pelo menos 1 GT positivo
    has_support = (tp + fn) > 0
    macro_f1 = f1_per_class[has_support].mean().item() if has_support.any() else 0.0

    return val_loss, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Treino AttributeHead sobre VG")
    parser.add_argument("--vg-dir", type=str, required=True)
    parser.add_argument("--aligner", type=str, default="checkpoints/best_aligner.pth",
                        help="Checkpoint do LoRACrossAttentionAligner (congelado)")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Objetos por batch (cada slot = 1 objeto de 1 imagem)")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=5)
    parser.add_argument("--n-objects", type=int, default=150)
    parser.add_argument("--n-attrs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pos-weight-clip", type=float, default=50.0)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/attribute_head")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device} | dtype: {autocast_dtype}")

    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"  VRAM: {free:.1f}/{total:.1f} GB livre")
        torch.cuda.empty_cache()

    # -- 1. Dados VG -------------------------------------------------------
    all_sgs = load_scene_graphs(args.vg_dir)
    obj_list, _, _, _ = build_vg150_vocab(all_sgs, n_objects=args.n_objects)
    attr_list, _ = build_attribute_vocab(all_sgs, n_attrs=args.n_attrs)

    train_all_idx, _test_idx = deterministic_split(
        len(all_sgs), args.test_ratio, args.seed
    )

    transform = get_transforms(args.image_size)
    full_train_ds = VisualGenomeAttributeDataset(
        vg_dir=args.vg_dir,
        scene_graphs=all_sgs,
        obj_list=obj_list,
        attr_list=attr_list,
        indices=train_all_idx,
        transform=transform,
    )
    print(f"Objetos train+val com atributos validos: {len(full_train_ds)}")

    # Split train/val deterministico
    n_val = int(len(full_train_ds) * args.val_ratio)
    n_train = len(full_train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_ds = Subset(full_train_ds, perm[:n_train])
    val_ds = Subset(full_train_ds, perm[n_train:])
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    # pos_weight para BCE (desbalanceamento de atributos raros)
    pos_weight = compute_attribute_pos_weight(
        full_train_ds, clip_max=args.pos_weight_clip
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=attribute_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=attribute_collate,
    )

    # -- 2. Modelos ---------------------------------------------------------
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

    head = AttributeHead(
        feat_dim=4096,
        vocab_size=len(attr_list),
    ).to(device).to(autocast_dtype)
    n_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"AttributeHead params treinaveis: {n_params:,}")

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.to(autocast_dtype)
    )
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # -- 3. Logging + early stopping ----------------------------------------
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, run_tag))
    early = EarlyStopping(
        save_dir=args.ckpt_dir,
        filename="best_attribute_head.pth",
        patience=args.patience,
        min_delta=1e-3,
    )

    # -- 4. Loop de treino --------------------------------------------------
    global_step = 0
    try:
        for epoch in range(args.epochs):
            head.train()
            running_loss = 0.0
            n_seen = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")

            for images, bboxes_batch, attrs_batch, img_idx in pbar:
                if bboxes_batch.numel() == 0:
                    continue
                images = images.to(device, non_blocking=True)
                bboxes_batch = bboxes_batch.to(device, non_blocking=True)
                attrs_batch = attrs_batch.to(device, non_blocking=True)
                img_idx = img_idx.to(device, non_blocking=True)

                # TODO: otimizacao futura — agrupar objetos por imagem,
                # extrair DINO uma vez por imagem unica no batch. Aceitar
                # ineficiencia (imagens duplicadas) neste primeiro checkpoint.
                visual_input = extract_visual_input(dino, images, autocast_dtype)

                logits = forward_attrs(
                    visual_input, bboxes_batch, img_idx, aligner, head
                )                                                # [M, vocab_attrs]
                loss = criterion(logits, attrs_batch.to(logits.dtype))

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
                optimizer.step()

                m = bboxes_batch.size(0)
                running_loss += loss.item() * m
                n_seen += m
                global_step += 1

                if global_step % 20 == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)
                pbar.set_postfix(
                    loss=f"{running_loss / max(n_seen, 1):.4f}",
                    objs=m,
                )

            train_loss = running_loss / max(n_seen, 1)
            val_loss, macro_f1 = evaluate(
                head, dino, aligner, val_loader, criterion, device, autocast_dtype
            )
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("val/macro_f1", macro_f1, epoch + 1)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  macro_f1={macro_f1:.4f}"
            )

            save_epoch_checkpoint(
                head, epoch + 1, directory=args.ckpt_dir, name="attribute_head_epoch"
            )

            if early(val_loss, head.state_dict(), epoch):
                break

            if device == "cuda":
                torch.cuda.empty_cache()

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise
    finally:
        writer.close()

    # Salva vocab de atributos usado
    vocab_path = os.path.join(args.ckpt_dir, "attribute_head_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"attr_list": attr_list}, f, ensure_ascii=False, indent=2)
    print(f"Vocab salvo em: {vocab_path}")


if __name__ == "__main__":
    main()
