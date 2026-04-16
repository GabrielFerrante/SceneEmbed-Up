"""
train_detr_vg150.py
-------------------
Fine-tune do DETR-R50 em VG-150 com imagens 640x640.

Backbone ResNet-50 fica congelado (LR=0). Transformer encoder/decoder treinam
com LR base. Heads de classificacao e bbox treinam com LR 10x maior.

Loss: a propria loss do DETR (classification CE + L1 bbox + GIoU), computada
internamente por ``DetrForObjectDetection(pixel_values, labels=targets)``.

Normalizacao de imagens:
  Usamos ``get_transforms(256)`` (Resize + ToTensor, sem Normalize adicional no
  pipeline atual) e instanciamos ``DetrImageProcessor(do_resize=False,
  do_normalize=False)`` para que o processor apenas converta para o formato de
  tensores esperado pelo DETR, sem re-normalizar. Isso mantem consistencia com
  o pipeline DINO/COYO. Se ``get_transforms`` passar a incluir Normalize
  ImageNet, basta manter ``do_normalize=False`` no processor.

TODO: class weights nao sao aplicados nesta versao. O DETR nao expoe facilmente
      um parametro de class_weight na loss interna (Hungarian matching + CE).
      Para mitigar desbalanceamento, considerar oversampling ou focal loss em
      versao futura.

Uso:
    python train_detr_vg150.py --vg-dir G:/vg
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor

from data.data_utils_pytorch import get_transforms
from data.vg_dataset import (
    build_vg150_vocab,
    deterministic_split,
    load_scene_graphs,
)
from data.vg_detection_dataset import (
    VisualGenomeDetectionDataset,
    detection_collate,
)
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping


def build_detr_targets(
    targets: list[dict],
    processor: DetrImageProcessor,
    device: str,
) -> list[dict]:
    """
    Converte targets do ``VisualGenomeDetectionDataset`` para o formato
    esperado pelo ``DetrForObjectDetection``.

    O dataset ja produz dicts com ``class_labels`` [K] long e ``boxes`` [K, 4]
    (cxcywh normalizado). Aqui apenas movemos para o device correto.

    Parameters
    ----------
    targets : list[dict]
        Cada dict com ``class_labels`` [K] long e ``boxes`` [K, 4] float32.
    processor : DetrImageProcessor
        Nao usado diretamente nesta versao, mantido para compatibilidade futura.
    device : str
        Device de destino.

    Returns
    -------
    list[dict]
        Mesmos dicts com tensores no device alvo.

    Shapes:
        class_labels : [K] long por imagem
        boxes        : [K, 4] float32 (cxcywh normalizado) por imagem
    """
    out = []
    for t in targets:
        out.append({
            "class_labels": t["class_labels"].to(device, non_blocking=True),
            "boxes": t["boxes"].to(device, non_blocking=True),
        })
    return out


@torch.no_grad()
def evaluate(
    model: DetrForObjectDetection,
    loader: DataLoader,
    processor: DetrImageProcessor,
    device: str,
) -> float:
    """
    Avalia o modelo no conjunto de validacao.

    Retorna a loss media (DETR loss = classification CE + L1 bbox + GIoU).

    Parameters
    ----------
    model : DetrForObjectDetection
        Modelo em modo eval.
    loader : DataLoader
        DataLoader de validacao (``detection_collate``).
    processor : DetrImageProcessor
        Processor para conversao de targets.
    device : str
        Device de computacao.

    Returns
    -------
    float
        Loss media no conjunto de validacao.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for images, targets in tqdm(loader, desc="Val", leave=False):
        # images: [B, 3, 640, 640] (ja normalizado pelo transform)
        pixel_values = images.to(device, non_blocking=True)
        labels = build_detr_targets(targets, processor, device)

        outputs = model(pixel_values=pixel_values, labels=labels)
        total_loss += outputs.loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune DETR-R50 em VG-150 (640x640)"
    )
    parser.add_argument("--vg-dir", type=str, required=True,
                        help="Diretorio raiz do Visual Genome (contendo scene_graphs.json)")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (640px e leve, default 8)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="LR base (transformer). Heads usam LR * 10.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs/detr_vg150")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"  VRAM: {free:.1f}/{total:.1f} GB livre")
        torch.cuda.empty_cache()

    # ── 1. Dados VG ────────────────────────────────────────────────────────
    all_sgs = load_scene_graphs(args.vg_dir)
    obj_list, _pred_list, _, _ = build_vg150_vocab(all_sgs)
    print(f"Vocabulario: {len(obj_list)} objetos")

    # Split deterministico: train_all vs test (mesmo seed dos outros scripts)
    train_all_idx, _test_idx = deterministic_split(
        len(all_sgs), args.test_ratio, args.seed
    )

    transform = get_transforms(args.image_size)

    full_train_ds = VisualGenomeDetectionDataset(
        vg_dir=args.vg_dir,
        scene_graphs=all_sgs,
        obj_list=obj_list,
        indices=train_all_idx,
        transform=transform,
    )
    print(f"Imagens train+val com deteccoes validas: {len(full_train_ds)}")

    # Subdividir train_all em train / val
    n_val = int(len(full_train_ds) * args.val_ratio)
    n_train = len(full_train_ds) - n_val
    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(len(full_train_ds), generator=g).tolist()
    train_ds = Subset(full_train_ds, perm[:n_train])
    val_ds = Subset(full_train_ds, perm[n_train:])
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        collate_fn=detection_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
        collate_fn=detection_collate,
    )

    # ── 2. Modelo DETR ────────────────────────────────────────────────────
    # num_labels=150 (+ "no object" gerenciado internamente pelo DETR = 151 saidas)
    # ignore_mismatched_sizes=True porque o pretrained COCO tem 91 classes
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=len(obj_list),
        ignore_mismatched_sizes=True,
    )

    # num_queries=100 (default DETR). Manter 100 inicialmente; subir para 300
    # se recall ficar baixo (requer re-inicializar query_position_embeddings).
    print(f"  num_queries: {model.config.num_queries}")
    print(f"  num_labels:  {model.config.num_labels}")

    model.to(device)

    # Processor: sem resize e sem normalize (get_transforms ja cuida disso)
    processor = DetrImageProcessor(
        do_resize=False,
        do_normalize=False,
    )

    # ── 3. Param groups ───────────────────────────────────────────────────
    # Backbone ResNet-50: congelado (LR=0)
    # Transformer encoder/decoder: LR base
    # Heads (class_labels_classifier + bbox_predictor): LR * 10

    backbone_params = list(model.model.backbone.parameters())
    transformer_params = (
        list(model.model.encoder.parameters())
        + list(model.model.decoder.parameters())
    )
    head_params = (
        list(model.class_labels_classifier.parameters())
        + list(model.bbox_predictor.parameters())
    )

    # Congela backbone explicitamente
    for p in backbone_params:
        p.requires_grad_(False)

    param_groups = [
        {"params": [p for p in transformer_params if p.requires_grad],
         "lr": args.lr},
        {"params": [p for p in head_params if p.requires_grad],
         "lr": args.lr * 10},
    ]

    n_total = sum(p.numel() for p in model.parameters())
    n_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = n_total - n_train_params
    print(f"  Params total: {n_total:,}  treinaveis: {n_train_params:,}  "
          f"congelados: {n_frozen:,}")

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 4. Logging + early stopping ───────────────────────────────────────
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    writer = SummaryWriter(os.path.join(args.log_dir, run_tag))

    early = EarlyStopping(
        save_dir=args.ckpt_dir,
        filename="best_detr_vg150.pth",
        patience=args.patience,
        min_delta=1e-3,
    )

    # ── 5. Training loop ──────────────────────────────────────────────────
    global_step = 0

    try:
        for epoch in range(args.epochs):
            model.train()
            # Backbone permanece em eval (BatchNorm congelado)
            model.model.backbone.eval()

            running_loss = 0.0
            n_batches = 0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
            for images, targets in pbar:
                # images: [B, 3, 640, 640] float32
                pixel_values = images.to(device, non_blocking=True)
                labels = build_detr_targets(targets, processor, device)

                # Forward: DETR computa internamente Hungarian matching +
                # classification CE + L1 bbox + GIoU loss
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss  # escalar

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                # DETR usa grad clipping de 0.1 no paper original
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=0.1,
                )
                optimizer.step()

                running_loss += loss.item()
                n_batches += 1
                global_step += 1

                if global_step % 20 == 0:
                    writer.add_scalar("train/loss_step", loss.item(), global_step)

                pbar.set_postfix(loss=f"{running_loss / max(n_batches, 1):.4f}")

            scheduler.step()

            train_loss = running_loss / max(n_batches, 1)

            # Validacao
            val_loss = evaluate(model, val_loader, processor, device)

            # Logging
            writer.add_scalar("train/loss_epoch", train_loss, epoch + 1)
            writer.add_scalar("val/loss", val_loss, epoch + 1)
            writer.add_scalar("lr/transformer", optimizer.param_groups[0]["lr"], epoch + 1)
            writer.add_scalar("lr/head", optimizer.param_groups[1]["lr"], epoch + 1)
            print(
                f"Epoch {epoch+1}: train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"lr_trans={optimizer.param_groups[0]['lr']:.2e}  "
                f"lr_head={optimizer.param_groups[1]['lr']:.2e}"
            )

            # Checkpoint por epoca (state_dict, nao model inteiro)
            save_epoch_checkpoint(
                model, epoch + 1, directory=args.ckpt_dir, name="detr_epoch"
            )

            # Early stopping
            if early(val_loss, model.state_dict(), epoch):
                break

            if device == "cuda":
                torch.cuda.empty_cache()

    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise
    finally:
        writer.close()

    # Salva vocabulario de objetos (necessario para carregar o detector)
    import json
    vocab_path = os.path.join(args.ckpt_dir, "detr_vg150_vocab.json")
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({"obj_list": obj_list}, f, ensure_ascii=False, indent=2)
    print(f"Vocab salvo em: {vocab_path}")
    print("Treino finalizado.")


if __name__ == "__main__":
    main()
