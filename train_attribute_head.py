
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5AttributeDataset_withHD, ShardedH5AttributeDataset_withSSD
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.SG.attribute_head import AttributeHead
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


def _compute_f1_macro(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) >= threshold).float()
    tp = (preds * labels).sum(dim=0)
    fp = (preds * (1 - labels)).sum(dim=0)
    fn = ((1 - preds) * labels).sum(dim=0)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.mean().item()


def train_attribute_head(
    epochs: int = 20,
    batch_size: int = 256,
    aligner_checkpoint: str | None = None,
    vocab_size: int = 200,
    feat_dim: int = 768,
    text_dim: int = 4096,
    hidden: int = 1024,
    dropout: float = 0.1,
) -> None:

    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16

    train_ds = ShardedH5AttributeDataset_withHD(TRAIN_SHARDS_PATH, shards_in_memory=1)
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_ds = ShardedH5AttributeDataset_withSSD(VAL_SHARDS_PATH)
    val_dataloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True,
    )

    # visual_proj: projeta obj_feats [B, 768] → [B, 4096] (espaco alinhado)
    aligner = LoRACrossAttentionAligner(visual_dim=feat_dim, text_dim=text_dim, rank=64)
    if aligner_checkpoint:
        state = torch.load(aligner_checkpoint, map_location="cpu")
        aligner.load_state_dict(state, strict=False)
        print(f"Aligner carregado de {aligner_checkpoint}")
    visual_proj: nn.Linear = aligner.visual_proj.to(device).to(autocast_dtype)
    for p in visual_proj.parameters():
        p.requires_grad_(False)
    visual_proj.eval()

    head = AttributeHead(feat_dim=text_dim, vocab_size=vocab_size, hidden=hidden, dropout=dropout)
    head = head.to(device).to(autocast_dtype)

    trainable_params = list(head.parameters())
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    global_step = 0
    controller = EarlyStopping(patience=7)
    num_shards = len(train_ds._all_shards)

    for epoch in range(epochs):
        head.train()
        epoch_loss, train_samples = 0.0, 0

        for shard_idx in range(num_shards):
            if shard_idx > 0:
                train_ds.rotate_buffer()

            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} shard {shard_idx+1}/{num_shards}",
            )
            for obj_feats, attr_labels in pbar:
                obj_feats   = obj_feats.to(device, non_blocking=True).to(autocast_dtype)
                attr_labels = attr_labels.to(device, non_blocking=True).to(autocast_dtype)

                try:
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        with torch.no_grad():
                            proj_feats = visual_proj(obj_feats)         # [B, 4096]
                        logits = head(proj_feats)                       # [B, vocab_size]
                        loss = F.binary_cross_entropy_with_logits(
                            logits, attr_labels.float(),
                        )

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(torch.cuda.memory_summary())
                    raise

                b = obj_feats.size(0)
                epoch_loss    += loss.item() * b
                train_samples += b
                global_step   += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

                if global_step % 10 == 0:
                    writer.add_scalar("Loss/train_step", loss.item(), global_step)

        train_ds.rotate_buffer()

        # ── Validação ───────────────────────────────────────────────────────
        head.eval()
        val_loss, val_samples = 0.0, 0
        all_logits, all_labels = [], []

        with torch.no_grad():
            for obj_feats, attr_labels in tqdm(val_dataloader, desc="Validating"):
                obj_feats   = obj_feats.to(device).to(autocast_dtype)
                attr_labels = attr_labels.to(device).to(autocast_dtype)

                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    proj_feats = visual_proj(obj_feats)
                    logits     = head(proj_feats)
                    loss = F.binary_cross_entropy_with_logits(logits, attr_labels.float())

                b = obj_feats.size(0)
                val_loss    += loss.item() * b
                val_samples += b
                all_logits.append(logits.float().cpu())
                all_labels.append(attr_labels.float().cpu())

        all_logits_cat = torch.cat(all_logits, dim=0)
        all_labels_cat = torch.cat(all_labels, dim=0)
        macro_f1 = _compute_f1_macro(all_logits_cat, all_labels_cat)

        avg_train_loss = epoch_loss / max(train_samples, 1)
        avg_val_loss   = val_loss   / max(val_samples, 1)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val_Epoch",   avg_val_loss,   epoch)
        writer.add_scalar("F1/Val_Macro",     macro_f1,       epoch)

        print(
            f"Época {epoch+1}: Train Loss={avg_train_loss:.4f} | "
            f"Val Loss={avg_val_loss:.4f} | F1 Macro={macro_f1:.4f}"
        )

        save_epoch_checkpoint(head, epoch + 1)

        if controller(avg_val_loss, head.state_dict(), epoch):
            break

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    EPOCHS     = 30
    BATCH_SIZE = 512
    VOCAB_SIZE = 200   # numero de atributos VG

    TRAIN_SHARDS_PATH  = "F:/VG/attr_shards/train/"
    VAL_SHARDS_PATH    = "F:/VG/attr_shards/val/"
    ALIGNER_CHECKPOINT = "checkpoints/aligner_epoch_best.pt"  # None para inicializar do zero

    print("--- Iniciando Treinamento da AttributeHead ---")
    torch.cuda.empty_cache()
    print(
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre"
    )

    try:
        train_attribute_head(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            aligner_checkpoint=ALIGNER_CHECKPOINT,
            vocab_size=VOCAB_SIZE,
        )
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print("Erro critico de GPU no loop principal.")
            print(torch.cuda.memory_summary())
        raise
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
        raise
