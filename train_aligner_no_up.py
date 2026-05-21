"""
train_aligner_no_up.py
----------------------
Treina o LoRACrossAttentionAligner sobre shards H5 gerados SEM upsampler
(patches LR nativos do DINOv3, [N, 196, 768]).

Diferenças vs train_aligner_with_h5.py:
- Lê shards de `E:/COYO/embeds/{train,val}_noup/`
- visual_feats shape: [N, 196, 768] (vs [N, 1024, 768] no original)
- Mesma arquitetura do aligner (visual_dim=768, text_dim=4096, rank=64)
- Checkpoint output: best_aligner_no_up.pth + aligner_no_up_epoch_N.pth

Pré-requisito: rodar `embeddings/generate_shards_no_up.py` antes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset_withHD, ShardedH5Dataset_withSSD
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


def train_lora_no_up(
    epochs: int = 20,
    batch_size: int = 64,
    train_shards_dir: str = "E:/COYO/embeds/train_noup/",
    val_shards_dir:   str = "E:/COYO/embeds/val_noup/",
    shards_in_memory: int = 1,
) -> None:
    """
    Shapes principais (consistentes com generate_shards_no_up.py)
    -------------------------------------------------------------
    visual_input:  [B, 196, 768]   patches LR do DINOv3 (sem upsampling)
    text_queries:  [B, 1, 4096]    embedding Qwen por legenda
    attn_weights:  [B, num_heads, N_queries, N_patches]
    """
    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16

    # ── DataLoaders ──────────────────────────────────────────────────────────
    train_ds = ShardedH5Dataset_withHD(train_shards_dir, shards_in_memory=shards_in_memory)
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )

    val_ds = ShardedH5Dataset_withSSD(val_shards_dir)
    val_dataloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True,
    )

    # ── Aligner ──────────────────────────────────────────────────────────────
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    aligner = aligner.to(device).to(autocast_dtype)
    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    global_step = 0
    controller = EarlyStopping(patience=10, filename="best_aligner_no_up.pth")

    # Hiperparâmetros de entropia (consistentes com h5 padrão)
    TARGET_ENTROPY = 1.5
    LAMBDA_ENTROPY = 0.05

    num_shards = len(train_ds._all_shards)

    for epoch in range(epochs):
        aligner.train()
        epoch_loss, epoch_acc, train_samples = 0.0, 0.0, 0

        # Itera por todos os shards dentro de uma época
        for shard_idx in range(num_shards):
            if shard_idx > 0:
                train_ds.rotate_buffer()

            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} shard {shard_idx+1}/{num_shards}",
            )
            for visual_input, text_queries in pbar:
                visual_input = visual_input.to(device, non_blocking=True).to(autocast_dtype)  # [B, 196, 768]
                text_queries = text_queries.to(device, non_blocking=True).to(autocast_dtype)  # [B, 1, 4096]

                try:
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        visual_refined, attn_weights, v_features = aligner(visual_input, text_queries)

                        v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                        t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)
                        logits = torch.matmul(v_norm, t_norm.T) / 0.05
                        labels = torch.arange(v_norm.size(0), device=device)

                        loss_v = F.cross_entropy(logits,   labels)
                        loss_t = F.cross_entropy(logits.T, labels)
                        contrastive_loss = (loss_v + loss_t) / 2

                        entropy = -torch.sum(
                            attn_weights * torch.log(attn_weights + 1e-8), dim=-1
                        )
                        mean_entropy = entropy.mean()
                        entropy_reg = (mean_entropy - TARGET_ENTROPY) ** 2

                        loss = contrastive_loss + LAMBDA_ENTROPY * entropy_reg

                        with torch.no_grad():
                            v_global      = v_features.mean(dim=1)
                            v_global_norm = F.normalize(v_global, p=2, dim=-1)
                            logits_vg     = torch.matmul(v_global_norm, t_norm.T) / 0.07
                            loss_vg       = (F.cross_entropy(logits_vg,   labels) +
                                             F.cross_entropy(logits_vg.T, labels)) / 2
                            acc = (torch.argmax(logits, dim=1) == labels).float().mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                    optimizer.step()

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(torch.cuda.memory_summary())
                    raise

                b = visual_input.size(0)
                epoch_loss    += loss.item() * b
                epoch_acc     += acc.item()  * b
                train_samples += b
                global_step   += 1

                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    acc=f"{acc.item():.2f}",
                    ent=f"{mean_entropy.item():.2f}",
                )

                if global_step % 10 == 0:
                    writer.add_scalar("Loss/total",              loss.item(),             global_step)
                    writer.add_scalar("Loss/contrastive",        contrastive_loss.item(), global_step)
                    writer.add_scalar("Loss/contrastive_vfeats", loss_vg.item(),          global_step)
                    writer.add_scalar("Loss/entropy",            mean_entropy.item(),     global_step)
                    writer.add_scalar("Loss/entropy_reg",        entropy_reg.item(),      global_step)
                    writer.add_scalar("Acc/train_step",          acc.item(),              global_step)

        # Rotaciona para preparar o primeiro shard da próxima época
        train_ds.rotate_buffer()

        # ── Validação ────────────────────────────────────────────────────────
        aligner.eval()
        val_loss, val_acc, val_samples = 0.0, 0.0, 0

        with torch.no_grad():
            for visual_input, text_queries in tqdm(val_dataloader, desc="Validating"):
                visual_input = visual_input.to(device).to(autocast_dtype)
                text_queries = text_queries.to(device).to(autocast_dtype)

                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    visual_refined, _, _ = aligner(visual_input, text_queries)

                    v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                    t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)
                    logits = torch.matmul(v_norm, t_norm.T) / 0.05
                    labels = torch.arange(v_norm.size(0), device=device)

                    loss_v = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    loss = (loss_v + loss_t) / 2
                    acc  = (torch.argmax(logits, dim=1) == labels).float().mean()

                b = v_norm.size(0)
                val_loss    += loss.item() * b
                val_acc     += acc.item()  * b
                val_samples += b

        avg_train_loss = epoch_loss / max(train_samples, 1)
        avg_train_acc  = epoch_acc  / max(train_samples, 1)
        avg_val_loss   = val_loss   / max(val_samples,   1)
        avg_val_acc    = val_acc    / max(val_samples,   1)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Acc/Train_Epoch",  avg_train_acc,  epoch)
        writer.add_scalar("Loss/Val_Epoch",   avg_val_loss,   epoch)
        writer.add_scalar("Acc/Val_Epoch",    avg_val_acc,    epoch)

        print(f"Época {epoch+1}: Treino Loss={avg_train_loss:.4f} Acc={avg_train_acc:.4f} | "
              f"Val Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f}")

        save_epoch_checkpoint(aligner, epoch + 1, name="aligner_no_up_epoch")

        if controller(avg_val_loss, aligner.state_dict(), epoch):
            break

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    EPOCHS = 20
    BATCH_SIZE = 64

    print("--- Treino do Aligner sem upsampler (shards H5 LR) ---")

    torch.cuda.empty_cache()
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
          f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre")

    try:
        train_lora_no_up(epochs=EPOCHS, batch_size=BATCH_SIZE)
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise
