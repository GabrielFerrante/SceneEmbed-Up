"""
Treino do LoRACrossAttentionAligner com seleção de patches via PCA
(critério de variância) — baseline clássico para comparação com o ATS.

Método
------
Para cada imagem, calcula-se PCA (SVD de baixo posto) sobre os N=1024
patches anyup. Cada patch recebe um "PCA projection score" = norma de sua
projeção nos top-M componentes principais (direções de maior variância).
Os K=196 patches com maior score são mantidos.

Diferente do AdaptiveTokenSampler (ATS, ver train_aligner_ats_h5.py), este
método é determinístico e não-aprendido — não há parâmetros, não há loss de
seleção. Serve como baseline clássico de seleção por variância.

Checkpoints
-----------
    checkpoints/pca_aligner_epoch_N.pth   — por época
    checkpoints/best_pca_aligner.pth      — melhor val loss
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset_withHD, ShardedH5Dataset_withSSD
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from utils.checkpoint import save_epoch_checkpoint
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


# ── PCA Variance Sampler ──────────────────────────────────────────────────────

class PCAVarianceSampler(nn.Module):
    """
    Seleciona K patches por imagem via PCA projection score (critério de
    variância). Método clássico, não-aprendido — baseline para o ATS.

    Para cada imagem, calcula PCA (SVD de baixo posto, torch.pca_lowrank)
    sobre os N patches e pontua cada patch pela norma de sua projeção nos
    top-M componentes principais (direções de maior variância). Mantém os K
    patches com maior score.

    Shapes
    ------
    patches    : [B, N, visual_dim]
    → selected : [B, K, visual_dim]
    → scores   : [B, N]   (PCA projection score, soma dos quadrados)
    → top_k_idx: [B, K]
    """

    def __init__(self, k: int = 196, n_components: int = 64) -> None:
        super().__init__()
        self.k = k
        self.n_components = n_components

    @torch.no_grad()
    def forward(self, patches: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # PCA requer float32 para estabilidade numérica do SVD
        x = patches.float()                                       # [B, N, D]
        mean = x.mean(dim=1, keepdim=True)                         # [B, 1, D]
        centered = x - mean                                        # [B, N, D]

        # SVD de baixo posto (batched): V → [B, D, n_components]
        _, _, V = torch.pca_lowrank(centered, q=self.n_components, center=False)

        # Projeção de cada patch nos top-M componentes principais
        proj = centered @ V                                        # [B, N, M]
        scores = (proj ** 2).sum(dim=-1)                           # [B, N]

        _, top_k_idx = torch.topk(scores, k=self.k, dim=-1)        # [B, K]
        idx_exp = top_k_idx.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        selected = torch.gather(patches, dim=1, index=idx_exp)     # [B, K, D]

        return selected, scores, top_k_idx


# ── Loop de treino ────────────────────────────────────────────────────────────

def train_lora_projection_pca(
    epochs: int = 20,
    batch_size: int = 64,
    k_patches: int = 196,
    n_components: int = 64,
    resume_checkpoint: str | None = None,
    start_epoch: int = 0,
) -> None:

    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")
    device        = "cuda" if torch.cuda.is_available() else "cpu"
    autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_ds = ShardedH5Dataset_withHD("F:/COYO/embeds/train_anyup/", shards_in_memory=1)
    train_dataloader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True,
    )
    val_ds = ShardedH5Dataset_withSSD("G:/coyo/embeds/val_anyup/")
    val_dataloader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=True, num_workers=2, prefetch_factor=2, persistent_workers=False,
    )

    # ── Modelos ───────────────────────────────────────────────────────────────
    sampler = PCAVarianceSampler(k=k_patches, n_components=n_components)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    aligner = aligner.to(device).to(autocast_dtype)

    if resume_checkpoint is not None:
        aligner.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Checkpoint carregado: {resume_checkpoint} (retomando da época {start_epoch + 1})")

    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # ── Hiperparâmetros ───────────────────────────────────────────────────────
    TARGET_ENTROPY = 1.5
    LAMBDA_ENTROPY = 0.05

    global_step = 0
    controller  = EarlyStopping(patience=10, filename="best_pca_aligner.pth")
    num_shards  = len(train_ds._all_shards)

    for epoch in range(start_epoch, epochs):
        aligner.train()
        epoch_loss, epoch_acc, train_samples = 0.0, 0.0, 0

        for shard_idx in range(num_shards):
            if shard_idx > 0:
                train_ds.rotate_buffer()

            pbar = tqdm(
                train_dataloader,
                desc=f"Epoch {epoch+1}/{epochs} shard {shard_idx+1}/{num_shards}",
            )
            for visual_input, text_queries in pbar:
                visual_input = visual_input.to(device, non_blocking=True).to(autocast_dtype)
                text_queries = text_queries.to(device, non_blocking=True).to(autocast_dtype)

                try:
                    # ── PCA: seleciona K patches por variância ────────────────
                    # visual_input: [B, 1024, 768]
                    # selected:     [B, K,    768]
                    selected, _, _ = sampler(visual_input)

                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        visual_refined, attn_weights, v_features = aligner(selected, text_queries)

                        v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                        t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)
                        logits = torch.matmul(v_norm, t_norm.T) / 0.05
                        labels = torch.arange(v_norm.size(0), device=device)
                        loss_v = F.cross_entropy(logits,   labels)
                        loss_t = F.cross_entropy(logits.T, labels)
                        contrastive_loss = (loss_v + loss_t) / 2

                        # Entropia do cross-attention (sobre K patches)
                        entropy      = -torch.sum(
                            attn_weights * torch.log(attn_weights + 1e-8), dim=-1
                        )
                        mean_entropy = entropy.mean()
                        entropy_reg  = (mean_entropy - TARGET_ENTROPY) ** 2

                        loss = contrastive_loss + LAMBDA_ENTROPY * entropy_reg

                        with torch.no_grad():
                            v_global      = v_features.mean(dim=1)
                            v_global_norm = F.normalize(v_global, p=2, dim=-1)
                            logits_vg     = torch.matmul(v_global_norm, t_norm.T) / 0.07
                            loss_vg       = (F.cross_entropy(logits_vg,   labels) +
                                            F.cross_entropy(logits_vg.T, labels)) / 2
                            acc           = (torch.argmax(logits, dim=1) == labels).float().mean()

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
                    writer.add_scalar("Loss/total",              loss.item(),              global_step)
                    writer.add_scalar("Loss/contrastive",        contrastive_loss.item(),  global_step)
                    writer.add_scalar("Loss/entropy",            mean_entropy.item(),      global_step)
                    writer.add_scalar("Loss/entropy_reg",        entropy_reg.item(),       global_step)
                    writer.add_scalar("Loss/contrastive_vfeats", loss_vg.item(),           global_step)
                    writer.add_scalar("Acc/train_step",          acc.item(),               global_step)

        train_ds.rotate_buffer()

        # ── Validação ─────────────────────────────────────────────────────────
        aligner.eval()
        val_loss, val_acc, val_samples = 0.0, 0.0, 0

        with torch.no_grad():
            for visual_input, text_queries in tqdm(val_dataloader, desc="Validating"):
                visual_input = visual_input.to(device).to(autocast_dtype)
                text_queries = text_queries.to(device).to(autocast_dtype)

                selected, _, _ = sampler(visual_input)

                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    visual_refined, _, _ = aligner(selected, text_queries)

                    v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                    t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)
                    logits = torch.matmul(v_norm, t_norm.T) / 0.05
                    labels = torch.arange(v_norm.size(0), device=device)
                    loss_v = F.cross_entropy(logits,   labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    loss   = (loss_v + loss_t) / 2
                    acc    = (torch.argmax(logits, dim=1) == labels).float().mean()

                b = v_norm.size(0)
                val_loss    += loss.item() * b
                val_acc     += acc.item()  * b
                val_samples += b

        # ── Métricas finais ───────────────────────────────────────────────────
        avg_train_loss = epoch_loss / max(train_samples, 1)
        avg_train_acc  = epoch_acc  / max(train_samples, 1)
        avg_val_loss   = val_loss   / max(val_samples,   1)
        avg_val_acc    = val_acc    / max(val_samples,   1)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Acc/Train_Epoch",  avg_train_acc,  epoch)
        writer.add_scalar("Loss/Val_Epoch",   avg_val_loss,   epoch)
        writer.add_scalar("Acc/Val_Epoch",    avg_val_acc,    epoch)

        print(
            f"Época {epoch+1}: Treino Loss={avg_train_loss:.4f} Acc={avg_train_acc:.4f} | "
            f"Val Loss={avg_val_loss:.4f} Acc={avg_val_acc:.4f}"
        )

        save_epoch_checkpoint(aligner, epoch + 1, name="pca_aligner_epoch")

        if controller(avg_val_loss, aligner.state_dict(), epoch):
            break

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    EPOCHS        = 20
    BATCH_SIZE    = 64
    K_PATCHES     = 196  # mesmo número de patches do noup (14×14 ViT)
    N_COMPONENTS  = 64   # nº de componentes principais usados no score de variância

    # Para retomar de um checkpoint salvo, aponte para o último .pth salvo e
    # ajuste START_EPOCH (0-based, = número da época do checkpoint, já que
    # ele foi salvo como epoch+1).
    RESUME_CHECKPOINT = None  # ex.: "checkpoints/pca_aligner_epoch_19.pth"
    START_EPOCH       = 0     # ex.: 19 para retomar a partir da época 20

    print("--- Starting PCA Aligner Training Pipeline ---")
    torch.cuda.empty_cache()
    print(
        f"VRAM livre antes do treino: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre"
    )
    try:
        train_lora_projection_pca(
            epochs=EPOCHS, batch_size=BATCH_SIZE, k_patches=K_PATCHES,
            n_components=N_COMPONENTS,
            resume_checkpoint=RESUME_CHECKPOINT, start_epoch=START_EPOCH,
        )
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print("Critical GPU error in main loop.")
            print(torch.cuda.memory_summary())
        raise
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
