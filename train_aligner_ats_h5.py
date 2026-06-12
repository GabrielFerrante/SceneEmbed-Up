"""
Treino do LoRACrossAttentionAligner com seleção adaptativa de tokens (ATS).

Inspirado em:
    Fayyaz, M. et al. (2022). Adaptive Token Sampling For Efficient Vision
    Transformers. ECCV 2022.

O AdaptiveTokenSampler reduz 1024 patches anyup → K=196 patches selecionados
por relevância (text-guided + content-based scoring), atacando o colapso de
manifold identificado pelo ID local (ID_anyup ≈ 1.88 vs ID_noup ≈ 5.0).

Fluxo de gradientes
-------------------
- Loss contrastiva → aligner (visual_proj, LoRA, cross_attn)
- ATS BCE loss     → sampler (content_proj, text_gate)   [straight-through]

Checkpoints
-----------
    checkpoints/ats_aligner_epoch_N.pth   — por época
    checkpoints/best_ats_aligner.pth      — melhor val loss
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


# ── Adaptive Token Sampler ────────────────────────────────────────────────────

class AdaptiveTokenSampler(nn.Module):
    """
    Seleciona K tokens visuais a partir de N via scoring text-guided + content.

    Inspirado em Fayyaz et al. (2022). Adaptive Token Sampling For Efficient
    Vision Transformers. ECCV 2022.

    Na formulação original, os scores são derivados da atenção do class token
    em cada camada do ViT. Aqui, como os embeddings são pré-computados (sem
    class token explícito), o score combina:
      - content_proj : saliência intrínseca do patch (norma de ativação)
      - text_gate    : relevância do patch para a query textual (dot-product)

    O scorer é treinado via BCE-with-logits straight-through: na forward pass,
    usa-se o top-K hard (não diferenciável); a BCE entre os logits e o target
    binário (1=selecionado, 0=descartado) fornece gradiente direto ao scorer.

    Shapes
    ------
    patches      : [B, N, visual_dim]
    text_queries : [B, N_q, text_dim]
    → selected   : [B, K, visual_dim]
    → scores     : [B, N]   (logits, pré-sigmoid)
    → top_k_idx  : [B, K]
    """

    def __init__(
        self,
        visual_dim: int = 768,
        text_dim: int = 4096,
        k: int = 196,
    ) -> None:
        super().__init__()
        self.k = k
        self.content_proj = nn.Linear(visual_dim, 1)
        self.text_gate    = nn.Linear(text_dim, visual_dim, bias=False)

    def forward(
        self,
        patches: torch.Tensor,
        text_queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # patches:      [B, N, 768]
        # text_queries: [B, 1, 4096]
        t             = self.text_gate(text_queries)                     # [B, 1, 768]
        content_score = self.content_proj(patches)                       # [B, N, 1]
        text_score    = (patches * t).sum(dim=-1, keepdim=True)          # [B, N, 1]
        scores        = (content_score + text_score).squeeze(-1)         # [B, N] (logits)

        _, top_k_idx = torch.topk(scores, k=self.k, dim=-1)              # [B, K]
        idx_exp  = top_k_idx.unsqueeze(-1).expand(-1, -1, patches.size(-1))
        selected = torch.gather(patches, dim=1, index=idx_exp)          # [B, K, 768]

        return selected, scores, top_k_idx


# ── Loop de treino ────────────────────────────────────────────────────────────

def train_lora_projection_ats(
    epochs: int = 20,
    batch_size: int = 64,
    k_patches: int = 196,
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
    sampler = AdaptiveTokenSampler(visual_dim=768, text_dim=4096, k=k_patches)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    sampler = sampler.to(device).to(autocast_dtype)
    aligner = aligner.to(device).to(autocast_dtype)

    # ModuleDict para salvar checkpoint unificado (sampler + aligner)
    combined = nn.ModuleDict({"sampler": sampler, "aligner": aligner})

    if resume_checkpoint is not None:
        combined.load_state_dict(torch.load(resume_checkpoint, map_location=device))
        print(f"Checkpoint carregado: {resume_checkpoint} (retomando da época {start_epoch + 1})")

    trainable_params = [p for p in combined.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # ── Hiperparâmetros ───────────────────────────────────────────────────────
    TARGET_ENTROPY = 1.5
    LAMBDA_ENTROPY = 0.05
    LAMBDA_ATS     = 0.1   # peso da ATS selection loss

    global_step = 0
    controller  = EarlyStopping(patience=10, filename="best_ats_aligner.pth")
    num_shards  = len(train_ds._all_shards)

    for epoch in range(start_epoch, epochs):
        aligner.train()
        sampler.train()
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
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):

                        # ── ATS: seleciona K patches por relevância ──────────
                        # visual_input: [B, 1024, 768]
                        # selected:     [B, K,    768]
                        selected, scores, top_k_idx = sampler(visual_input, text_queries)

                        # ATS selection loss (straight-through BCE-with-logits)
                        # Provê gradiente direto ao scorer via scores (logits):
                        #   scores altos → patches selecionados (target=1)
                        #   scores baixos → patches descartados (target=0)
                        selection_target = torch.zeros_like(scores)
                        selection_target.scatter_(1, top_k_idx, 1.0)
                        ats_loss = F.binary_cross_entropy_with_logits(
                            scores.float(),
                            selection_target.detach().float(),
                        )

                        # ── Aligner sobre os K patches selecionados ──────────
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

                        loss = contrastive_loss + LAMBDA_ENTROPY * entropy_reg + LAMBDA_ATS * ats_loss

                        with torch.no_grad():
                            v_global      = v_features.mean(dim=1)
                            v_global_norm = F.normalize(v_global, p=2, dim=-1)
                            logits_vg     = torch.matmul(v_global_norm, t_norm.T) / 0.07
                            loss_vg       = (F.cross_entropy(logits_vg,   labels) +
                                            F.cross_entropy(logits_vg.T, labels)) / 2
                            acc            = (torch.argmax(logits, dim=1) == labels).float().mean()
                            score_sparsity = (torch.sigmoid(scores) > 0.5).float().mean()

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
                    ats=f"{ats_loss.item():.4f}",
                    sp=f"{score_sparsity.item():.2f}",
                )

                if global_step % 10 == 0:
                    writer.add_scalar("Loss/total",              loss.item(),              global_step)
                    writer.add_scalar("Loss/contrastive",        contrastive_loss.item(),  global_step)
                    writer.add_scalar("Loss/entropy",            mean_entropy.item(),      global_step)
                    writer.add_scalar("Loss/entropy_reg",        entropy_reg.item(),       global_step)
                    writer.add_scalar("Loss/ats",                ats_loss.item(),          global_step)
                    writer.add_scalar("Loss/contrastive_vfeats", loss_vg.item(),           global_step)
                    writer.add_scalar("Acc/train_step",          acc.item(),               global_step)
                    writer.add_scalar("Metric/score_sparsity",   score_sparsity.item(),    global_step)

        train_ds.rotate_buffer()

        # ── Validação ─────────────────────────────────────────────────────────
        aligner.eval()
        sampler.eval()
        val_loss, val_acc, val_samples = 0.0, 0.0, 0

        with torch.no_grad():
            for visual_input, text_queries in tqdm(val_dataloader, desc="Validating"):
                visual_input = visual_input.to(device).to(autocast_dtype)
                text_queries = text_queries.to(device).to(autocast_dtype)

                with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                    selected, _, _ = sampler(visual_input, text_queries)
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

        save_epoch_checkpoint(combined, epoch + 1, name="ats_aligner_epoch")

        if controller(avg_val_loss, combined.state_dict(), epoch):
            break

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    EPOCHS     = 20
    BATCH_SIZE = 64
    K_PATCHES  = 196  # mesmo número de patches do noup (14×14 ViT)
    #K_PATCHES  = 256
    #K_PATCHES  = 512

    # Para retomar de um checkpoint salvo (ex.: crash na validação da última
    # época), aponte para o último .pth salvo e ajuste START_EPOCH (0-based,
    # = número da época do checkpoint, já que ele foi salvo como epoch+1).
    RESUME_CHECKPOINT = "checkpoints/ats_aligner_epoch_19.pth"  # ex.: "checkpoints/ats_aligner_epoch_19.pth"
    START_EPOCH       = 19     # ex.: 19 para retomar a partir da época 20

    print("--- Starting ATS Aligner Training Pipeline ---")
    torch.cuda.empty_cache()
    print(
        f"VRAM livre antes do treino: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre"
    )
    try:
        train_lora_projection_ats(
            epochs=EPOCHS, batch_size=BATCH_SIZE, k_patches=K_PATCHES,
            resume_checkpoint=RESUME_CHECKPOINT, start_epoch=START_EPOCH,
        )
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print("Critical GPU error in main loop.")
            print(torch.cuda.memory_summary())
        raise
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
