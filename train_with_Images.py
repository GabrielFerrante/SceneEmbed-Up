import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import create_all_dataloaders
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from utils.checkpoint import save_epoch_checkpoint
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


def train_lora_projection(epochs: int = 10, batch_size: int = 2) -> None:
    """
    Treina o LoRACrossAttentionAligner fim-a-fim a partir de imagens e textos.

    Shapes principais
    -----------------
    images:        [B, 3, H, W]       — imagens transformadas pelo dataloader
    visual_input:  [B, N_patches, 768] — patches após DinoV3 + pool 32×32
    text_queries:  [B, 1, 4096]        — embedding Qwen por legenda
    attn_weights:  [B, num_heads, N_queries, N_patches]
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Hiperparâmetros de entropia ──────────────────────────────────────────
    # target=1.5 → foco em ~4-5 patches por head (era 2.5 → ~12 patches, difuso)
    # lambda=0.05 → ~15% do contrastivo, pressão real (era 0.01 → ignorado)
    TARGET_ENTROPY = 1.5
    LAMBDA_ENTROPY = 0.05
    # ────────────────────────────────────────────────────────────────────────

    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")

    # ── Modelos ──────────────────────────────────────────────────────────────
    dino_encoder = DinoSceneEncoder(device=device)
    qwen_embedder = QwenSceneEmbedder(device=device)

    autocast_dtype = qwen_embedder.dtype  # bfloat16

    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=16)
    aligner = aligner.to(device).to(autocast_dtype)

    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # ── Dataloaders ──────────────────────────────────────────────────────────
    train_dataloader, val_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=batch_size, num_workers=8, t="train"
    )

    global_step = 0
    controller = EarlyStopping()

    for epoch in range(epochs):
        aligner.train()
        epoch_loss, epoch_acc, train_samples = 0.0, 0.0, 0  # inicializados por época

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, texts in pbar:

            # ── Extração de features (sem grad — encoders congelados) ────────
            with torch.no_grad():
                features_list = []
                for img in images:
                    _, hr_feat = dino_encoder.extract_features(img.unsqueeze(0).to(device))
                    hr_feat_small = F.adaptive_avg_pool2d(hr_feat, (32, 32))
                    c = hr_feat_small.shape[1]
                    flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)
                    features_list.append(flat_feat.unsqueeze(0))

                visual_input = torch.cat(features_list, dim=0).to(autocast_dtype)  # [B, 1024, 768]

                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)
                if text_queries.dim() == 2:
                    text_queries = text_queries.unsqueeze(1)
                elif text_queries.dim() == 3 and text_queries.size(0) != visual_input.size(0):
                    text_queries = text_queries.view(visual_input.size(0), -1, text_queries.size(-1))
                # text_queries: [B, 1, 4096]

            # ── Forward + loss ───────────────────────────────────────────────
            try:
                with torch.amp.autocast(device_type="cuda", dtype=autocast_dtype):
                    visual_refined, attn_weights, v_features = aligner(visual_input, text_queries)
                    # visual_refined: [B, 1, 4096]
                    # attn_weights:   [B, num_heads, N_queries, N_patches]
                    # v_features:     [B, N_patches, text_dim]

                    v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)  # [B, 4096]
                    t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)  # [B, 4096]
                    logits = torch.matmul(v_norm, t_norm.T) / 0.07               # [B, B]
                    labels = torch.arange(v_norm.size(0), device=device)

                    loss_v = F.cross_entropy(logits,   labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    contrastive_loss = (loss_v + loss_t) / 2

                    # Supervisão direta em v_features (média dos patches)
                    v_global      = v_features.mean(dim=1)                        # [B, text_dim]
                    v_global_norm = F.normalize(v_global, p=2, dim=-1)
                    logits_vg     = torch.matmul(v_global_norm, t_norm.T) / 0.07
                    loss_vg       = (F.cross_entropy(logits_vg,   labels) +
                                     F.cross_entropy(logits_vg.T, labels)) / 2

                    # ── Regularização de entropia ────────────────────────────
                    entropy      = -torch.sum(
                        attn_weights * torch.log(attn_weights + 1e-8), dim=-1
                    )  # [B, num_heads, N_queries]
                    mean_entropy = entropy.mean()
                    entropy_reg  = (mean_entropy - TARGET_ENTROPY) ** 2

                    loss = contrastive_loss + 0.5 * loss_vg + LAMBDA_ENTROPY * entropy_reg
                    # ────────────────────────────────────────────────────────

                    with torch.no_grad():
                        acc = (torch.argmax(logits, dim=1) == labels).float().mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

            except RuntimeError as e:
                if "CUDA" in str(e).upper() and torch.cuda.is_available():
                    print(f"Erro de GPU durante o treinamento: {e}")
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
                writer.add_scalar("Loss/total",               loss.item(),             global_step)
                writer.add_scalar("Loss/contrastive",         contrastive_loss.item(), global_step)
                writer.add_scalar("Loss/contrastive_vfeats",  loss_vg.item(),          global_step)
                writer.add_scalar("Loss/entropy",             mean_entropy.item(),     global_step)
                writer.add_scalar("Loss/entropy_reg",         entropy_reg.item(),      global_step)
                writer.add_scalar("Acc/train_step",           acc.item(),              global_step)

        # ── Validação ────────────────────────────────────────────────────────
        aligner.eval()
        val_loss, val_acc, val_samples = 0.0, 0.0, 0

        print(f"Iniciando Validação Época {epoch+1}...")
        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validating"):

                features_list = []
                for img in images:
                    _, hr_feat = dino_encoder.extract_features(img.unsqueeze(0).to(device))
                    hr_feat_small = F.adaptive_avg_pool2d(hr_feat, (32, 32))
                    c = hr_feat_small.shape[1]
                    flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)
                    features_list.append(flat_feat.unsqueeze(0))

                visual_input = torch.cat(features_list, dim=0).to(autocast_dtype)
                text_queries = qwen_embedder.embed_components(
                    [[t] for t in texts], normalize=False
                )
                if text_queries.dim() == 2:
                    text_queries = text_queries.unsqueeze(1)

                try:
                    with torch.amp.autocast(device_type=device, dtype=autocast_dtype):
                        visual_refined, _, _ = aligner(visual_input, text_queries)  # desempacota tupla

                        v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                        t_norm = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)

                        logits = torch.matmul(v_norm, t_norm.T) / 0.07
                        labels = torch.arange(v_norm.size(0), device=device)

                        loss_v = F.cross_entropy(logits,   labels)
                        loss_t = F.cross_entropy(logits.T, labels)
                        loss   = (loss_v + loss_t) / 2
                        acc    = (torch.argmax(logits, dim=1) == labels).float().mean()

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(f"Erro de GPU durante a validação: {e}")
                        print(torch.cuda.memory_summary())
                    raise

                b = v_norm.size(0)
                val_loss    += loss.item() * b
                val_acc     += acc.item()  * b
                val_samples += b

        # ── Métricas finais da época ─────────────────────────────────────────
        avg_train_loss = epoch_loss / max(train_samples, 1)
        avg_train_acc  = epoch_acc  / max(train_samples, 1)
        avg_val_loss   = val_loss   / max(val_samples,   1)
        avg_val_acc    = val_acc    / max(val_samples,   1)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Acc/Train_Epoch",  avg_train_acc,  epoch)
        writer.add_scalar("Loss/Val_Epoch",   avg_val_loss,   epoch)
        writer.add_scalar("Acc/Val_Epoch",    avg_val_acc,    epoch)

        print(f"Época {epoch+1}:")
        print(f"  Treino -> Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")
        print(f"  Val    -> Loss: {avg_val_loss:.4f}   | Acc: {avg_val_acc:.4f}")

        save_epoch_checkpoint(aligner, epoch + 1)

        if controller(avg_val_loss, aligner.state_dict(), epoch):
            break

    writer.close()
    print("Treino finalizado!")


if __name__ == "__main__":
    EPOCHS     = 100
    BATCH_SIZE = 4   # conservador para caber DinoV3 + Qwen + Aligner na VRAM

    print("--- Iniciando Pipeline de Treinamento (imagens diretas) ---")

    torch.cuda.empty_cache()
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
          f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre")

    try:
        train_lora_projection(epochs=EPOCHS, batch_size=BATCH_SIZE)
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print("Erro crítico de GPU.")
            print(torch.cuda.memory_summary())
        raise
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")