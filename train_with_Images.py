
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import create_all_dataloaders
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner, calculate_retrieval_score
from utils.checkpoint import save_epoch_checkpoint
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


def train_lora_projection(epochs: int = 10, batch_size: int = 2) -> None:
    """
    Treina o `LoRACrossAttentionAligner` fim‑a‑fim a partir de imagens e textos.

    Shapes principais
    -----------------
    images:
        `[B, 3, H, W]` — imagens já transformadas pelo `create_all_dataloaders`.
    visual_input:
        `[B, N_patches, 768]` — patches visuais após extração Dino + pooling 32×32.
    text_queries:
        `[B, 1, 4096]` — embedding textual do Qwen por legenda.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Configuração de Logs
    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")

    # 2. Inicializar modelos
    dino_encoder = DinoSceneEncoder(device=device) 
    qwen_embedder = QwenSceneEmbedder(device=device)
    
    # visual_dim=768 (AnyUp) ou 384 (FeatUp)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=16).to(device)
    target_dtype = qwen_embedder.dtype
    aligner = aligner.to(target_dtype)
    
    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    train_dataloader, val_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=batch_size, num_workers=8, t="train"
    )
    global_step = 0 # Para o TensorBoard
    controller = EarlyStopping()
    batchs = 0
    for epoch in range(epochs):
        aligner.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for i, (images, texts) in enumerate(pbar):
            with torch.no_grad():
                features_list = []
                for img in images:
                    _, hr_feat = dino_encoder.extract_features(img.unsqueeze(0).to(device))
                    hr_feat_small = torch.nn.functional.adaptive_avg_pool2d(hr_feat, (32, 32))
                    c, h, w = hr_feat_small.shape[1:]
                    flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)
                    features_list.append(flat_feat.unsqueeze(0))

                visual_input = torch.cat(features_list, dim=0).to(target_dtype)
                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)
                if text_queries.dim() == 2:
                    text_queries = text_queries.unsqueeze(1)
                elif text_queries.dim() == 3 and text_queries.size(0) != visual_input.size(0):
                    text_queries = text_queries.view(visual_input.size(0), -1, text_queries.size(-1))

            lambda_entropy = 0.01

            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    visual_refined, attn_weights, v_features = aligner(visual_input, text_queries)

                    v_norm  = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                    t_norm  = F.normalize(text_queries.squeeze(1),   p=2, dim=-1)
                    logits  = torch.matmul(v_norm, t_norm.T) / 0.07
                    labels  = torch.arange(v_norm.size(0), device=device)
                    loss_v  = F.cross_entropy(logits,   labels)
                    loss_t  = F.cross_entropy(logits.T, labels)
                    contrastive_loss = (loss_v + loss_t) / 2

                    # Loss novo — supervisa v_features diretamente:
                    v_global      = v_features.mean(dim=1)                     # [B, text_dim]
                    v_global_norm = F.normalize(v_global, p=2, dim=-1)
                    logits_vg     = torch.matmul(v_global_norm, t_norm.T) / 0.07
                    loss_vg       = (F.cross_entropy(logits_vg,   labels) +
                                    F.cross_entropy(logits_vg.T, labels)) / 2

                    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                    mean_entropy = entropy.mean()
                    entropy_reg = (mean_entropy - 2.5) ** 2
                    # Loss final:
                    loss = contrastive_loss + 0.5 * loss_vg + 0.01 * entropy_reg

                    with torch.no_grad():
                        preds = torch.argmax(logits, dim=1)
                        acc = (preds == labels).float().mean()

                if global_step % 10 == 0:
                    print(f"Loss: {loss.item():.4f} | Entropy: {mean_entropy.item():.4f} | Acc: {acc.item():.2f}")

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("Loss/total", loss.item(), global_step)
                writer.add_scalar("Loss/contrastive", contrastive_loss.item(), global_step)
                writer.add_scalar("Loss/entropy", mean_entropy.item(), global_step)
                writer.add_scalar("Acc/train_step", acc.item(), global_step)
                writer.add_scalar("Loss/contrastive_attn",    contrastive_loss.item(), global_step)
                writer.add_scalar("Loss/contrastive_vfeats",  loss_vg.item(),          global_step)

                global_step += 1
            except RuntimeError as e:
                if "CUDA" in str(e).upper():
                    print(f"Erro de GPU durante o treinamento: {e}")
                    if torch.cuda.is_available():
                        print(torch.cuda.memory_summary())
                raise

            # 3. Métricas da Época (Usando a soma ponderada)
            # Aqui usamos a loss "cheia" para ter a média real das imagens
            epoch_loss += loss.item() * visual_input.size(0)
            epoch_acc += acc.item() * visual_input.size(0)
            processed_samples += visual_input.size(0)

            # 4. Feedback visual
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}", 
                'acc': f"{acc.item():.2f}"
            })
        # VALIDAÇÃO
        aligner.eval()
        val_loss, val_acc = 0.0, 0.0
        print(f"Iniciando Validação Época {epoch+1}...")
        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validating"):
                features_list = []

                for img in images:
                    img = img.to(device)

                    _, hr_feat = dino_encoder.extract_features(img)
                    hr_feat_small = torch.nn.functional.adaptive_avg_pool2d(hr_feat, (32, 32))
                    c, h, w = hr_feat_small.shape[1:]
                    flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)

                    features_list.append(flat_feat.unsqueeze(0))

                visual_input = torch.cat(features_list, dim=0).to(device)

                text_queries = qwen_embedder.embed_components([[t] for t in texts], normalize=False).to(device)

                try:
                    with torch.amp.autocast(device_type=device, dtype=target_dtype):
                        visual_refined = aligner(visual_input, text_queries)

                        v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                        t_norm = F.normalize(text_queries.squeeze(1), p=2, dim=-1)

                        logits = torch.matmul(v_norm, t_norm.T) / 0.07

                        labels = torch.arange(v_norm.size(0), device=device)

                        loss_v = F.cross_entropy(logits, labels)
                        loss_t = F.cross_entropy(logits.T, labels)
                        loss = (loss_v + loss_t) / 2

                        acc = (torch.argmax(logits, dim=1) == labels).float().mean()
                except RuntimeError as e:
                    if "CUDA" in str(e).upper():
                        print(f"Erro de GPU durante a validação: {e}")
                        if torch.cuda.is_available():
                            print(torch.cuda.memory_summary())
                    raise

                val_loss += loss.item()
                val_acc += acc.item()
                num_batches += 1

        
        avg_train_loss = epoch_loss / batchs
        avg_train_acc = epoch_acc / batchs
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_acc / len(val_dataloader)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Acc/Train_Epoch", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Acc/Val_Epoch", avg_val_acc, epoch)

        print(f"Época {epoch+1}:")
        print(f"  Treino -> Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")
        print(f"  Val    -> Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")
        
        save_epoch_checkpoint(aligner, epoch + 1)
        
        
        
        # O controller decide se o treino deve parar
        stop_now = controller(avg_val_loss, aligner.state_dict(), epoch)
    
        if stop_now:
            break # Sai do loop de épocas
        

    writer.close()
    print("Treino finalizado!")
    

if __name__ == "__main__":
    # 1. Configurações de hiperparâmetros
    EPOCHS = 100
    BATCH_SIZE = 4 # Ajustado para segurança de memória com AnyUp

    
    print("--- Iniciando Pipeline de Treinamento da camada de projeção ---")

    try:
        train_lora_projection(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print("Erro crítico de GPU no loop principal.")
            print(torch.cuda.memory_summary())
        raise
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")
