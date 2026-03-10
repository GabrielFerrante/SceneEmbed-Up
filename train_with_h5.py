
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from utils.checkpoint import save_epoch_checkpoint
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from utils.early_stopping import EarlyStopping
from utils.logging_utils import create_tensorboard_writer


def train_lora_projection(epochs: int = 10, batch_size: int = 2) -> None:
    """
    Treina o `LoRACrossAttentionAligner` a partir de embeddings pré‑computados em HDF5.

    Shapes principais
    -----------------
    visual_input:
        `[B, N_patches, 768]` — patches visuais já projetados na resolução 32×32.
    text_queries:
        `[B, 1, 4096]` — embedding textual do Qwen para cada legenda.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Configuração de Logs
    writer, log_dir = create_tensorboard_writer(log_root="logs")
    print(f"Logs salvos em: {log_dir}")

    # 2. Inicializar modelos
    #dino_encoder = DinoSceneEncoder(device=device) 
    qwen_embedder = QwenSceneEmbedder(device=device)
    
    # visual_dim=768 (AnyUp) ou 384 (FeatUp)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=16).to(device)
    target_dtype = qwen_embedder.dtype
    aligner = aligner.to(target_dtype)
    
    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
   
    # 3. EMBEDDINGS JÁ CALCULADAS (H5)
    train_ds= ShardedH5Dataset("F:/COYO/embeds/train_anyup/")
    train_dataloader = DataLoader(train_ds, batch_size=16, shuffle=True, pin_memory=True)
    val_ds = ShardedH5Dataset("F:/COYO/embeds/val_anyup/")
    val_dataloader = DataLoader(val_ds, batch_size=16, shuffle=True, pin_memory=True)
    
    

    global_step = 0 # Para o TensorBoard
    controller = EarlyStopping()
    batchs = 0
    for epoch in range(epochs):
        
        epoch_loss = 0.0
        epoch_acc = 0.0 # Acurácia de recuperação (Retrieval Acc)
        
        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        
        
        aligner.train()
        for i, (visual_input, text_queries) in enumerate(pbar):
            visual_input = visual_input.to(device)
            text_queries = text_queries.to(device)
            lambda_entropy = 0.01

            try:
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    visual_refined, attn_weights = aligner(visual_input, text_queries)

                    visual_projected = visual_refined.squeeze(1)
                    text_target = text_queries.squeeze(1)

                    v_norm = F.normalize(visual_projected, p=2, dim=-1)
                    t_norm = F.normalize(text_target, p=2, dim=-1)

                    temperature = 0.07
                    logits = torch.matmul(v_norm, t_norm.T) / temperature

                    current_batch_size = visual_projected.size(0)
                    labels = torch.arange(current_batch_size, device=device)

                    loss_v = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    contrastive_loss = (loss_v + loss_t) / 2

                    entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                    mean_entropy = entropy.mean()

                    target_entropy = 2.5
                    entropy_reg = (mean_entropy - target_entropy) ** 2
                    loss = contrastive_loss + lambda_entropy * entropy_reg

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
            batchs += visual_input.size(0)

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
            for visual_input, text_queries in tqdm(val_dataloader, desc="Validating"):
                visual_input = visual_input.to(device)
                text_queries = text_queries.to(device)

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
                batchs += 1

        # --- LOGS FINAIS DA ÉPOCA ---
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
    EPOCHS = 100
    BATCH_SIZE = 4  # Ajustado para segurança de memória com AnyUp

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
