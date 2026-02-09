
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Para logs
from tqdm import tqdm
import datetime
import os

from models.SG.projection import LoRACrossAttentionAligner, calculate_retrieval_score
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from data.data_utils_pytorch import create_all_dataloaders



def train_lora_projection(epochs=10, batch_size=2, accumulation_steps=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler() 

    # 1. Configuração de Logs
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    writer = SummaryWriter(log_dir=log_dir)
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

    train_dataloader, val_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=batch_size, num_workers=4, t="train")

    global_step = 0 # Para o TensorBoard

    for epoch in range(epochs):
        aligner.train()
        epoch_loss = 0.0
        epoch_acc = 0.0 # Acurácia de recuperação (Retrieval Acc)
        
        optimizer.zero_grad()
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (images, texts) in enumerate(pbar):
            with torch.no_grad():
                features_list = []
                for img in images:
                    _, hr_feat = dino_encoder.extract_features(img) 
                    features_list.append(hr_feat.reshape(1, hr_feat.shape[1], -1).transpose(1, 2))
                
                visual_input = torch.cat(features_list, dim=0).to(target_dtype)
                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)

            # B. Forward com Autocast
            with torch.amp.autocast(device_type='cuda', dtype=target_dtype):
                visual_refined = aligner(visual_input, text_queries) 
                visual_projected = visual_refined.squeeze(1) 
                text_target = text_queries.squeeze(1)

                v_norm = F.normalize(visual_projected, p=2, dim=-1)
                t_norm = F.normalize(text_target, p=2, dim=-1)
                
                temperature = 0.07
                logits = torch.matmul(v_norm, t_norm.T) / temperature
                
                current_batch_size = visual_projected.size(0)
                labels = torch.arange(current_batch_size, device=device)
                
                # C. Cálculo das Métricas
                loss_v = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                loss = (loss_v + loss_t) / 2
                
                # Métrica: Acurácia do Batch (Quantos itens a diagonal é o maior valor)
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean()

                loss_scaled = loss / accumulation_steps

            # D. Backward
            scaler.scale(loss_scaled).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Log no TensorBoard a cada step de otimização
                writer.add_scalar("Loss/train_step", loss.item(), global_step)
                writer.add_scalar("Acc/train_step", acc.item(), global_step)
                global_step += 1

            epoch_loss += loss.item() * accumulation_steps
            epoch_acc += acc.item()
            pbar.set_postfix({'loss': loss.item(), 'acc': acc.item()})
            
        #VALIDAÇÃO        
        aligner.eval()
        val_loss, val_acc = 0.0, 0.0
        
        print(f"Iniciando Validação Época {epoch+1}...")
        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validating"):
                # Extração (Mesma lógica do treino)
                features_list = []
                for img in images:
                    _, hr_feat = dino_encoder.extract_features(img) 
                    features_list.append(hr_feat.reshape(1, hr_feat.shape[1], -1).transpose(1, 2))
                
                visual_input = torch.cat(features_list, dim=0).to(target_dtype)
                text_queries = qwen_embedder.embed_components([[t] for t in texts], normalize=False)

                # Forward Sem Gradiente
                with torch.amp.autocast(device_type='cuda', dtype=target_dtype):
                    visual_refined = aligner(visual_input, text_queries) 
                    v_norm = F.normalize(visual_refined.squeeze(1), p=2, dim=-1)
                    t_norm = F.normalize(text_queries.squeeze(1), p=2, dim=-1)
                    
                    logits = torch.matmul(v_norm, t_norm.T) / 0.07
                    labels = torch.arange(v_norm.size(0), device=device)
                    
                    loss_v = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    loss = (loss_v + loss_t) / 2
                    
                    acc = (torch.argmax(logits, dim=1) == labels).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()

        # --- LOGS FINAIS DA ÉPOCA ---
        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_train_acc = epoch_acc / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        avg_val_acc = val_acc / len(val_dataloader)

        writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
        writer.add_scalar("Acc/Train_Epoch", avg_train_acc, epoch)
        writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)
        writer.add_scalar("Acc/Val_Epoch", avg_val_acc, epoch)

        print(f"Época {epoch+1}:")
        print(f"  Treino -> Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.4f}")
        print(f"  Val    -> Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.4f}")
        
        # Salvar 
        torch.save(aligner.state_dict(), f"checkpoints/aligner_epoch_{epoch+1}.pth")

    writer.close()
    print("Treino finalizado!")
    

if __name__ == "__main__":
    # 1. Configurações de hiperparâmetros
    EPOCHS = 10
    BATCH_SIZE = 4 # Ajustado para segurança de memória com AnyUp
    ACCUMULATION_STEPS = 8 # Batch Real = 4 * 8 = 32

    
    print("--- Iniciando Pipeline de Treinamento SceneGraph ---")
    
    try:
        # Chamada da função de treino
        train_lora_projection( 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            accumulation_steps=ACCUMULATION_STEPS
        )
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")