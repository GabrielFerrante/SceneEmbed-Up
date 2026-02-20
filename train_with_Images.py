
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

class EarlyStopping:
    def __init__(self, save_dir="checkpoints", filename="best_aligner.pth", patience=5, min_delta=0.001):
        self.save_dir = save_dir
        self.filename = filename
        self.patience = patience      # Quantas épocas esperar sem melhora
        self.min_delta = min_delta    # Redução mínima necessária para considerar uma melhora
        self.best_loss = float('inf') # Inicializado com infinito para qualquer loss ser menor
        self.counter = 0              # Contador de épocas sem melhora
        self.early_stop = False
        
        os.makedirs(save_dir, exist_ok=True)

    def __call__(self, current_loss, model_state, epoch):
        # Para a Loss, queremos que o valor atual seja MENOR que o melhor anterior menos o delta
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            path = os.path.join(self.save_dir, self.filename)
            torch.save(model_state, path)
            print(f" >>> Época {epoch+1}: Novo melhor modelo salvo! (Loss: {current_loss:.4f})")
        else:
            self.counter += 1
            print(f" >>> Época {epoch+1}: Sem melhora na Loss. ({self.counter}/{self.patience})")
            
            if self.counter >= self.patience:
                self.early_stop = True
                print(" !!! Early Stopping acionado! Encerrando treinamento.")
        
        return self.early_stop


def train_lora_projection(epochs=10, batch_size=2):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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



    train_dataloader, val_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=batch_size, num_workers=8, t="train")

    
    

    global_step = 0 # Para o TensorBoard
    controller = EarlyStopping()
    batchs = 0
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
                    # 1. Extração: retorna [1, 768, 224, 224]
                    _, hr_feat = dino_encoder.extract_features(img.unsqueeze(0).to("cuda")) 
                    
                    # 2. Pooling
                    # Reduzimos aqui para 32x32
                    hr_feat_small = torch.nn.functional.adaptive_avg_pool2d(hr_feat, (32, 32))
                    
                    # 3. Squeeze e Transpose: [1, 768, 32, 32] -> [768, 1024] -> [1024, 768]
                    # Aqui removemos o batch do loop para achatar
                    c, h, w = hr_feat_small.shape[1:]
                    flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)
                    
                    features_list.append(flat_feat.unsqueeze(0))
                
                visual_input = torch.cat(features_list, dim=0).to(target_dtype)
                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)
                if text_queries.dim() == 2:
                    # Se for [B, Dim], transforma em [B, 1, Dim]
                    text_queries = text_queries.unsqueeze(1)
                elif text_queries.dim() == 3 and text_queries.size(0) != visual_input.size(0):
                    # Se o Qwen retornou tudo colapsado [Total_Componentes, Seq, Dim], 
                    # reorganize para bater com o batch_size do visual
                    text_queries = text_queries.view(visual_input.size(0), -1, text_queries.size(-1))

            lambda_entropy = 0.01 

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # 1. Forward com retorno dos pesos de atenção
                visual_refined, attn_weights = aligner(visual_input, text_queries) 
                
                visual_projected = visual_refined.squeeze(1) 
                text_target = text_queries.squeeze(1)

                v_norm = F.normalize(visual_projected, p=2, dim=-1)
                t_norm = F.normalize(text_target, p=2, dim=-1)
                
                temperature = 0.07
                logits = torch.matmul(v_norm, t_norm.T) / temperature
                
                current_batch_size = visual_projected.size(0)
                labels = torch.arange(current_batch_size, device=device)
                
                # 2. Perda Contrastiva (CLIP-style)
                loss_v = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                contrastive_loss = (loss_v + loss_t) / 2
                
                # --- INÍCIO DA REGULARIZAÇÃO DA ENTROPIA ---
                # attn_weights shape esperado: [Batch, Num_Queries, Num_Patches]
                # Adicionamos um epsilon (1e-8) para evitar log(0)
                entropy = -torch.sum(attn_weights * torch.log(attn_weights + 1e-8), dim=-1)
                mean_entropy = entropy.mean()
                
                # Perda Total = Contraste + (Lambda * Entropia)
                # Minimizar a entropia força a atenção a ser mais "focada" (spiky)
                
                target_entropy = 2.5  # exemplo
                entropy_reg = (mean_entropy - target_entropy) ** 2
                loss = contrastive_loss + lambda_entropy * entropy_reg
                # --- FIM DA REGULARIZAÇÃO ---

                # C. Cálculo das Métricas (sem gradiente)
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    acc = (preds == labels).float().mean()

                # Log das métricas para monitoramento
                if global_step % 10 == 0:
                    print(f"Loss: {loss.item():.4f} | Entropy: {mean_entropy.item():.4f} | Acc: {acc.item():.2f}")

                optimizer.zero_grad() 
                loss.backward()      
                optimizer.step()      
                
                # Logging no TensorBoard
                writer.add_scalar("Loss/total", loss.item(), global_step) 
                writer.add_scalar("Loss/contrastive", contrastive_loss.item(), global_step)
                writer.add_scalar("Loss/entropy", mean_entropy.item(), global_step)
                writer.add_scalar("Acc/train_step", acc.item(), global_step)
                
                global_step += 1

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
        #VALIDAÇÃO        
        aligner.eval()
        val_loss, val_acc = 0.0, 0.0
        print(f"Iniciando Validação Época {epoch+1}...")
        with torch.no_grad():
            for images, texts in tqdm(val_dataloader, desc="Validating"):

                features_list = []

                for img in images:
                    img = img.to(device)

                    _, hr_feat = dino_encoder.extract_features(img)

                    # (1, C, H, W) → (1, HW, C)
                    feat = hr_feat.reshape(1, hr_feat.shape[1], -1).transpose(1, 2)

                    features_list.append(feat)

                visual_input = torch.cat(features_list, dim=0).to(device)

                
                text_queries = qwen_embedder.embed_components(
                    [[t] for t in texts],
                    normalize=False
                ).to(device)

                
                with torch.amp.autocast( device_type=device, dtype=target_dtype):

                    visual_refined = aligner(
                        visual_input,
                        text_queries
                    )

                    v_norm = F.normalize(
                        visual_refined.squeeze(1),
                        p=2,
                        dim=-1
                    )

                    t_norm = F.normalize(
                        text_queries.squeeze(1),
                        p=2,
                        dim=-1
                    )

                    logits = torch.matmul(v_norm, t_norm.T) / 0.07

                    labels = torch.arange(
                        v_norm.size(0),
                        device=device
                    )

                    loss_v = F.cross_entropy(logits, labels)
                    loss_t = F.cross_entropy(logits.T, labels)
                    loss = (loss_v + loss_t) / 2

                    acc = (
                        torch.argmax(logits, dim=1) == labels
                    ).float().mean()

                val_loss += loss.item()
                val_acc += acc.item()
                num_batches += 1

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
        
        torch.save(aligner.state_dict(), f"checkpoints/aligner_epoch_{epoch+1}.pth")
        
        
        
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
        # Chamada da função de treino
        train_lora_projection( 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
        )
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")