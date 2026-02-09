import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from torch.cuda.amp import GradScaler # Para Mixed Precision


# Importando suas classes modularizadas
from models.SG.projection import LoRACrossAttentionAligner, calculate_retrieval_score
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from data.data_utils_pytorch import create_dataloader


def train_lora_projection(dataloader, epochs, batch_size, accumulation_steps=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler() # Gerencia a precisão automática

    # Inicializar modelos
    dino_encoder = DinoSceneEncoder(device=device) 
    qwen_embedder = QwenSceneEmbedder(device=device)
    
    # Inicializar o Alinhador
    # O visual_dim=768 (DINO) e text_dim=4096 (Qwen)
    # 768 para rodar com AnyUp
    # 384 para rodar com ViT FeatUp
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=16).to(device)
    
    # Sincroniza o dtype do aligner com o do Qwen (geralmente float16 ou bfloat16)
    target_dtype = qwen_embedder.dtype
    aligner = aligner.to(target_dtype)
    
    #  Apenas parâmetros treináveis (LoRA + CrossAttention)
    # Definimos requires_grad=False para a projeção base dentro do __init__ do Aligner
    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)
    

    print(f"Iniciando treino: {epochs} épocas, Batch Real: {batch_size * accumulation_steps}")

    for epoch in range(epochs):
        aligner.train()
        running_loss = 0.0
        optimizer.zero_grad()
    
        # tqdm para barra de progresso
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, (images, texts) in enumerate(pbar):
            # A. Extração de Features (Sem gradiente nos encoders pesados)
            with torch.no_grad():
                all_hr_features = []
                for img in images:
                    # Dino extrai features com o upsampler
                    _, hr_feat = dino_encoder.extract_features(img) 
                    C_dim = hr_feat.shape[1]
                    # Reshape variavel para [1, H*W, C_DIM]
                    all_hr_features.append(hr_feat.view(1, C_dim, -1).transpose(1, 2))
                
                # Junta o batch e converte para o dtype correto (FP16/BF16)
                visual_input = torch.cat(all_hr_features, dim=0).to(target_dtype)
                
                # Prepara o texto para o Qwen no formato esperado (lista de listas)
                # ex: [['gato'], ['mesa']]
                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)
                # text_queries shape: [Batch, 1, 4096]

            # B. Forward com Autocast
            with torch.amp.autocast(device_type='cuda', dtype=target_dtype):
                # Aligner processa Cross-Attention entre Query (Texto) e KV (Imagem)
                visual_refined = aligner(visual_input, text_queries) 
                
                # Extrai os vetores para cálculo de perda
                visual_projected = visual_refined.squeeze(1) # [B, 4096]
                text_target = text_queries.squeeze(1)      # [B, 4096]

                # Normalização para InfoNCE (Similiaridade de Cosseno)
                v_norm = F.normalize(visual_projected, p=2, dim=-1)
                t_norm = F.normalize(text_target, p=2, dim=-1)
                
                # Matriz de Logits [B, B]
                temperature = 0.07
                logits = torch.matmul(v_norm, t_norm.T) / temperature
                
                # Labels: a diagonal principal (imagem i deve bater com texto i)
                current_batch_size = visual_projected.size(0)
                labels = torch.arange(current_batch_size, device=device)
                
                # Loss Simétrica (CLIP-style)
                loss_v = F.cross_entropy(logits, labels)
                loss_t = F.cross_entropy(logits.T, labels)
                loss = (loss_v + loss_t) / 2
                
                # Ajuste por accumulation steps
                loss = loss / accumulation_steps

            # C. Backward com Scaler (Mixed Precision)
            scaler.scale(loss).backward()

            # D. Update do Otimizador
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss.item() * accumulation_steps
            pbar.set_postfix({'loss': running_loss / (i + 1)})

    # 5. Salvar pesos treináveis
    # Filtramos apenas o que não está congelado
    save_dict = {k: v for k, v in aligner.state_dict().items() if "lora_" in k or "cross_attn" in k}
    torch.save(save_dict, "lora_cross_aligner_weights.pth")
    print("Treino finalizado e pesos salvos!")
    

if __name__ == "__main__":
    # 1. Configurações de hiperparâmetros
    EPOCHS = 10
    BATCH_SIZE = 4 # Ajustado para segurança de memória com AnyUp
    ACCUMULATION_STEPS = 8 # Batch Real = 4 * 8 = 32
    
    # 2. Caminho para os dados (Ajuste conforme sua estrutura)
    DATA_PATH = "F:/COYO/coyo/extracted"
    
    # Nota: O seu create_dataloader parece já estar definido no arquivo data_utils_pytorch
    # Se você quiser passar uma lista manual de dados em vez do caminho da pasta, 
    # use o ProjectionDataset que criamos antes.
    
    coyo_loader = create_dataloader("F:/COYO/coyo/extracted", batch_size=8, num_workers=4)
    
    print("--- Iniciando Pipeline de Treinamento SceneGraph ---")
    
    try:
        # Chamada da função de treino
        train_lora_projection(
            dataloader= coyo_loader, 
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            accumulation_steps=ACCUMULATION_STEPS
        )
    except Exception as e:
        print(f"Erro durante o treinamento: {e}")