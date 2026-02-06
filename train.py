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

class ProjectionDataset(Dataset):
    def __init__(self, data_list):
        """
        data_list: Lista de dicionários [{'image': PIL_IMG, 'text': 'gato preto'}, ...]
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Retorna a imagem PIL e a string de texto
        return self.data[idx]['image'], self.data[idx]['text']

def train_lora_projection(data_list, epochs=10, batch_size=2, accumulation_steps=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler = torch.amp.GradScaler() # Gerencia a precisão automática

    # 1. Inicializar modelos
    dino_encoder = DinoSceneEncoder(device=device) 
    qwen_embedder = QwenSceneEmbedder(device=device)
    
    # 2. Inicializar o Alinhador
    # O visual_dim=768 (DINO) e text_dim=4096 (Qwen)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=16).to(device)
    
    # Sincroniza o dtype do aligner com o do Qwen (geralmente float16 ou bfloat16)
    target_dtype = qwen_embedder.dtype
    aligner = aligner.to(target_dtype)
    
    # 3. FILTRO: Apenas parâmetros treináveis (LoRA + CrossAttention)
    # Definimos requires_grad=False para a projeção base dentro do __init__ do Aligner
    trainable_params = [p for p in aligner.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    # 4. DataLoader
    dataset = ProjectionDataset(data_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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
                    # Dino extrai features com AnyUp
                    _, hr_feat = dino_encoder.extract_features(img) 
                    # Reshape de [1, 768, H, W] para [1, H*W, 768]
                    all_hr_features.append(hr_feat.view(1, 768, -1).transpose(1, 2))
                
                # Junta o batch e converte para o dtype correto (FP16/BF16)
                visual_input = torch.cat(all_hr_features, dim=0).to(target_dtype)
                
                # Prepara o texto para o Qwen no formato esperado (lista de listas)
                # ex: [['gato'], ['mesa']]
                formatted_texts = [[t] for t in texts]
                text_queries = qwen_embedder.embed_components(formatted_texts, normalize=False)
                # text_queries shape: [Batch, 1, 4096]

            # B. Forward com Autocast
            with torch.amp.autocast():
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