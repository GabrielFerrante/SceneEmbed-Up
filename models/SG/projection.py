import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRACrossAttentionAligner(nn.Module):
    def __init__(self, visual_dim=768, text_dim=4096, rank=16, num_heads=8):
        super().__init__()
        
        # 1. Projeção Base da Imagem (Congelada)
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        #BUG FIX: Certifique-se de congelar o Bias também
        self.visual_proj.weight.requires_grad = False
        if self.visual_proj.bias is not None:
            self.visual_proj.bias.requires_grad = False
        
        # 2. Camada de Cross-Attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim, 
            num_heads=num_heads, 
            batch_first=True
        )
        
        # 3. LoRA (A e B)
        self.lora_A_v = nn.Parameter(torch.empty(visual_dim, rank))
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5))
        self.lora_B_v = nn.Parameter(torch.zeros(rank, text_dim))
        
        self.scaling = 32 / rank # alpha = 32

    def forward(self, hr_patches, text_queries):
        # A. Projeção Visual com LoRA
        # [B, 658560, 4096]
        base_v = self.visual_proj(hr_patches)
        
        # Otimização de operação: (x @ A) @ B
        lora_v = (hr_patches @ self.lora_A_v) @ self.lora_B_v
        v_features = base_v + (lora_v * self.scaling)
        
        # B. Cross-Attention
        # Usamos context manager para garantir eficiência de memória se possível
        attn_output, attn_weights = self.cross_attn(
            query=text_queries,       
            key=v_features,           
            value=v_features          
        )
        
        return attn_output # [Batch, N_termos, 4096]

def calculate_retrieval_score(visual_aligned, text_embedding):
    """
    visual_aligned: Saída do Aligner [N_termos, 4096] ou [4096]
    text_embedding: Embedding original do Qwen [4096]
    """
    # Se temos múltiplos termos refinados, comparamos cada um com o alvo
    # e pegamos a média ou o máximo de similaridade.
    if visual_aligned.dim() > 1:
        # Em vez de média simples, calculamos a similaridade de cada termo
        # e depois tiramos a média dos scores.
        # Isso evita "poluir" o embedding do gato com o embedding do sofá.
        visual_aligned = F.normalize(visual_aligned, p=2, dim=-1)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        
        similarities = torch.matmul(visual_aligned, text_embedding.T)
        return similarities.mean() 
        
    return F.cosine_similarity(visual_aligned.unsqueeze(0), text_embedding.unsqueeze(0))