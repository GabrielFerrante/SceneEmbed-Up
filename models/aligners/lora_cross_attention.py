from __future__ import annotations

from typing import Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRACrossAttentionAligner(nn.Module):
    """
    Aligner baseado em Cross-Attention com LoRA nos patches visuais.

    Shapes
    ------
    hr_patches:
        `[B, N_patches, visual_dim]` — sequência de patches visuais já achatados
        (ex.: `visual_dim=768`, `N_patches=32*32=1024`).
    text_queries:
        `[B, N_queries, text_dim]` — embeddings textuais do Qwen
        (tipicamente `text_dim=4096`, `N_queries` igual ao número de termos/textos).

    Retorno
    -------
    attn_output:
        `[B, N_queries, text_dim]` — embeddings textuais refinidos pelo contexto visual.
    attn_weights:
        `[B, num_heads, N_queries, N_patches]` — pesos de atenção sobre os patches visuais.
    """

    def __init__(self, visual_dim: int, text_dim: int, rank: int = 16, num_heads: int = 8) -> None:
        super().__init__()

        # 1. Projeção base da imagem (congelada)
        self.visual_proj = nn.Linear(visual_dim, text_dim)
        self.visual_proj.weight.requires_grad = False
        if self.visual_proj.bias is not None:
            self.visual_proj.bias.requires_grad = False

        # 2. Camada de Cross-Attention
        # Observação: algumas versões do PyTorch não suportam o argumento
        # `average_attn_weights`. Mantemos a assinatura mínima compatível.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        # 3. LoRA (A e B)
        self.lora_A_v = nn.Parameter(torch.empty(visual_dim, rank))
        # Usa um escalar float padrão para o parâmetro `a` (negative_slope)
        nn.init.kaiming_uniform_(self.lora_A_v, a=math.sqrt(5.0))
        self.lora_B_v = nn.Parameter(torch.zeros(rank, text_dim))

        self.scaling = 32.0 / rank  # alpha = 32

    def forward(
        self,
        hr_patches: torch.Tensor,
        text_queries: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executa projeção visual + LoRA e cross-attention texto‑imagem.

        Parameters
        ----------
        hr_patches:
            Tensor `[B, N_patches, visual_dim]`.
        text_queries:
            Tensor `[B, N_queries, text_dim]`.
        """
        # A. Projeção visual com LoRA
        base_v = self.visual_proj(hr_patches)
        lora_v = (hr_patches @ self.lora_A_v) @ self.lora_B_v
        v_features = base_v + (lora_v * self.scaling)

        # B. Cross-Attention
        attn_output, attn_weights = self.cross_attn(
            query=text_queries,
            key=v_features,
            value=v_features,
        )

        return attn_output, attn_weights, v_features


def calculate_retrieval_score(visual_aligned: torch.Tensor, text_embedding: torch.Tensor) -> torch.Tensor:
    """
    Calcula um score de similaridade entre embeddings visuais alinhados e texto.

    Parameters
    ----------
    visual_aligned:
        `[N_terms, text_dim]` ou `[text_dim]` — saídas do aligner.
    text_embedding:
        `[text_dim]` — embedding textual do Qwen.

    Returns
    -------
    torch.Tensor
        Escalar com a similaridade média (multi‑termos) ou cos‑sim (vetor único).
    """
    if visual_aligned.dim() > 1:
        visual_aligned = F.normalize(visual_aligned, p=2, dim=-1)
        text_embedding = F.normalize(text_embedding, p=2, dim=-1)
        similarities = torch.matmul(visual_aligned, text_embedding.T)
        return similarities.mean()

    return F.cosine_similarity(visual_aligned.unsqueeze(0), text_embedding.unsqueeze(0))

