import torch
import torch.nn as nn


class SGClassifierHead(nn.Module):
    def __init__(self, visual_dim: int = 768, vocab_size: int = 150, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(visual_dim)
        self.mlp = nn.Sequential(
            nn.Linear(visual_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, visual_feats: torch.Tensor) -> torch.Tensor:
        # visual_feats: [B, N, visual_dim] (N=1024 para grade HR 32x32)
        # returns patch_logits: [B, N, vocab_size] (sem agregacao; MIL/CC no caller)
        x = self.norm(visual_feats)
        return self.mlp(x)
