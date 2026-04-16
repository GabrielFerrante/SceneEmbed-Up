import torch
import torch.nn as nn


class AttributeHead(nn.Module):
    """
    Classificador multi-label de atributos sobre features no espaco aligned (4096-d).

    Input:  obj_feat [M, feat_dim]  (ja projetadas via aligner.visual_proj)
    Output: logits   [M, vocab_size] (BCE; multi-label)
    """

    def __init__(
        self,
        feat_dim: int = 4096,
        vocab_size: int = 200,
        hidden: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(feat_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(self, obj_feat: torch.Tensor) -> torch.Tensor:
        # obj_feat: [M, feat_dim]
        # returns: [M, vocab_size]
        return self.mlp(self.norm(obj_feat))
