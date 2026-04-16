"""
relation_head.py
----------------
Classificador de predicados para pares (sub, obj) no espaco alinhado do Qwen (4096-d).

Extraido de classifier_head.py com renomeacoes:
  ctx_proj  -> union_proj
  pair_ctx  -> union_feat
Semantica: union_feat vem do ROI mean-pool do retangulo uniao de (sub_bbox, obj_bbox).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RelationHead(nn.Module):
    """
    Classificador de predicados para pares (sub, obj) no espaco alinhado (4096-d).

    Shapes
    ------
    sub_feat   : [P, text_dim]   (4096) — embedding projetado do sujeito
    obj_feat   : [P, text_dim]   (4096) — embedding projetado do objeto
    union_feat : [P, text_dim]   (4096) — ROI mean-pool do retangulo uniao
                 (opcional; requerido se use_ctx=True)
    retorno    : [P, vocab_size] (50)   — logits sobre predicados

    Direcionalidade:
      Projecoes separadas sub_proj/obj_proj + feature de diferenca (s - o)
      garantem que (A, B) != (B, A) mesmo sem union_feat.
    """

    def __init__(
        self,
        text_dim: int = 4096,
        vocab_size: int = 50,
        proj_dim: int = 512,
        hidden: int = 1024,
        dropout: float = 0.1,
        use_ctx: bool = True,
    ):
        super().__init__()
        self.use_ctx = use_ctx
        self.sub_proj = nn.Linear(text_dim, proj_dim)
        self.obj_proj = nn.Linear(text_dim, proj_dim)
        if use_ctx:
            self.union_proj = nn.Linear(text_dim, proj_dim)
            in_dim = proj_dim * 4  # [sub, obj, sub-obj, union]
        else:
            in_dim = proj_dim * 3  # [sub, obj, sub-obj]

        self.norm = nn.LayerNorm(in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, vocab_size),
        )

    def forward(
        self,
        sub_feat: torch.Tensor,
        obj_feat: torch.Tensor,
        union_feat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        sub_feat   : [P, text_dim] — embedding projetado do sujeito
        obj_feat   : [P, text_dim] — embedding projetado do objeto
        union_feat : [P, text_dim] — feature do retangulo uniao (requerido se use_ctx=True)

        Returns
        -------
        logits : [P, vocab_size]
        """
        s = self.sub_proj(sub_feat)   # [P, proj_dim]
        o = self.obj_proj(obj_feat)   # [P, proj_dim]
        parts = [s, o, s - o]
        if self.use_ctx:
            if union_feat is None:
                raise ValueError("RelationHead foi criada com use_ctx=True, mas union_feat=None")
            parts.append(self.union_proj(union_feat))  # [P, proj_dim]
        x = torch.cat(parts, dim=-1)  # [P, proj_dim * 3 ou 4]
        x = self.norm(x)
        return self.mlp(x)            # [P, vocab_size]
