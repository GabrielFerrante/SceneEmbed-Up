"""
attribute_classifier.py
-----------------------
Stage 2 do pipeline SGG: classifica atributos de objetos detectados.

Entrada: lista de deteccoes (com bbox_grid) + DINOv3 visual grid.
Saida: lista de deteccoes com campo "attributes" adicionado.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.SG.attribute_head import AttributeHead
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner


class AttributeClassifier:
    """
    Stage 2 do pipeline SGG: classifica atributos de objetos detectados.

    Entrada: lista de Detection (com bbox_grid) + DINOv3 visual grid.
    Saida: lista de Detection com campo "attributes" adicionado.

    Detection e qualquer objeto com atributo `bbox_grid` (tensor [4],
    coordenadas x1, y1, x2, y2 na grade 32x32). O campo `attributes`
    sera adicionado como lista de {"name": str, "score": float}.
    """

    def __init__(
        self,
        attribute_head: AttributeHead,
        aligner: LoRACrossAttentionAligner,
        attr_vocab: list[str],
        score_threshold: float = 0.3,
        top_k: int = 5,
    ):
        self.attribute_head = attribute_head
        self.aligner = aligner
        self.attr_vocab = attr_vocab
        self.score_threshold = score_threshold
        self.top_k = top_k

    @staticmethod
    def _roi_mean_pool(
        grid: torch.Tensor,       # [B, 32, 32, 768]
        bboxes: torch.Tensor,     # [M, 4]
        img_idx: torch.Tensor,    # [M]
    ) -> torch.Tensor:
        """
        Mean-pool de patches dentro de cada bbox.
        Identico a train_relation_head.roi_mean_pool (paridade train/test).

        Shapes:
          grid    : [B, 32, 32, 768]
          bboxes  : [M, 4] (x1, y1, x2, y2) em coords de grade
          img_idx : [M] indice da imagem no batch

        Returns:
          feats : [M, 768]
        """
        M = bboxes.size(0)
        out = grid.new_zeros(M, grid.size(-1))
        # quantiza bbox -> indices de patch
        x1 = bboxes[:, 0].floor().clamp(0, 31).long()
        y1 = bboxes[:, 1].floor().clamp(0, 31).long()
        x2 = bboxes[:, 2].ceil().clamp(1, 32).long()
        y2 = bboxes[:, 3].ceil().clamp(1, 32).long()
        # garante x2 > x1, y2 > y1
        x2 = torch.maximum(x2, x1 + 1)
        y2 = torch.maximum(y2, y1 + 1)
        for i in range(M):
            bi = img_idx[i].item()
            region = grid[bi, y1[i]:y2[i], x1[i]:x2[i]]  # [h, w, 768]
            out[i] = region.reshape(-1, region.size(-1)).mean(dim=0)
        return out

    @torch.no_grad()
    def predict(
        self,
        detections: list,          # list de objetos com .bbox_grid [4]
        visual_grid: torch.Tensor, # [1, 32, 32, 768]
    ) -> list:
        """
        Pipeline de predicao de atributos para objetos detectados.

        Passos:
          1. Empilha bboxes de todos os detections: [M, 4]
          2. roi_mean_pool(grid, bboxes) -> [M, 768]
          3. aligner.visual_proj -> [M, 4096]
          4. attribute_head -> [M, vocab_size] logits
          5. sigmoid, threshold, top-K por objeto
          6. Adiciona campo "attributes" a cada Detection

        Parameters
        ----------
        detections:
            Lista de objetos com atributo `bbox_grid` (tensor [4],
            coordenadas x1, y1, x2, y2 na grade 32x32).
        visual_grid:
            Grid de features DINOv3 [1, 32, 32, 768].

        Returns
        -------
        list
            Mesma lista de detections com .attributes adicionado.
            Cada .attributes e lista de {"name": str, "score": float}.
        """
        if len(detections) == 0:
            return detections

        device = visual_grid.device
        dtype = visual_grid.dtype

        # 1. Empilha bboxes: [M, 4]
        bboxes = torch.stack(
            [d.bbox_grid for d in detections], dim=0
        ).to(device=device, dtype=dtype)                  # [M, 4]
        img_idx = torch.zeros(
            bboxes.size(0), dtype=torch.long, device=device
        )                                                  # [M] (batch=1)

        # 2. ROI mean pool: [M, 768]
        feat_768 = self._roi_mean_pool(visual_grid, bboxes, img_idx)

        # 3. Projecao para espaco texto: [M, 4096]
        feat_4096 = self.aligner.visual_proj(feat_768)

        # 4. Classificacao de atributos: [M, vocab_size]
        logits = self.attribute_head(feat_4096)

        # 5. Sigmoid + threshold + top-K
        probs = torch.sigmoid(logits)                      # [M, vocab_size]

        for i, det in enumerate(detections):
            scores_i = probs[i]                            # [vocab_size]
            # Mascara de threshold
            mask = scores_i >= self.score_threshold
            valid_indices = mask.nonzero(as_tuple=False).squeeze(-1)

            if valid_indices.numel() == 0:
                det.attributes = []
                continue

            valid_scores = scores_i[valid_indices]
            # Top-K entre os validos
            k = min(self.top_k, valid_indices.numel())
            topk_scores, topk_local = valid_scores.topk(k)
            topk_indices = valid_indices[topk_local]

            attrs = []
            for j in range(k):
                attr_idx = topk_indices[j].item()
                attrs.append({
                    "name": self.attr_vocab[attr_idx],
                    "score": round(topk_scores[j].item(), 4),
                })
            det.attributes = attrs

        return detections
