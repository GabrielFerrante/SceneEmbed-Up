"""
mdetr_detector.py
-----------------
Wrapper MDETR (ashkamath/mdetr-resnet-50) com a mesma interface do DetrDetector.

MDETR e um detector open-vocabulary condicionado em texto, treinado no Visual
Genome entre outros datasets. Para deteccao VG-150 o texto de consulta contem
todos os 150 nomes de classes separados por " . ":
  "person . sky . building . tree . ..."

O modelo retorna pred_logits [B, Q, seq_len] com scores de alinhamento entre
cada query box e cada token do texto. Mapeamos o token de maior score de volta
para o indice de classe via offset_mapping do tokenizador.

Nao requer checkpoint local — carrega direto do HuggingFace Hub.
"""
from __future__ import annotations

import torch
from PIL import Image
from dataclasses import dataclass

import torch
from transformers import MdetrForObjectDetection, MdetrImageProcessor


@dataclass
class Detection:
    """Resultado de uma deteccao individual.

    Attributes
    ----------
    bbox_xyxy : [4] float, coordenadas (x1,y1,x2,y2) em pixels.
    bbox_grid : [4] float, coordenadas (x1,y1,x2,y2) em grade 0..image_size_grid.
    label     : str, nome do objeto no vocabulario VG-150.
    label_idx : int, indice em vg150_labels.
    score     : float, confianca da deteccao.
    """

    bbox_xyxy: torch.Tensor
    bbox_grid: torch.Tensor
    label: str
    label_idx: int
    score: float

    def to_node_dict(self) -> dict:
        return {"label": self.label, "bbox": self.bbox_grid.clone(), "score": self.score}

CHECKPOINT = "ashkamath/mdetr-resnet-50"


class MdetrDetector:
    """
    Detector VG-150 baseado em MDETR pre-treinado no Visual Genome.

    Interface identica ao DetrDetector: detect() e detect_batch()
    retornam list[Detection] com bbox_xyxy e bbox_grid.

    Parameters
    ----------
    vg150_labels : list[str]
        Lista ordenada de 150 nomes de objetos VG-150.
    score_threshold : float
        Confianca minima. Default 0.5.
    device : str
        "cuda" ou "cpu".
    """

    def __init__(
        self,
        vg150_labels: list[str],
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.labels = vg150_labels
        self.score_threshold = score_threshold
        self.device = device

        # Texto de consulta: todos os labels separados por " . "
        self.text = " . ".join(vg150_labels)

        self.processor = MdetrImageProcessor.from_pretrained(CHECKPOINT)
        self.model = MdetrForObjectDetection.from_pretrained(CHECKPOINT)
        self.model.to(device).eval()

        # Pre-computa mapeamento token_idx -> class_idx
        self._token_to_cls = self._build_token_map()

    def _build_token_map(self) -> dict[int, int]:
        """
        Tokeniza self.text e mapeia cada posicao de token ao indice de classe.

        Usa offset_mapping para associar token (char_start, char_end) ao
        label cujo nome ocupa aquela posicao no texto.
        """
        enc = self.processor.tokenizer(
            self.text,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
        )
        offsets = enc["offset_mapping"][0]  # [seq_len, 2]

        # Mapa char_start -> class_idx
        char_to_cls: dict[int, int] = {}
        pos = 0
        for cls_idx, label in enumerate(self.labels):
            for c in range(pos, pos + len(label)):
                char_to_cls[c] = cls_idx
            pos += len(label) + 3  # " . " = 3 chars

        token_to_cls: dict[int, int] = {}
        for tok_i, (cs, ce) in enumerate(offsets.tolist()):
            cs, ce = int(cs), int(ce)
            if cs < ce and cs in char_to_cls:
                token_to_cls[tok_i] = char_to_cls[cs]

        return token_to_cls

    def _postprocess(
        self,
        pred_logits: torch.Tensor,  # [Q, seq_len]
        pred_boxes: torch.Tensor,   # [Q, 4] cxcywh norm
        image_size: int,
        image_size_grid: int,
    ) -> list[Detection]:
        """
        Converte saidas brutas do MDETR em list[Detection].

        Score de cada query = max(sigmoid(logits)) sobre tokens do texto.
        Label = token com maior score -> mapeado para class_idx.
        Boxes convertidas de cxcywh normalizado para xyxy em pixels.
        """
        scores_per_token = torch.sigmoid(pred_logits)          # [Q, seq_len]
        scores, best_tokens = scores_per_token.max(dim=-1)     # [Q]

        keep = scores > self.score_threshold
        scores = scores[keep].cpu()
        best_tokens = best_tokens[keep].cpu()
        boxes_k = pred_boxes[keep].cpu()                        # [K, 4]

        cx, cy, w, h = boxes_k.unbind(-1)
        x1 = (cx - w / 2) * image_size
        y1 = (cy - h / 2) * image_size
        x2 = (cx + w / 2) * image_size
        y2 = (cy + h / 2) * image_size
        bbox_xyxy_all = torch.stack([x1, y1, x2, y2], dim=-1).clamp(0, image_size)
        scale = image_size_grid / image_size

        detections: list[Detection] = []
        order = scores.argsort(descending=True)
        for i in order:
            cls_idx = self._token_to_cls.get(int(best_tokens[i].item()))
            if cls_idx is None:
                continue
            bxyxy = bbox_xyxy_all[i]
            detections.append(Detection(
                bbox_xyxy=bxyxy,
                bbox_grid=bxyxy * scale,
                label=self.labels[cls_idx],
                label_idx=cls_idx,
                score=float(scores[i].item()),
            ))
        return detections

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        image_size: int = 640,
        image_size_grid: int = 32,
    ) -> list[Detection]:
        inputs = self.processor(
            images=image, text=self.text, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        return self._postprocess(
            outputs.pred_logits[0],
            outputs.pred_boxes[0],
            image_size,
            image_size_grid,
        )

    @torch.no_grad()
    def detect_batch(
        self,
        images: list[Image.Image],
        image_size: int = 640,
        image_size_grid: int = 32,
    ) -> list[list[Detection]]:
        inputs = self.processor(
            images=images,
            text=[self.text] * len(images),
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)

        return [
            self._postprocess(
                outputs.pred_logits[b],
                outputs.pred_boxes[b],
                image_size,
                image_size_grid,
            )
            for b in range(len(images))
        ]
