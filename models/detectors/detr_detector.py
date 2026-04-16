"""
detr_detector.py
----------------
Wrapper de inferencia sobre ``facebook/detr-resnet-50`` fine-tuned para VG-150.

Carrega o DETR-R50 com head re-treinada para 150 classes de objetos
(+ slot "no object" gerenciado internamente pelo DETR).
Imagens entram em 256x256 (mesma resolucao do pipeline DINO/COYO).

Uso tipico:
    detector = DetrDetector(
        checkpoint_path="checkpoints/best_detr_vg150.pth",
        vg150_labels=obj_list,
        score_threshold=0.5,
    )
    detections = detector.detect(pil_image)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import DetrForObjectDetection, DetrImageProcessor


@dataclass
class Detection:
    """Resultado de uma deteccao individual.

    Attributes
    ----------
    bbox_xyxy : torch.Tensor
        [4] float, coordenadas (x1, y1, x2, y2) em pixels da imagem resized (640x640).
    bbox_grid : torch.Tensor
        [4] float, coordenadas (x1, y1, x2, y2) em grade 0..``image_size_grid``
        (default 32, alinhado com visual_feats [B, 1024, 768] = 32x32 patches).
    label : str
        Nome do objeto no vocabulario VG-150.
    label_idx : int
        Indice do objeto em ``vg150_labels``.
    score : float
        Confianca da deteccao (softmax prob da classe vencedora).
    """

    bbox_xyxy: torch.Tensor  # [4] float, coords em pixels 256x256
    bbox_grid: torch.Tensor  # [4] float, coords em grade 0..32
    label: str
    label_idx: int
    score: float

    def to_node_dict(self) -> dict:
        """Serializa para formato ``nodes[]`` do scene graph.

        Returns
        -------
        dict
            ``{"label": str, "bbox": Tensor[4], "score": float}``
        """
        return {
            "label": self.label,
            "bbox": self.bbox_grid.clone(),
            "score": self.score,
        }


class DetrDetector:
    """
    Detector VG-150 baseado em DETR-R50.

    Carrega ``facebook/detr-resnet-50`` com head re-treinada para 150+1 classes.
    Imagens entram em 256x256 (mesma resolucao do DINO/COYO).

    Parameters
    ----------
    checkpoint_path : str
        Caminho para o ``state_dict`` do modelo fine-tuned (salvo via
        ``torch.save(model.state_dict(), path)``).
    vg150_labels : list[str]
        Lista ordenada de 150 nomes de objetos (mesma ordem usada no treino).
    score_threshold : float
        Confianca minima para manter uma deteccao. Default 0.5.
    device : str
        ``"cuda"`` ou ``"cpu"``.
    """

    def __init__(
        self,
        checkpoint_path: str,
        vg150_labels: list[str],
        score_threshold: float = 0.5,
        device: str = "cuda",
    ) -> None:
        self.device = device
        self.score_threshold = score_threshold
        self.labels = vg150_labels
        self.num_labels = len(vg150_labels)  # 150

        # Carrega arquitetura DETR-R50 com head para 150 classes.
        # ignore_mismatched_sizes=True porque a head original tem 91 classes COCO.
        self.model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )

        # Carrega pesos do fine-tune sobre VG-150.
        state = torch.load(checkpoint_path, map_location=device)
        self.model.load_state_dict(state, strict=True)

        self.model.to(device).eval()

        # Processor: nao faz resize (nos ja resizamos para 256x256 externamente)
        # e nao faz normalize (o get_transforms ja aplica ImageNet normalization).
        # O processor apenas converte para o formato de tensores esperado pelo DETR.
        self.processor = DetrImageProcessor(
            do_resize=False,
            do_normalize=False,
        )

    @torch.no_grad()
    def detect(
        self,
        image: Image.Image,
        image_size: int = 640,
        image_size_grid: int = 32,
    ) -> list[Detection]:
        """
        Detecta objetos em imagem PIL ja em 640x640 (pre-transformada).

        Pipeline:
          1. processor(image) -> pixel_values [1, 3, H, W]
          2. model(pixel_values) -> logits [1, Q, num_labels+1],
             pred_boxes [1, Q, 4] (cxcywh normalizado [0,1])
          3. softmax(logits, dim=-1) -> probs [1, Q, num_labels+1]
             Exclui ultima coluna ("no object"): probs[..., :-1] -> [1, Q, num_labels]
             max_score, max_class = probs.max(dim=-1)
          4. Filtra max_score > score_threshold
          5. Converte cxcywh normalizado -> xyxy em pixels 256x256:
             x1 = (cx - w/2) * image_size
             y1 = (cy - h/2) * image_size
             x2 = (cx + w/2) * image_size
             y2 = (cy + h/2) * image_size
          6. Converte xyxy pixels -> xyxy grid:
             bbox_grid = bbox_xyxy * (image_size_grid / image_size)
          7. Monta lista de Detection

        NMS nao e necessario — DETR produz outputs unicos via attention
        (sem proposals duplicadas).

        Parameters
        ----------
        image : PIL.Image.Image
            Imagem RGB, ja redimensionada para 256x256.
        image_size : int
            Tamanho da imagem em pixels (default 256).
        image_size_grid : int
            Tamanho da grade de destino (default 32, corresponde a 256/8).

        Returns
        -------
        list[Detection]
            Lista de deteccoes filtradas por score_threshold, ordenadas por
            score decrescente.
        """
        # 1. Processor: converte PIL -> pixel_values [1, 3, H, W]
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)  # [1, 3, H, W]

        # 2. Forward: logits [1, Q, num_labels+1], pred_boxes [1, Q, 4] (cxcywh norm)
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # [1, Q, num_labels+1]
        pred_boxes = outputs.pred_boxes  # [1, Q, 4] (cx, cy, w, h) normalizado

        # 3. Probabilidades por classe (exclui "no object" = ultima coluna)
        probs = F.softmax(logits, dim=-1)  # [1, Q, num_labels+1]
        scores, class_ids = probs[0, :, :-1].max(dim=-1)  # [Q], [Q]

        # 4. Filtro por threshold
        keep = scores > self.score_threshold  # [Q] bool
        scores = scores[keep]  # [K]
        class_ids = class_ids[keep]  # [K]
        boxes_cxcywh = pred_boxes[0][keep]  # [K, 4] (cx, cy, w, h) normalizado

        # 5. cxcywh normalizado -> xyxy em pixels
        cx, cy, w, h = boxes_cxcywh.unbind(dim=-1)  # cada [K]
        x1 = (cx - w / 2.0) * image_size  # [K]
        y1 = (cy - h / 2.0) * image_size  # [K]
        x2 = (cx + w / 2.0) * image_size  # [K]
        y2 = (cy + h / 2.0) * image_size  # [K]
        bbox_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)  # [K, 4]

        # Clamp para dentro da imagem
        bbox_xyxy = bbox_xyxy.clamp(min=0.0, max=float(image_size))

        # 6. xyxy pixels -> xyxy grade
        scale = image_size_grid / image_size
        bbox_grid = bbox_xyxy * scale  # [K, 4]

        # 7. Montar lista de Detection (ordenada por score desc)
        detections: list[Detection] = []
        order = scores.argsort(descending=True)
        for i in order:
            idx = class_ids[i].item()
            detections.append(
                Detection(
                    bbox_xyxy=bbox_xyxy[i].cpu(),
                    bbox_grid=bbox_grid[i].cpu(),
                    label=self.labels[idx],
                    label_idx=idx,
                    score=scores[i].item(),
                )
            )

        return detections

    @torch.no_grad()
    def detect_batch(
        self,
        images: list[Image.Image],
        image_size: int = 640,
        image_size_grid: int = 32,
    ) -> list[list[Detection]]:
        """
        Detecta objetos em um batch de imagens PIL (todas 640x640).

        Parameters
        ----------
        images : list[PIL.Image.Image]
            Batch de imagens RGB, ja redimensionadas para 640x640x.
        image_size : int
            Tamanho da imagem em pixels (default 256).
        image_size_grid : int
            Tamanho da grade de destino (default 32).

        Returns
        -------
        list[list[Detection]]
            Uma lista de deteccoes por imagem do batch.

        Shapes intermediarios:
          pixel_values : [B, 3, H, W]
          logits       : [B, Q, num_labels+1]
          pred_boxes   : [B, Q, 4] (cxcywh normalizado)
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)  # [B, 3, H, W]

        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # [B, Q, num_labels+1]
        pred_boxes = outputs.pred_boxes  # [B, Q, 4]

        probs = F.softmax(logits, dim=-1)  # [B, Q, num_labels+1]

        batch_detections: list[list[Detection]] = []
        B = pixel_values.size(0)

        for b in range(B):
            scores, class_ids = probs[b, :, :-1].max(dim=-1)  # [Q], [Q]
            keep = scores > self.score_threshold
            scores_k = scores[keep]
            class_ids_k = class_ids[keep]
            boxes_k = pred_boxes[b][keep]  # [K, 4]

            cx, cy, w, h = boxes_k.unbind(dim=-1)
            x1 = (cx - w / 2.0) * image_size
            y1 = (cy - h / 2.0) * image_size
            x2 = (cx + w / 2.0) * image_size
            y2 = (cy + h / 2.0) * image_size
            bbox_xyxy = torch.stack([x1, y1, x2, y2], dim=-1).clamp(
                min=0.0, max=float(image_size)
            )
            bbox_grid = bbox_xyxy * (image_size_grid / image_size)

            dets: list[Detection] = []
            order = scores_k.argsort(descending=True)
            for i in order:
                idx = class_ids_k[i].item()
                dets.append(
                    Detection(
                        bbox_xyxy=bbox_xyxy[i].cpu(),
                        bbox_grid=bbox_grid[i].cpu(),
                        label=self.labels[idx],
                        label_idx=idx,
                        score=scores_k[i].item(),
                    )
                )
            batch_detections.append(dets)

        return batch_detections
