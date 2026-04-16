"""
vg_detection_dataset.py
-----------------------
Dataset Visual Genome VG-150 para fine-tune de detectores (DETR-R50).

Formato de saida compativel com `transformers.DetrForObjectDetection`:
  image  : [3, 640, 640] float32 (normalizado pelo transform ImageNet)
  target : dict com:
    - "class_labels" : [K] long, indices em obj_list (0..149)
    - "boxes"        : [K, 4] float32, formato (cx, cy, w, h) normalizado [0, 1]

Objetos cujo label nao esta em VG-150 sao descartados. Imagens sem nenhum
objeto valido sao filtradas antes de virar amostra.

Imagem e redimensionada para 256x256 (mesma resolucao de COYO/DINO); bboxes
normalizadas sao invariantes ao resize.
"""

from __future__ import annotations

from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from data.vg_dataset import get_image_path


def _xywh_to_cxcywh_normalized(
    x: float, y: float, w: float, h: float,
    img_w: float, img_h: float,
) -> tuple[float, float, float, float]:
    """
    Converte bbox VG (x, y, w, h) em pixels -> (cx, cy, w, h) normalizado [0, 1].

    Shapes: escalar-a-escalar.
    """
    cx = (x + w / 2.0) / max(img_w, 1.0)
    cy = (y + h / 2.0) / max(img_h, 1.0)
    wn = w / max(img_w, 1.0)
    hn = h / max(img_h, 1.0)
    # clamp para dentro da imagem (proteger contra anotacoes defeituosas)
    cx = min(max(cx, 0.0), 1.0)
    cy = min(max(cy, 0.0), 1.0)
    wn = min(max(wn, 1e-4), 1.0)
    hn = min(max(hn, 1e-4), 1.0)
    return cx, cy, wn, hn


class VisualGenomeDetectionDataset(Dataset):
    """
    Dataset VG-150 para fine-tune de detector (formato DETR).

    Cada amostra:
      image  : [3, 640, 640] float32 (ImageNet-normalized)
      target : {"class_labels": [K] long, "boxes": [K, 4] float32 (cxcywh norm)}
    """

    def __init__(
        self,
        vg_dir: str,
        scene_graphs: list[dict],
        obj_list: list[str],
        indices: list[int],
        transform: Callable,
    ):
        self.vg_dir = vg_dir
        self.transform = transform
        self.obj_list = obj_list
        self.obj_to_idx = {obj: i for i, obj in enumerate(obj_list)}
        obj_set = set(obj_list)

        self.samples: list[dict] = []
        for idx in tqdm(indices, desc="Filtrando amostras de deteccao VG"):
            sg = scene_graphs[idx]

            objects: list[dict] = []
            for obj in sg.get("objects", []):
                names = obj.get("names") or [obj.get("name", "")]
                name = names[0].lower().strip() if names else ""
                if name not in obj_set:
                    continue
                try:
                    x = float(obj["x"])
                    y = float(obj["y"])
                    w = float(obj["w"])
                    h = float(obj["h"])
                except (KeyError, TypeError, ValueError):
                    continue
                if w <= 0 or h <= 0:
                    continue
                objects.append({"label": name, "bbox": (x, y, w, h)})

            if not objects:
                continue

            img_path = get_image_path(vg_dir, sg["image_id"])
            if img_path is None:
                continue

            self.samples.append({
                "img_path": img_path,
                "img_w": sg.get("width"),
                "img_h": sg.get("height"),
                "objects": objects,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        W = float(s["img_w"] or orig_w)
        H = float(s["img_h"] or orig_h)

        img_tensor = self.transform(img)  # [3, 256, 256]

        class_labels: list[int] = []
        boxes: list[list[float]] = []
        for o in s["objects"]:
            cx, cy, wn, hn = _xywh_to_cxcywh_normalized(*o["bbox"], W, H)
            class_labels.append(self.obj_to_idx[o["label"]])
            boxes.append([cx, cy, wn, hn])

        target = {
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
            "boxes": torch.tensor(boxes, dtype=torch.float32),
        }
        return img_tensor, target


def detection_collate(batch):
    """
    Collate para DETR: imagens em tensor [B, 3, 640, 640], targets como list[dict].
    DetrForObjectDetection recebe `labels=List[Dict]`, uma entrada por imagem.
    """
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def compute_object_class_weight(
    dataset: VisualGenomeDetectionDataset,
    clip_max: float = 50.0,
) -> torch.Tensor:
    """
    class_weight para o classificador do DETR: (N/n_classes)/count por classe.

    Retorna tensor [n_classes + 1] incluindo o slot "no object" (ultimo indice)
    com peso fixo 0.1 (convencao DETR).
    """
    n_classes = len(dataset.obj_to_idx)
    counts = torch.zeros(n_classes, dtype=torch.float64)
    total = 0
    for s in dataset.samples:
        for o in s["objects"]:
            counts[dataset.obj_to_idx[o["label"]]] += 1.0
            total += 1
    counts = counts.clamp(min=1.0)
    weight = (total / n_classes) / counts
    weight = weight.clamp(max=clip_max).to(torch.float32)

    # Slot "no object" com peso reduzido (padrao DETR: eos_coef=0.1)
    full = torch.cat([weight, torch.tensor([0.1], dtype=torch.float32)])
    return full
