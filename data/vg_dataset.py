"""
vg_dataset.py
-------------
Helpers e datasets Visual Genome (VG-150).

Conteudo:
  - Vocab builders: build_vg150_vocab, build_attribute_vocab
  - Split deterministico: deterministic_split
  - Datasets: VisualGenomeMultiLabelDataset (objetos),
              VisualGenomeRelationDataset (predicados),
              VisualGenomeAttributeDataset (atributos)
  - Collate: relation_collate, attribute_collate
  - Pesos: compute_pos_weight, compute_predicate_class_weight,
           compute_attribute_pos_weight
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


def load_scene_graphs(vg_dir: str) -> list[dict]:
    """Carrega scene_graphs.json do Visual Genome."""
    path = os.path.join(vg_dir, "scene_graphs.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"scene_graphs.json nao encontrado em {vg_dir}. "
            "Rode: python data/download_visual_genome.py --data-dir <vg_dir>"
        )
    print(f"Carregando scene_graphs.json ({os.path.getsize(path) / 1e9:.1f} GB)...")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_vg150_vocab(
    scene_graphs: list[dict],
    n_objects: int = 150,
    n_predicates: int = 50,
) -> tuple[list[str], list[str], set[str], set[str]]:
    """Constroi vocabulario VG-150: top-150 objetos, top-50 predicados."""
    obj_counter: Counter[str] = Counter()
    pred_counter: Counter[str] = Counter()

    for sg in tqdm(scene_graphs, desc="Construindo vocabulario VG-150"):
        for obj in sg.get("objects", []):
            names = obj.get("names") or [obj.get("name", "")]
            name = names[0].lower().strip() if names else ""
            if name:
                obj_counter[name] += 1
        for rel in sg.get("relationships", []):
            pred = rel.get("predicate", "").lower().strip()
            if pred:
                pred_counter[pred] += 1

    obj_list = [n for n, _ in obj_counter.most_common(n_objects)]
    pred_list = [p for p, _ in pred_counter.most_common(n_predicates)]

    print(f"  Objetos: {len(obj_counter)} unicos -> top {n_objects}")
    print(f"  Predicados: {len(pred_counter)} unicos -> top {n_predicates}")

    return obj_list, pred_list, set(obj_list), set(pred_list)


def get_image_path(vg_dir: str, image_id: int) -> str | None:
    """Procura imagem em VG_100K ou VG_100K_2."""
    for subdir in ["VG_100K", "VG_100K_2"]:
        path = os.path.join(vg_dir, subdir, f"{image_id}.jpg")
        if os.path.exists(path):
            return path
    return None


def deterministic_split(
    n: int, test_ratio: float, seed: int
) -> tuple[list[int], list[int]]:
    """Retorna (train_indices, test_indices) com mesmo esquema do eval_sg_vg.py."""
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    test_start = int(n * (1 - test_ratio))
    return indices[:test_start], indices[test_start:]


def extract_object_labels(sg: dict, obj_vocab: set[str]) -> set[str]:
    """Extrai labels de objetos presentes na imagem filtrados pelo vocabulario."""
    labels: set[str] = set()
    for obj in sg.get("objects", []):
        names = obj.get("names") or [obj.get("name", "")]
        name = names[0].lower().strip() if names else ""
        if name in obj_vocab:
            labels.add(name)
    return labels


class VisualGenomeMultiLabelDataset(Dataset):
    """
    Dataset VG-150 para classificacao multi-label de objetos.

    Cada item retorna:
      - image: tensor [3, image_size, image_size], float32 (normalizado pelo transform)
      - target: tensor [vocab_size], float32 multi-hot

    Pre-filtra amostras sem imagem no disco ou sem labels no vocab.
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
        self.vocab_size = len(obj_list)
        obj_set = set(obj_list)

        self.samples: list[tuple[str, set[str]]] = []
        for idx in tqdm(indices, desc="Filtrando amostras VG"):
            sg = scene_graphs[idx]
            labels = extract_object_labels(sg, obj_set)
            if not labels:
                continue
            img_path = get_image_path(vg_dir, sg["image_id"])
            if img_path is None:
                continue
            self.samples.append((img_path, labels))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path, labels = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img)  # [3, H, W]

        target = torch.zeros(self.vocab_size, dtype=torch.float32)
        for label in labels:
            target[self.obj_to_idx[label]] = 1.0

        return img_tensor, target


class VisualGenomeRelationDataset(Dataset):
    """
    Dataset VG-150 para classificacao de predicados por par (sub, obj).

    Cada item retorna:
      image     : [3, H, W] float32
      sub_bboxes: [M, 4] float32, coords em grid 32x32 (x1, y1, x2, y2)
      obj_bboxes: [M, 4] float32
      pred_idxs : [M] long, indices em pred_list

    M varia por imagem. Use `relation_collate` para empilhar com ponteiros.
    Bboxes originais (x, y, w, h) sao mapeadas para a grade 32x32 via
    escala (32/width, 32/height) para alinhar com visual_feats [B, 1024, 768].
    """

    GRID = 32

    def __init__(
        self,
        vg_dir: str,
        scene_graphs: list[dict],
        obj_list: list[str],
        pred_list: list[str],
        indices: list[int],
        transform: Callable,
    ):
        self.vg_dir = vg_dir
        self.transform = transform
        self.obj_set = set(obj_list)
        self.pred_to_idx = {p: i for i, p in enumerate(pred_list)}

        self.samples: list[dict] = []
        for idx in tqdm(indices, desc="Filtrando pares VG"):
            sg = scene_graphs[idx]

            id_to_obj: dict[int, dict] = {}
            for obj in sg.get("objects", []):
                names = obj.get("names") or [obj.get("name", "")]
                name = names[0].lower().strip() if names else ""
                if name not in self.obj_set:
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
                id_to_obj[obj["object_id"]] = {"bbox": (x, y, w, h)}

            relations: list[dict] = []
            for rel in sg.get("relationships", []):
                sub_id = rel.get("subject_id")
                obj_id = rel.get("object_id")
                pred = rel.get("predicate", "").lower().strip()
                if (
                    sub_id in id_to_obj
                    and obj_id in id_to_obj
                    and sub_id != obj_id
                    and pred in self.pred_to_idx
                ):
                    relations.append({
                        "sub_bbox": id_to_obj[sub_id]["bbox"],
                        "obj_bbox": id_to_obj[obj_id]["bbox"],
                        "pred_idx": self.pred_to_idx[pred],
                    })

            if not relations:
                continue
            img_path = get_image_path(vg_dir, sg["image_id"])
            if img_path is None:
                continue

            self.samples.append({
                "img_path": img_path,
                "img_w": sg.get("width"),
                "img_h": sg.get("height"),
                "relations": relations,
            })

    def __len__(self) -> int:
        return len(self.samples)

    def _map_bbox(self, bbox: tuple, W: float, H: float) -> list[float]:
        # (x, y, w, h) em pixels originais -> (x1, y1, x2, y2) em grid 32x32
        x, y, w, h = bbox
        sx = self.GRID / max(W, 1.0)
        sy = self.GRID / max(H, 1.0)
        x1 = max(0.0, min(float(self.GRID), x * sx))
        y1 = max(0.0, min(float(self.GRID), y * sy))
        x2 = max(0.0, min(float(self.GRID), (x + w) * sx))
        y2 = max(0.0, min(float(self.GRID), (y + h) * sy))
        # garante area >= 1 patch
        if x2 - x1 < 1.0:
            x2 = min(float(self.GRID), x1 + 1.0)
        if y2 - y1 < 1.0:
            y2 = min(float(self.GRID), y1 + 1.0)
        return [x1, y1, x2, y2]

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        orig_w, orig_h = img.size
        W = float(s["img_w"] or orig_w)
        H = float(s["img_h"] or orig_h)
        img_t = self.transform(img)

        sub, obj, preds = [], [], []
        for r in s["relations"]:
            sub.append(self._map_bbox(r["sub_bbox"], W, H))
            obj.append(self._map_bbox(r["obj_bbox"], W, H))
            preds.append(r["pred_idx"])

        return (
            img_t,
            torch.tensor(sub, dtype=torch.float32),
            torch.tensor(obj, dtype=torch.float32),
            torch.tensor(preds, dtype=torch.long),
        )


def relation_collate(batch):
    """Empilha imagens e concatena pares com ponteiro img_idx."""
    images = torch.stack([b[0] for b in batch])
    sub_parts, obj_parts, pred_parts, idx_parts = [], [], [], []
    for i, (_, sb, ob, pr) in enumerate(batch):
        if sb.numel() == 0:
            continue
        sub_parts.append(sb)
        obj_parts.append(ob)
        pred_parts.append(pr)
        idx_parts.append(torch.full((pr.size(0),), i, dtype=torch.long))
    sub_bboxes = torch.cat(sub_parts, dim=0) if sub_parts else torch.zeros(0, 4)
    obj_bboxes = torch.cat(obj_parts, dim=0) if obj_parts else torch.zeros(0, 4)
    pred_idxs = torch.cat(pred_parts, dim=0) if pred_parts else torch.zeros(0, dtype=torch.long)
    img_indices = torch.cat(idx_parts, dim=0) if idx_parts else torch.zeros(0, dtype=torch.long)
    return images, sub_bboxes, obj_bboxes, pred_idxs, img_indices


def compute_predicate_class_weight(
    dataset: "VisualGenomeRelationDataset",
    clip_max: float = 50.0,
) -> torch.Tensor:
    """class_weight para CrossEntropyLoss: (N/n_classes)/count por classe."""
    n_classes = len(dataset.pred_to_idx)
    counts = torch.zeros(n_classes, dtype=torch.float64)
    total = 0
    for s in dataset.samples:
        for r in s["relations"]:
            counts[r["pred_idx"]] += 1.0
            total += 1
    counts = counts.clamp(min=1.0)
    weight = (total / n_classes) / counts
    return weight.clamp(max=clip_max).to(torch.float32)


def compute_pos_weight(
    dataset: VisualGenomeMultiLabelDataset,
    clip_max: float = 50.0,
) -> torch.Tensor:
    """
    pos_weight para BCEWithLogitsLoss: (N - pos) / pos por classe.

    Clipa em clip_max para evitar gradientes explosivos em classes muito raras.
    """
    vocab_size = dataset.vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    for _, labels in dataset.samples:
        for label in labels:
            counts[dataset.obj_to_idx[label]] += 1.0
    n = float(len(dataset))
    pos_weight = torch.where(
        counts > 0, (n - counts) / counts, torch.tensor(clip_max, dtype=torch.float64)
    )
    return pos_weight.clamp(max=clip_max).to(torch.float32)


# ── Attributes ─────────────────────────────────────────────────────────


def build_attribute_vocab(
    scene_graphs: list[dict],
    n_attrs: int = 200,
) -> tuple[list[str], set[str]]:
    """
    Constroi vocabulario dos top-N atributos do VG por frequencia.

    Atributos vem de objects[i]["attributes"] (lista de strings).
    Retorna (attr_list, attr_set).
    """
    counter: Counter[str] = Counter()
    for sg in tqdm(scene_graphs, desc="Construindo vocabulario de atributos"):
        for obj in sg.get("objects", []):
            for attr in obj.get("attributes", []):
                a = attr.lower().strip()
                if a:
                    counter[a] += 1

    attr_list = [a for a, _ in counter.most_common(n_attrs)]
    print(f"  Atributos: {len(counter)} unicos -> top {n_attrs}")
    return attr_list, set(attr_list)


class VisualGenomeAttributeDataset(Dataset):
    """
    Dataset VG para classificacao multi-label de atributos por objeto.

    Cada amostra e um **objeto individual** (bbox GT) com seus atributos.
    Retorna:
      image     : [3, 256, 256] float32
      bbox      : [4] float32, (x1, y1, x2, y2) em grid 32x32
      attrs     : [vocab_size] float32, multi-hot de atributos
      img_idx   : int (indice da imagem, usado em collate para batching)

    Objetos sem bbox valida ou sem atributos no vocab sao descartados.
    """

    GRID = 32

    def __init__(
        self,
        vg_dir: str,
        scene_graphs: list[dict],
        obj_list: list[str],
        attr_list: list[str],
        indices: list[int],
        transform: Callable,
    ):
        self.vg_dir = vg_dir
        self.transform = transform
        self.attr_list = attr_list
        self.attr_to_idx = {a: i for i, a in enumerate(attr_list)}
        self.vocab_size = len(attr_list)
        obj_set = set(obj_list)
        attr_set = set(attr_list)

        self.samples: list[dict] = []
        for idx in tqdm(indices, desc="Filtrando atributos VG"):
            sg = scene_graphs[idx]
            img_path = get_image_path(vg_dir, sg["image_id"])
            if img_path is None:
                continue

            img_w = float(sg.get("width") or 1)
            img_h = float(sg.get("height") or 1)

            for obj in sg.get("objects", []):
                names = obj.get("names") or [obj.get("name", "")]
                name = names[0].lower().strip() if names else ""
                if name not in obj_set:
                    continue

                raw_attrs = [
                    a.lower().strip()
                    for a in obj.get("attributes", [])
                    if a.lower().strip() in attr_set
                ]
                if not raw_attrs:
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

                self.samples.append({
                    "img_path": img_path,
                    "img_w": img_w,
                    "img_h": img_h,
                    "bbox": (x, y, w, h),
                    "attrs": raw_attrs,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def _map_bbox(self, bbox: tuple, W: float, H: float) -> list[float]:
        # (x, y, w, h) pixels -> (x1, y1, x2, y2) em grid 32x32
        x, y, w, h = bbox
        sx = self.GRID / max(W, 1.0)
        sy = self.GRID / max(H, 1.0)
        x1 = max(0.0, min(float(self.GRID), x * sx))
        y1 = max(0.0, min(float(self.GRID), y * sy))
        x2 = max(0.0, min(float(self.GRID), (x + w) * sx))
        y2 = max(0.0, min(float(self.GRID), (y + h) * sy))
        if x2 - x1 < 1.0:
            x2 = min(float(self.GRID), x1 + 1.0)
        if y2 - y1 < 1.0:
            y2 = min(float(self.GRID), y1 + 1.0)
        return [x1, y1, x2, y2]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        img = Image.open(s["img_path"]).convert("RGB")
        img_tensor = self.transform(img)

        bbox = torch.tensor(
            self._map_bbox(s["bbox"], s["img_w"], s["img_h"]),
            dtype=torch.float32,
        )

        attrs = torch.zeros(self.vocab_size, dtype=torch.float32)
        for a in s["attrs"]:
            attrs[self.attr_to_idx[a]] = 1.0

        return img_tensor, bbox, attrs


def attribute_collate(batch):
    """
    Collate para VisualGenomeAttributeDataset.

    Como cada amostra e 1 objeto, agrupa imagens com ponteiro img_idx.
    Imagens duplicadas (varios objetos da mesma imagem) sao empilhadas
    repetidamente — eficiente via cache L2, mas para datasets muito grandes
    considerar agrupar na sampling.

    Retorna:
      images  : [B, 3, 256, 256]
      bboxes  : [B, 4]
      attrs   : [B, vocab_size]
      img_idx : [B] (arange simples — 1 objeto por slot de batch)
    """
    images = torch.stack([b[0] for b in batch])
    bboxes = torch.stack([b[1] for b in batch])
    attrs = torch.stack([b[2] for b in batch])
    img_idx = torch.arange(len(batch), dtype=torch.long)
    return images, bboxes, attrs, img_idx


def compute_attribute_pos_weight(
    dataset: VisualGenomeAttributeDataset,
    clip_max: float = 50.0,
) -> torch.Tensor:
    """pos_weight para BCEWithLogitsLoss sobre atributos: (N - pos) / pos."""
    vocab_size = dataset.vocab_size
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    for s in dataset.samples:
        for a in s["attrs"]:
            counts[dataset.attr_to_idx[a]] += 1.0
    n = float(len(dataset))
    pos_weight = torch.where(
        counts > 0, (n - counts) / counts, torch.tensor(clip_max, dtype=torch.float64)
    )
    return pos_weight.clamp(max=clip_max).to(torch.float32)
