"""
vg_dataset.py
-------------
Visual Genome VG-150 multi-label dataset para treino da SGClassifierHead.

Helpers compartilhados com eval_sg_vg.py (vocab, split deterministico).
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
