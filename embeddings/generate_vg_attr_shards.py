"""
generate_vg_attr_shards.py
--------------------------
Extrai features DINO por objeto (ROI mean-pool) + labels de atributos VG
e salva em shards .h5 prontos para treino da AttributeHead.

Shapes de saída por shard (objects_per_shard=10_000):
    obj_feats   : [N, 768]  float16  — feature DINO ROI-poolada por objeto
    attr_labels : [N, 200]  float16  — multi-hot de atributos VG (top-200)

Uso:
    python embeddings/generate_vg_attr_shards.py

Retomada automática:
    Se interrompido, basta rodar novamente — ShardWriter continua do ponto exato.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Optional

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.vg_dataset import (
    build_attribute_vocab,
    build_vg150_vocab,
    deterministic_split,
    get_image_path,
    load_scene_graphs,
)
from models.encoders.dinov3_extrator import DinoSceneEncoder
from utils.io_utils import ensure_dir

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
OBJ_FEAT_DIM  = 768
GRID_SIZE     = 32   # grade 32×32 de patches após pool no generate_shards.py

_TRANSFORM = T.Compose([
    T.Resize((448, 448)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# ROI mean-pool
# ---------------------------------------------------------------------------

def roi_mean_pool(
    grid: torch.Tensor,       # [32, 32, 768]
    bbox_xywh: tuple[float, float, float, float],
    img_w: float,
    img_h: float,
) -> torch.Tensor:
    """
    Extrai feature de um objeto via mean-pool sobre os patches do grid DINO.

    Mapeia bbox (x, y, w, h) em pixels originais para indices inteiros no
    grid 32x32, depois faz mean dos patches cobertos.

    Returns [768].
    """
    x, y, w, h = bbox_xywh
    sx = GRID_SIZE / max(img_w, 1.0)
    sy = GRID_SIZE / max(img_h, 1.0)

    c1 = max(0, math.floor(x * sx))
    r1 = max(0, math.floor(y * sy))
    c2 = min(GRID_SIZE, math.ceil((x + w) * sx))
    r2 = min(GRID_SIZE, math.ceil((y + h) * sy))

    # garante ao menos 1 patch
    if c2 <= c1:
        c2 = min(GRID_SIZE, c1 + 1)
    if r2 <= r1:
        r2 = min(GRID_SIZE, r1 + 1)

    return grid[r1:r2, c1:c2, :].mean(dim=(0, 1))  # [768]


# ---------------------------------------------------------------------------
# Construção da lista de amostras por imagem
# ---------------------------------------------------------------------------

def build_image_samples(
    scene_graphs: list[dict],
    indices: list[int],
    vg_dir: str,
    obj_set: set[str],
    attr_to_idx: dict[str, int],
) -> list[dict]:
    """
    Retorna lista de dicts por imagem, cada um com todos os objetos válidos.

    Objeto válido: está no obj_set VG-150, tem bbox válida e tem >= 1 atributo
    no vocabulário.
    """
    samples: list[dict] = []
    for idx in tqdm(indices, desc="Indexando imagens VG"):
        sg = scene_graphs[idx]
        img_path = get_image_path(vg_dir, sg["image_id"])
        if img_path is None:
            continue

        img_w = float(sg.get("width") or 1)
        img_h = float(sg.get("height") or 1)

        objects: list[dict] = []
        for obj in sg.get("objects", []):
            names = obj.get("names") or [obj.get("name", "")]
            name = names[0].lower().strip() if names else ""
            if name not in obj_set:
                continue

            raw_attrs = [
                a.lower().strip()
                for a in obj.get("attributes", [])
                if a.lower().strip() in attr_to_idx
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

            attr_indices = sorted({attr_to_idx[a] for a in raw_attrs})
            objects.append({"bbox": (x, y, w, h), "attr_indices": attr_indices})

        if not objects:
            continue

        samples.append({
            "img_path": img_path,
            "img_w": img_w,
            "img_h": img_h,
            "objects": objects,
        })

    return samples


# ---------------------------------------------------------------------------
# ShardWriter (obj_feats + attr_labels)
# ---------------------------------------------------------------------------

def _shard_path(output_dir: str, shard_idx: int) -> str:
    return os.path.join(output_dir, f"shard_{str(shard_idx).zfill(6)}.h5")


def _get_resume_state(output_dir: str) -> tuple[int, int]:
    existing = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0, 0
    shard_idx = int(existing[-1].replace("shard_", "").replace(".h5", ""))
    path = _shard_path(output_dir, shard_idx)
    try:
        with h5py.File(path, "r") as f:
            n = f["obj_feats"].shape[0]
    except Exception:
        n = 0
    return shard_idx, n


def _count_global_samples(output_dir: str, objects_per_shard: int) -> int:
    existing = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0
    total = (len(existing) - 1) * objects_per_shard
    path = os.path.join(output_dir, existing[-1])
    try:
        with h5py.File(path, "r") as f:
            total += f["obj_feats"].shape[0]
    except Exception:
        pass
    return total


class AttrShardWriter:
    def __init__(
        self,
        output_dir: str,
        n_attrs: int,
        objects_per_shard: int = 10_000,
        resume: bool = True,
    ):
        self.output_dir       = output_dir
        self.n_attrs          = n_attrs
        self.objects_per_shard = objects_per_shard

        ensure_dir(output_dir)

        if resume:
            self._shard_idx, self._in_shard = _get_resume_state(output_dir)
        else:
            self._shard_idx, self._in_shard = 0, 0

        self._file     : Optional[h5py.File]    = None
        self._ds_feats : Optional[h5py.Dataset] = None
        self._ds_labels: Optional[h5py.Dataset] = None

        if self._in_shard > 0:
            self._open_shard(new=False)

        self.total_written = _count_global_samples(output_dir, objects_per_shard)
        print(
            f"[AttrShardWriter] shard={self._shard_idx} | "
            f"objetos no shard atual={self._in_shard} | "
            f"total ja exportado={self.total_written:,}"
        )

    def _open_shard(self, new: bool = True) -> None:
        if self._file is not None:
            self._close_shard()
        path = _shard_path(self.output_dir, self._shard_idx)
        if new or not os.path.exists(path):
            self._file = h5py.File(path, "w")
            self._ds_feats = self._file.create_dataset(
                "obj_feats",
                shape=(0, OBJ_FEAT_DIM),
                maxshape=(None, OBJ_FEAT_DIM),
                dtype="float16",
                compression="gzip",
                chunks=(256, OBJ_FEAT_DIM),
            )
            self._ds_labels = self._file.create_dataset(
                "attr_labels",
                shape=(0, self.n_attrs),
                maxshape=(None, self.n_attrs),
                dtype="float16",
                compression="gzip",
                chunks=(256, self.n_attrs),
            )
        else:
            self._file      = h5py.File(path, "a")
            self._ds_feats  = self._file["obj_feats"]
            self._ds_labels = self._file["attr_labels"]

    def _close_shard(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None

    def _rotate_shard(self) -> None:
        self._close_shard()
        self._shard_idx += 1
        self._in_shard   = 0
        self._open_shard(new=True)

    def write_batch(
        self,
        feats : np.ndarray,   # [K, 768]  float16
        labels: np.ndarray,   # [K, n_attrs] float16
    ) -> None:
        k      = feats.shape[0]
        offset = 0
        while offset < k:
            if self._file is None:
                self._open_shard(new=(self._in_shard == 0))
            slots    = self.objects_per_shard - self._in_shard
            n        = min(slots, k - offset)
            end      = offset + n
            new_size = self._in_shard + n

            self._ds_feats.resize(new_size, axis=0)
            self._ds_labels.resize(new_size, axis=0)
            self._ds_feats [self._in_shard:new_size] = feats [offset:end]
            self._ds_labels[self._in_shard:new_size] = labels[offset:end]

            self._in_shard     = new_size
            self.total_written += n
            offset             = end

            if self._in_shard >= self.objects_per_shard:
                self._rotate_shard()

    def close(self) -> None:
        self._close_shard()

    def __enter__(self) -> "AttrShardWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Exportação principal
# ---------------------------------------------------------------------------

def export_vg_attr_shards(
    image_samples     : list[dict],
    dino              : DinoSceneEncoder,
    output_dir        : str,
    n_attrs           : int,
    objects_per_shard : int = 10_000,
    batch_size        : int = 8,
    resume            : bool = True,
) -> None:
    """
    Itera sobre imagens VG em batches, extrai grid DINO, ROI-poola por objeto
    e salva obj_feats + attr_labels em shards .h5.

    Parameters
    ----------
    image_samples:
        Lista de dicts com img_path, img_w, img_h, objects.
    dino:
        DinoSceneEncoder já carregado em GPU.
    output_dir:
        Destino dos shards .h5.
    n_attrs:
        Tamanho do vocabulário de atributos.
    objects_per_shard:
        Objetos por arquivo .h5 (padrão: 10_000).
    batch_size:
        Imagens por forward pass DINO.
    resume:
        Continua de onde parou após crash.
    """
    with AttrShardWriter(output_dir, n_attrs, objects_per_shard, resume) as writer:
        skip_objects = writer.total_written
        skipped      = 0

        pbar = tqdm(
            range(0, len(image_samples), batch_size),
            desc=f"Exportando → {output_dir}",
        )

        for batch_start in pbar:
            batch = image_samples[batch_start: batch_start + batch_size]

            # Conta objetos neste batch para skip de retomada
            batch_obj_counts = [len(s["objects"]) for s in batch]
            total_batch_objs = sum(batch_obj_counts)

            if skipped + total_batch_objs <= skip_objects:
                skipped += total_batch_objs
                continue

            # Carrega e transforma imagens
            imgs_pil   = [Image.open(s["img_path"]).convert("RGB") for s in batch]
            imgs_tensor = torch.stack([_TRANSFORM(img) for img in imgs_pil])

            try:
                with torch.no_grad():
                    _, hr_features = dino.extract_features(imgs_tensor.to("cuda"))
                    # hr_features: [B, 768, H_hr, W_hr] → pool → [B, 768, 32, 32]
                    grid32 = F.adaptive_avg_pool2d(hr_features, (GRID_SIZE, GRID_SIZE))
                    # [B, 768, 32, 32] → [B, 32, 32, 768]
                    grid32 = grid32.permute(0, 2, 3, 1).cpu()

            except RuntimeError as e:
                if "CUDA" in str(e).upper() and torch.cuda.is_available():
                    print(torch.cuda.memory_summary())
                raise

            # ROI-pool por objeto
            batch_feats : list[np.ndarray] = []
            batch_labels: list[np.ndarray] = []

            for b_i, sample in enumerate(batch):
                grid = grid32[b_i]  # [32, 32, 768]
                for obj in sample["objects"]:
                    feat = roi_mean_pool(
                        grid, obj["bbox"], sample["img_w"], sample["img_h"]
                    )  # [768]
                    label = np.zeros(n_attrs, dtype="float16")
                    for ai in obj["attr_indices"]:
                        label[ai] = 1.0

                    batch_feats.append(feat.numpy().astype("float16"))
                    batch_labels.append(label)

            if not batch_feats:
                continue

            feats_np  = np.stack(batch_feats,  axis=0)   # [K, 768]
            labels_np = np.stack(batch_labels, axis=0)   # [K, n_attrs]

            # Aplica skip parcial (borda de retomada)
            already_done = max(0, skip_objects - skipped)
            if already_done > 0:
                feats_np  = feats_np [already_done:]
                labels_np = labels_np[already_done:]

            skipped += total_batch_objs

            if feats_np.shape[0] > 0:
                writer.write_batch(feats_np, labels_np)

            pbar.set_postfix(total=f"{writer.total_written:,}")

    print(f"\nExportacao concluida — {writer.total_written:,} objetos em {output_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    VG_DIR            = "G:/vg"
    TRAIN_OUTPUT_DIR  = "G:/vg/attr_shards/train/"
    VAL_OUTPUT_DIR    = "G:/vg/attr_shards/val/"
    N_ATTRS           = 200
    OBJECTS_PER_SHARD = 10_000
    BATCH_SIZE        = 8    # imagens por forward DINO
    TEST_RATIO        = 0.2
    SPLIT_SEED        = 42

    scene_graphs = load_scene_graphs(VG_DIR)

    _, _, obj_set, _ = build_vg150_vocab(scene_graphs)
    attr_list, attr_set = build_attribute_vocab(scene_graphs, n_attrs=N_ATTRS)
    attr_to_idx = {a: i for i, a in enumerate(attr_list)}

    train_indices, val_indices = deterministic_split(
        len(scene_graphs), test_ratio=TEST_RATIO, seed=SPLIT_SEED
    )

    print(f"Split: {len(train_indices):,} treino | {len(val_indices):,} val")

    train_samples = build_image_samples(scene_graphs, train_indices, VG_DIR, obj_set, attr_to_idx)
    val_samples   = build_image_samples(scene_graphs, val_indices,   VG_DIR, obj_set, attr_to_idx)

    print(f"Imagens validas: {len(train_samples):,} treino | {len(val_samples):,} val")

    dino = DinoSceneEncoder(device="cuda", upsampler="anyup")
    dino.model.eval()

    torch.cuda.empty_cache()
    print(
        f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total | "
        f"{(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB livre"
    )

    export_vg_attr_shards(
        train_samples, dino,
        output_dir        = TRAIN_OUTPUT_DIR,
        n_attrs           = N_ATTRS,
        objects_per_shard = OBJECTS_PER_SHARD,
        batch_size        = BATCH_SIZE,
    )

    export_vg_attr_shards(
        val_samples, dino,
        output_dir        = VAL_OUTPUT_DIR,
        n_attrs           = N_ATTRS,
        objects_per_shard = OBJECTS_PER_SHARD,
        batch_size        = BATCH_SIZE,
    )
