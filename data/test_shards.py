"""
test_shards.py
--------------
Inspeção e comparação estatística de shards H5.

Suporta dois modos:
  1. Inspeção detalhada de um diretório (shard por shard)
  2. Comparação agregada entre dois conjuntos (ex.: com vs sem AnyUp)

Uso:
    python data/test_shards.py
"""

from __future__ import annotations

import os
import glob
from typing import Dict, List, Optional

import h5py
import numpy as np


# ---------------------------------------------------------------------------
# Inspeção detalhada
# ---------------------------------------------------------------------------

def inspect_dir(folder: str, max_shards: Optional[int] = None) -> None:
    """
    Imprime shape, dtype, nan/inf e estatísticas básicas de cada shard.

    Parameters
    ----------
    folder:
        Diretório contendo shard_*.h5
    max_shards:
        Se definido, limita a inspeção aos N primeiros shards (acelera).
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))

    if not files:
        print(f"[ERRO] Nenhum .h5 encontrado em {folder}")
        return

    if max_shards is not None:
        files = files[:max_shards]

    print(f"\n{'='*70}")
    print(f"INSPEÇÃO: {folder}")
    print(f"{'='*70}")
    print(f"Shards encontrados: {len(files)}")

    for sf in files:
        print(f"\n--- {os.path.basename(sf)} ---")
        try:
            with h5py.File(sf, "r") as f:
                for key in f.keys():
                    arr = f[key][:]
                    # Para datasets string (image_paths), trata separadamente
                    if arr.dtype.kind in ("O", "S", "U"):
                        print(f"  {key}: shape={arr.shape}  dtype={arr.dtype}")
                        if arr.size > 0:
                            sample = arr[0]
                            if isinstance(sample, bytes):
                                sample = sample.decode("utf-8", errors="ignore")
                            print(f"    exemplo: {sample[:80]}")
                        continue

                    print(f"  {key}: shape={arr.shape}  dtype={arr.dtype}")
                    print(f"    nan={np.isnan(arr).any()}  inf={np.isinf(arr).any()}")
                    flat = arr.reshape(-1)
                    print(f"    min={flat.min():.4f}  max={flat.max():.4f}  "
                          f"mean={flat.mean():.4f}  std={flat.std():.4f}")
        except Exception as e:
            print(f"  ERRO ao abrir: {e}")


# ---------------------------------------------------------------------------
# Estatísticas agregadas
# ---------------------------------------------------------------------------

def aggregate_stats(folder: str) -> Dict:
    """
    Calcula estatísticas agregadas sobre todos os shards de um diretório.

    Returns
    -------
    dict com:
        n_shards, n_samples, total_size_gb,
        visual_shape, text_shape, has_paths,
        visual_norm_mean, text_norm_mean,
        nan_count, inf_count
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))

    stats = {
        "folder": folder,
        "n_shards": 0,
        "n_samples": 0,
        "total_size_gb": 0.0,
        "visual_shape": None,
        "text_shape": None,
        "global_shape": None,
        "has_paths": False,
        "visual_dtype": None,
        "visual_norm_mean": 0.0,
        "visual_norm_std": 0.0,
        "text_norm_mean": 0.0,
        "text_norm_std": 0.0,
        "nan_count": 0,
        "inf_count": 0,
        "corrupted_shards": [],
    }

    if not files:
        return stats

    stats["n_shards"] = len(files)

    v_norms: List[float] = []
    t_norms: List[float] = []

    for sf in files:
        try:
            stats["total_size_gb"] += os.path.getsize(sf) / 1e9
            with h5py.File(sf, "r") as f:
                v = f["visual_feats"]
                t = f["text_feats"]
                n = v.shape[0]
                stats["n_samples"] += n

                if stats["visual_shape"] is None:
                    stats["visual_shape"] = tuple(v.shape[1:])  # (P, D)
                    stats["text_shape"] = tuple(t.shape[1:])
                    stats["visual_dtype"] = str(v.dtype)
                    if "visual_global" in f:
                        stats["global_shape"] = tuple(f["visual_global"].shape[1:])
                    stats["has_paths"] = "image_paths" in f

                # Amostra ~100 vetores por shard para norma (evita ler tudo)
                sample_idx = np.random.choice(n, size=min(100, n), replace=False)
                sample_idx.sort()
                v_sample = v[sample_idx].astype(np.float32)         # [K, P, D]
                t_sample = t[sample_idx].astype(np.float32)         # [K, 1, D]

                # Norma L2 do mean dos patches (proxy de magnitude global)
                v_mean = v_sample.mean(axis=1)                      # [K, D]
                v_norms.extend(np.linalg.norm(v_mean, axis=-1).tolist())
                t_norms.extend(np.linalg.norm(t_sample.squeeze(1), axis=-1).tolist())

                if np.isnan(v_sample).any() or np.isnan(t_sample).any():
                    stats["nan_count"] += 1
                if np.isinf(v_sample).any() or np.isinf(t_sample).any():
                    stats["inf_count"] += 1

        except Exception as e:
            stats["corrupted_shards"].append((sf, str(e)))

    if v_norms:
        stats["visual_norm_mean"] = float(np.mean(v_norms))
        stats["visual_norm_std"] = float(np.std(v_norms))
    if t_norms:
        stats["text_norm_mean"] = float(np.mean(t_norms))
        stats["text_norm_std"] = float(np.std(t_norms))

    return stats


def print_stats(stats: Dict) -> None:
    """Imprime estatísticas em formato tabular."""
    print(f"\n{'─'*70}")
    print(f"  {stats['folder']}")
    print(f"{'─'*70}")
    print(f"  Shards            : {stats['n_shards']}")
    print(f"  Amostras          : {stats['n_samples']:,}")
    print(f"  Tamanho total     : {stats['total_size_gb']:.2f} GB")
    print(f"  visual_feats shape: [N, {stats['visual_shape']}]   dtype={stats['visual_dtype']}")
    print(f"  text_feats shape  : [N, {stats['text_shape']}]")
    print(f"  visual_global     : {'[N, ' + str(stats['global_shape']) + ']' if stats['global_shape'] else 'AUSENTE'}")
    print(f"  image_paths       : {'PRESENTE' if stats['has_paths'] else 'AUSENTE'}")
    print(f"  visual norm (mean): {stats['visual_norm_mean']:.4f} ± {stats['visual_norm_std']:.4f}")
    print(f"  text norm (mean)  : {stats['text_norm_mean']:.4f} ± {stats['text_norm_std']:.4f}")
    print(f"  NaN/Inf shards    : {stats['nan_count']} / {stats['inf_count']}")
    if stats["corrupted_shards"]:
        print(f"  CORROMPIDOS       : {len(stats['corrupted_shards'])}")
        for path, err in stats["corrupted_shards"][:3]:
            print(f"    - {os.path.basename(path)}: {err}")


def compare(stats_a: Dict, stats_b: Dict, label_a: str, label_b: str) -> None:
    """Imprime comparação lado a lado entre dois conjuntos."""
    print(f"\n{'='*70}")
    print(f"COMPARAÇÃO: {label_a}  vs  {label_b}")
    print(f"{'='*70}")
    print(f"  {'Metric':<25} {label_a:>20} {label_b:>20}")
    print(f"  {'-'*25} {'-'*20:>20} {'-'*20:>20}")

    def row(name, va, vb, fmt="{:.4f}"):
        sa = fmt.format(va) if isinstance(va, float) else str(va)
        sb = fmt.format(vb) if isinstance(vb, float) else str(vb)
        print(f"  {name:<25} {sa:>20} {sb:>20}")

    row("Shards",            stats_a["n_shards"],         stats_b["n_shards"])
    row("Amostras",          f"{stats_a['n_samples']:,}", f"{stats_b['n_samples']:,}")
    row("Tamanho (GB)",      stats_a["total_size_gb"],    stats_b["total_size_gb"], "{:.2f}")
    row("Visual shape",      str(stats_a["visual_shape"]), str(stats_b["visual_shape"]))
    row("Bytes/amostra (raw)", _bytes_per_sample(stats_a), _bytes_per_sample(stats_b), "{:.0f}")
    row("Compressão (×)",    _compression_ratio(stats_a), _compression_ratio(stats_b), "{:.2f}")
    row("image_paths",       stats_a["has_paths"],         stats_b["has_paths"])
    row("Visual norm (μ)",   stats_a["visual_norm_mean"],  stats_b["visual_norm_mean"])
    row("Visual norm (σ)",   stats_a["visual_norm_std"],   stats_b["visual_norm_std"])
    row("Text norm (μ)",     stats_a["text_norm_mean"],    stats_b["text_norm_mean"])
    row("Text norm (σ)",     stats_a["text_norm_std"],     stats_b["text_norm_std"])
    row("NaN shards",        stats_a["nan_count"],         stats_b["nan_count"])
    row("Inf shards",        stats_a["inf_count"],         stats_b["inf_count"])


def _bytes_per_sample(stats: Dict) -> float:
    """Bytes raw por amostra (visual + text + global), float16."""
    if stats["visual_shape"] is None:
        return 0
    v = int(np.prod(stats["visual_shape"]))
    t = int(np.prod(stats["text_shape"]))
    g = int(np.prod(stats["global_shape"])) if stats["global_shape"] else 0
    return (v + t + g) * 2  # float16


def _compression_ratio(stats: Dict) -> float:
    """Razão entre tamanho raw esperado e tamanho real em disco."""
    if stats["n_samples"] == 0 or stats["total_size_gb"] == 0:
        return 0.0
    raw_gb = (_bytes_per_sample(stats) * stats["n_samples"]) / 1e9
    return raw_gb / stats["total_size_gb"]


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ANÁLISE DE SHARDS — Com vs Sem Upsampling")
    print("=" * 70)

    pairs = [
        ("train", "F:/COYO/embeds/train_anyup/", "E:/COYO/embeds/train_noup/"),
        ("val",   "G:/coyo/embeds/val_anyup/",   "E:/COYO/embeds/val_noup/"),
        ("test",  "G:/coyo/embeds/test_anyup/",  "E:/COYO/embeds/test_noup/"),
    ]

    for split, dir_anyup, dir_noup in pairs:
        print(f"\n\n{'#'*70}")
        print(f"# SPLIT: {split.upper()}")
        print(f"{'#'*70}")

        stats_anyup = aggregate_stats(dir_anyup)
        stats_noup  = aggregate_stats(dir_noup)

        print_stats(stats_anyup)
        print_stats(stats_noup)

        if stats_anyup["n_shards"] > 0 and stats_noup["n_shards"] > 0:
            compare(stats_anyup, stats_noup,
                    label_a=f"{split}_anyup",
                    label_b=f"{split}_noup")
