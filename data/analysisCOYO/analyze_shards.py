"""
analyze_shards.py
-----------------
Inspeciona a estrutura, consistência e integridade dos shards H5 do COYO.

Métricas geradas
----------------
Por shard:
  - N de amostras, shapes e dtypes
  - Presença de NaN / Inf
  - Amostras zeradas (norma < 1e-6)
  - Tamanho em disco (MB)
  - Taxa de compressão estimada

Globais:
  - Total de amostras em cada split
  - Distribuição de amostras por shard
  - Detecção de shards corrompidos ou faltando
  - Consistência de shape entre shards

Saída
-----
  results/dataset_analysis_shards.json   — dados estruturados por split
  results/dataset_analysis_shards.png    — painel de gráficos

Uso
---
  python data/analyze_shards.py
  python data/analyze_shards.py \
      --train_dir F:/COYO/embeds/train_anyup \
      --val_dir   G:/coyo/embeds/val_anyup   \
      --test_dir  G:/coyo/embeds/test_anyup
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Análise de um único shard
# ---------------------------------------------------------------------------

def _inspect_shard(path: str) -> Dict[str, Any]:
    """
    Extrai métricas de integridade e estrutura de um arquivo .h5.

    Parameters
    ----------
    path:
        Caminho do arquivo .h5.

    Returns
    -------
    dict com as métricas do shard, incluindo flag ``ok`` (bool).
    """
    result: Dict[str, Any] = {
        "path":     path,
        "filename": os.path.basename(path),
        "size_mb":  os.path.getsize(path) / 1e6,
        "ok":       False,
        "error":    None,
    }

    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            result["keys"] = keys

            if "visual_feats" not in f or "text_feats" not in f:
                result["error"] = f"Datasets obrigatórios ausentes. Keys: {keys}"
                return result

            vf = f["visual_feats"]
            tf = f["text_feats"]

            result["n_samples"]      = int(vf.shape[0])
            result["visual_shape"]   = list(vf.shape)
            result["text_shape"]     = list(tf.shape)
            result["visual_dtype"]   = str(vf.dtype)
            result["text_dtype"]     = str(tf.dtype)
            result["has_global"]     = "visual_global" in f

            if result["has_global"]:
                gf = f["visual_global"]
                result["global_shape"] = list(gf.shape)
                result["global_dtype"] = str(gf.dtype)

            # Verificação rápida: amostra 10% ou 200 linhas
            n = result["n_samples"]
            sample_size = min(max(int(n * 0.1), 1), 200)
            idxs = np.sort(np.random.choice(n, sample_size, replace=False)) if n > 1 else [0]

            v_sample = vf[idxs].astype(np.float32)
            t_sample = tf[idxs].astype(np.float32)

            # NaN / Inf
            result["visual_has_nan"]  = bool(np.isnan(v_sample).any())
            result["visual_has_inf"]  = bool(np.isinf(v_sample).any())
            result["text_has_nan"]    = bool(np.isnan(t_sample).any())
            result["text_has_inf"]    = bool(np.isinf(t_sample).any())

            # Amostras zeradas: norma < 1e-6 ao longo do último eixo
            v_norms = np.linalg.norm(v_sample.reshape(sample_size, -1), axis=1)
            t_norms = np.linalg.norm(t_sample.reshape(sample_size, -1), axis=1)
            result["visual_zero_ratio"] = float((v_norms < 1e-6).mean())
            result["text_zero_ratio"]   = float((t_norms < 1e-6).mean())

            # Estatísticas básicas de norma
            result["visual_norm_mean"] = float(v_norms.mean())
            result["visual_norm_std"]  = float(v_norms.std())
            result["text_norm_mean"]   = float(t_norms.mean())
            result["text_norm_std"]    = float(t_norms.std())

            # Shape consistency
            expected_v_ndim = 3  # [N, P, D]
            expected_t_ndim = 3  # [N, 1, D] ou [N, T, D]
            result["visual_shape_ok"] = len(result["visual_shape"]) == expected_v_ndim
            result["text_shape_ok"]   = len(result["text_shape"])   >= expected_t_ndim

            result["ok"] = (
                not result["visual_has_nan"]
                and not result["visual_has_inf"]
                and not result["text_has_nan"]
                and not result["text_has_inf"]
                and result["visual_zero_ratio"] < 0.05
                and result["visual_shape_ok"]
                and result["text_shape_ok"]
            )

    except Exception as e:
        result["error"] = str(e)

    return result


# ---------------------------------------------------------------------------
# Análise de um split completo
# ---------------------------------------------------------------------------

def _analyze_split(split_dir: str, split_name: str) -> Dict[str, Any]:
    """
    Inspeciona todos os shards de um split e consolida métricas.

    Parameters
    ----------
    split_dir:
        Diretório com os arquivos .h5 do split.
    split_name:
        Nome do split ('train', 'val', 'test').

    Returns
    -------
    dict com métricas consolidadas do split.
    """
    pattern = os.path.join(split_dir, "**", "*.h5")
    shard_files = sorted(glob.glob(pattern, recursive=True))

    if not shard_files:
        return {
            "split":       split_name,
            "dir":         split_dir,
            "n_shards":    0,
            "n_samples":   0,
            "error":       "Nenhum arquivo .h5 encontrado",
            "shards":      [],
        }

    print(f"\n[{split_name}] Inspecionando {len(shard_files)} shards em {split_dir} ...")
    shard_results: List[Dict] = []

    for path in tqdm(shard_files, desc=f"  {split_name}", unit="shard"):
        shard_results.append(_inspect_shard(path))

    # ── Consolidação ─────────────────────────────────────────────────────────
    total_samples    = sum(s.get("n_samples", 0) for s in shard_results)
    n_corrupt        = sum(1 for s in shard_results if not s["ok"])
    n_nan_visual     = sum(1 for s in shard_results if s.get("visual_has_nan"))
    n_inf_visual     = sum(1 for s in shard_results if s.get("visual_has_inf"))
    n_high_zeros     = sum(1 for s in shard_results if s.get("visual_zero_ratio", 0) > 0.05)

    sample_counts = [s.get("n_samples", 0) for s in shard_results]
    sizes_mb      = [s.get("size_mb", 0)   for s in shard_results]

    # Shape majoritário
    visual_shapes = [tuple(s["visual_shape"]) for s in shard_results if "visual_shape" in s]
    text_shapes   = [tuple(s["text_shape"])   for s in shard_results if "text_shape"   in s]
    from collections import Counter
    dominant_visual = Counter(visual_shapes).most_common(1)[0] if visual_shapes else (None, 0)
    dominant_text   = Counter(text_shapes).most_common(1)[0]   if text_shapes   else (None, 0)

    return {
        "split":             split_name,
        "dir":               split_dir,
        "n_shards":          len(shard_files),
        "n_samples":         total_samples,
        "n_corrupt":         n_corrupt,
        "n_nan_visual":      n_nan_visual,
        "n_inf_visual":      n_inf_visual,
        "n_high_zero_ratio": n_high_zeros,
        "corrupt_rate":      n_corrupt / max(len(shard_files), 1),
        "dominant_visual_shape": list(dominant_visual[0]) if dominant_visual[0] else None,
        "dominant_text_shape":   list(dominant_text[0])   if dominant_text[0]   else None,
        "samples_per_shard": {
            "mean":  float(np.mean(sample_counts)) if sample_counts else 0,
            "std":   float(np.std(sample_counts))  if sample_counts else 0,
            "min":   int(np.min(sample_counts))    if sample_counts else 0,
            "max":   int(np.max(sample_counts))    if sample_counts else 0,
        },
        "size_mb_per_shard": {
            "mean":  float(np.mean(sizes_mb)) if sizes_mb else 0,
            "total": float(np.sum(sizes_mb))  if sizes_mb else 0,
        },
        "shards": shard_results,
    }


# ---------------------------------------------------------------------------
# Plotagem
# ---------------------------------------------------------------------------

def _plot_results(splits: List[Dict], out_path: str) -> None:
    """
    Gera painel de visualização da estrutura dos shards por split.

    Parameters
    ----------
    splits:
        Lista de dicionários retornados por _analyze_split().
    out_path:
        Caminho de saída do PNG.
    """
    n_splits = len([s for s in splits if s["n_shards"] > 0])
    if n_splits == 0:
        print("[WARN] Nenhum split válido para plotar.")
        return

    fig, axes = plt.subplots(2, max(n_splits, 2), figsize=(7 * max(n_splits, 2), 12))
    fig.suptitle("COYO Shards — Análise de Estrutura", fontsize=15, fontweight="bold")

    col = 0
    for split in splits:
        if split["n_shards"] == 0:
            continue

        shards = split["shards"]

        # ── Linha 1: Distribuição de amostras por shard ──────────────────────
        counts = [s.get("n_samples", 0) for s in shards]
        axes[0, col].hist(counts, bins=min(30, len(counts)), color="#3498db",
                          edgecolor="white", linewidth=0.4)
        axes[0, col].set_title(f"[{split['split']}] Amostras por Shard\n"
                               f"Total: {split['n_samples']:,} | {split['n_shards']} shards")
        axes[0, col].set_xlabel("Nº de amostras")
        axes[0, col].set_ylabel("Contagem de shards")

        # Linha vertical na mediana
        if counts:
            axes[0, col].axvline(np.median(counts), color="red", linestyle="--",
                                 linewidth=1.2, label=f"Mediana: {int(np.median(counts))}")
            axes[0, col].legend(fontsize=8)

        # ── Linha 2: Qualidade dos shards ────────────────────────────────────
        quality_labels = ["OK", "Corrompidos", "NaN visual", "Inf visual", "Zeros >5%"]
        quality_vals = [
            split["n_shards"] - split["n_corrupt"],
            split["n_corrupt"],
            split["n_nan_visual"],
            split["n_inf_visual"],
            split["n_high_zero_ratio"],
        ]
        colors_q = ["#2ecc71", "#e74c3c", "#f39c12", "#e74c3c", "#95a5a6"]
        bars = axes[1, col].bar(quality_labels, quality_vals, color=colors_q,
                                edgecolor="white")
        for bar, val in zip(bars, quality_vals):
            if val > 0:
                axes[1, col].text(bar.get_x() + bar.get_width() / 2,
                                  bar.get_height() + 0.1,
                                  str(val), ha="center", fontsize=9)
        axes[1, col].set_title(f"[{split['split']}] Qualidade dos Shards")
        axes[1, col].set_ylabel("Nº de shards")
        axes[1, col].tick_params(axis="x", rotation=20, labelsize=8)

        col += 1

    # Esconde subplots extras
    for extra in range(col, axes.shape[1]):
        axes[0, extra].set_visible(False)
        axes[1, extra].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo em: {out_path}")


# ---------------------------------------------------------------------------
# Impressão de relatório de console
# ---------------------------------------------------------------------------

def _print_report(splits: List[Dict]) -> None:
    """
    Imprime relatório resumido de cada split no console.

    Parameters
    ----------
    splits:
        Lista de resultados por split.
    """
    print("\n" + "=" * 65)
    print("RELATÓRIO DE SHARDS H5")
    print("=" * 65)

    total_samples = 0
    for s in splits:
        print(f"\n  Split [{s['split'].upper()}]  ← {s['dir']}")
        if s["n_shards"] == 0:
            print(f"    ⚠  Nenhum shard encontrado!")
            continue
        print(f"    Shards           : {s['n_shards']:>8}")
        print(f"    Amostras totais  : {s['n_samples']:>8,}")
        print(f"    Corrompidos      : {s['n_corrupt']:>8}"
              f"  ({100*s['corrupt_rate']:.1f}%)")
        print(f"    NaN visual       : {s['n_nan_visual']:>8}")
        print(f"    Inf visual       : {s['n_inf_visual']:>8}")
        print(f"    Zeros >5%        : {s['n_high_zero_ratio']:>8}")
        if s["dominant_visual_shape"]:
            print(f"    Shape visual     : {s['dominant_visual_shape']}")
        if s["dominant_text_shape"]:
            print(f"    Shape texto      : {s['dominant_text_shape']}")
        sp = s["samples_per_shard"]
        print(f"    Amostras/shard   : {sp['mean']:.0f} ± {sp['std']:.0f}"
              f"  (min {sp['min']}, max {sp['max']})")
        sz = s["size_mb_per_shard"]
        print(f"    Tamanho total    : {sz['total']:.1f} MB  "
              f"({sz['mean']:.1f} MB/shard)")

        total_samples += s["n_samples"]

    print(f"\n  TOTAL DE AMOSTRAS (todos os splits): {total_samples:,}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(
    train_dir: Optional[str],
    val_dir:   Optional[str],
    test_dir:  Optional[str],
    output_dir: str,
) -> None:
    """
    Analisa os shards H5 dos três splits do COYO.

    Parameters
    ----------
    train_dir, val_dir, test_dir:
        Diretórios dos shards. None = ignorar split.
    output_dir:
        Diretório de saída para JSON e PNG.
    """
    os.makedirs(output_dir, exist_ok=True)

    splits: List[Dict] = []
    for name, path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if path:
            splits.append(_analyze_split(path, name))
        else:
            splits.append({"split": name, "n_shards": 0, "n_samples": 0, "shards": []})

    _print_report(splits)

    # Serializa (remove lista detalhada de shards para JSON legível)
    summary = []
    for s in splits:
        s_clean = {k: v for k, v in s.items() if k != "shards"}
        s_clean["n_ok_shards"] = s.get("n_shards", 0) - s.get("n_corrupt", 0)
        summary.append(s_clean)

    output: Dict = {
        "timestamp": datetime.now().isoformat(),
        "splits":    summary,
    }
    json_path = os.path.join(output_dir, "dataset_analysis_shards.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"\nJSON salvo em: {json_path}")

    png_path = os.path.join(output_dir, "dataset_analysis_shards.png")
    valid_splits = [s for s in splits if s["n_shards"] > 0]
    if valid_splits:
        _plot_results(valid_splits, png_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise de shards H5 do COYO.")
    parser.add_argument("--train_dir",  default="F:/COYO/embeds/train_anyup",  help="Dir train")
    parser.add_argument("--val_dir",    default="G:/coyo/embeds/val_anyup",    help="Dir val")
    parser.add_argument("--test_dir",   default="G:/coyo/embeds/test_anyup",   help="Dir test")
    parser.add_argument("--output_dir", default="results",                      help="Dir de saída")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        train_dir=args.train_dir  or None,
        val_dir=args.val_dir      or None,
        test_dir=args.test_dir    or None,
        output_dir=args.output_dir,
    )
