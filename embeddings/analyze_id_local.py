"""
analyze_id_local.py
-------------------
Análise geométrica LOCAL via Dimensão Intrínseca (Intrinsic Dimension)
estimada pelo método TwoNN [Facco et al., 2017].

Para cada imagem, mede a complexidade geométrica da nuvem de patches:
  - SEM upsampling: matriz [196, 768] de patches LR do DINOv3
  - COM AnyUp     : matriz [1024, 768] de patches HR upsampled

Hipótese:
  - ID similar (ex: 8.4 → 8.8) → patches HR são redundantes, mesma subvariedade
  - ID muito maior (ex: 8.4 → 22) → upsampling adiciona complexidade real
    (microtexturas / detalhes de alta frequência)

TwoNN (Facco et al., 2017):
  Para cada ponto x_i, calcula r1 = dist(x_i, NN1) e r2 = dist(x_i, NN2).
  Define μ_i = r2 / r1 (≥ 1). A CDF empírica de μ segue
        F(μ) = 1 - μ^(-d)
  onde d é a dimensão intrínseca. Estima-se d por regressão linear sobre
        -log(1 - F(μ_i))  vs  log(μ_i)
  usando apenas a porção bem comportada (descarte de empates e outliers).

Uso:
    python embeddings/analyze_id_local.py
    python embeddings/analyze_id_local.py --n_images 500 --split test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io_utils import ensure_dir


# ---------------------------------------------------------------------------
# TwoNN estimator
# ---------------------------------------------------------------------------

def twonn_id(
    X: np.ndarray,
    discard_fraction: float = 0.1,
    return_diagnostics: bool = False,
):
    """
    Estima dimensão intrínseca via TwoNN (Facco et al., 2017).

    Parameters
    ----------
    X:
        `[N, D]` array de pontos (N = patches, D = features).
    discard_fraction:
        Fração superior de μ a descartar antes da regressão (default: 0.1).
        Pontos com μ muito grande são outliers numéricos da CDF empírica.
    return_diagnostics:
        Se True, retorna também (mu, F, slope, intercept) para debug.

    Returns
    -------
    d:
        Dimensão intrínseca estimada (escalar float).
    diag (opcional):
        dict com arrays internos para visualização.

    Notas
    -----
    - O método assume nuvem localmente uniforme. Para D >> N o estimador
      é ainda válido (os primeiros 2 NNs capturam estrutura local).
    - Distâncias euclidianas no espaço original 768D.
    """
    N = X.shape[0]
    if N < 3:
        return (np.nan, {}) if return_diagnostics else np.nan

    # Distâncias pairwise — para N=1024 e D=768 cabe em RAM facilmente
    # ||a - b||² = ||a||² + ||b||² - 2 a·b
    norms_sq = (X * X).sum(axis=1)                       # [N]
    sq_dists = norms_sq[:, None] + norms_sq[None, :] - 2.0 * (X @ X.T)
    sq_dists = np.maximum(sq_dists, 0.0)                  # numérico

    # NN1 e NN2 por linha (excluindo o próprio ponto = diagonal)
    np.fill_diagonal(sq_dists, np.inf)

    # Pega os 2 menores por linha via argpartition (O(N²) — ok aqui)
    # partition seleciona os 2 menores nas primeiras 2 posições
    part = np.argpartition(sq_dists, kth=1, axis=1)[:, :2]   # [N, 2]
    r_sq = np.take_along_axis(sq_dists, part, axis=1)         # [N, 2]
    # ordena as 2 colunas (menor = NN1)
    r_sq.sort(axis=1)

    r1 = np.sqrt(r_sq[:, 0])
    r2 = np.sqrt(r_sq[:, 1])

    # Filtra empates e zeros (μ deve ser estritamente > 1)
    valid = (r1 > 0) & (r2 > r1)
    if valid.sum() < 5:
        return (np.nan, {}) if return_diagnostics else np.nan

    mu = r2[valid] / r1[valid]
    mu_sorted = np.sort(mu)

    # CDF empírica
    n_eff = len(mu_sorted)
    F = np.arange(1, n_eff + 1) / (n_eff + 1)  # plotting position

    # Descarta os top `discard_fraction` (μ grandes = cauda ruidosa)
    keep = int(n_eff * (1.0 - discard_fraction))
    mu_fit = mu_sorted[:keep]
    F_fit = F[:keep]

    # Regressão linear: y = d * x onde
    #   y = -log(1 - F)   (intercept forçado a 0 no modelo teórico)
    #   x = log(μ)
    x = np.log(mu_fit)
    y = -np.log(1.0 - F_fit)

    # Mínimos quadrados sem intercepto (modelo F = 1 - μ^(-d))
    slope = float((x @ y) / (x @ x))

    if return_diagnostics:
        return slope, {
            "mu": mu_sorted,
            "F": F,
            "x": x,
            "y": y,
            "slope": slope,
            "n_valid": int(valid.sum()),
        }
    return slope


# ---------------------------------------------------------------------------
# Pipeline de análise
# ---------------------------------------------------------------------------

def load_visual_feats(
    folder: str,
    n_images: int,
) -> Tuple[np.ndarray, int]:
    """
    Carrega `n_images` matrizes de patches dos shards.

    Returns
    -------
    feats:
        Array `[n_images, P, 768]` float32 (P depende do conjunto).
    total_available:
        Total de imagens disponíveis no diretório.
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum shard em {folder}")

    total_available = 0
    chunks = []
    collected = 0

    for f in files:
        if collected >= n_images:
            break
        try:
            with h5py.File(f, "r") as h5:
                vf = h5["visual_feats"]
                total_available += vf.shape[0]
                take = min(vf.shape[0], n_images - collected)
                chunks.append(vf[:take].astype(np.float32))
                collected += take
        except Exception as e:
            print(f"  [WARN] Falha ao ler {f}: {e}")

    if not chunks:
        raise RuntimeError(f"Nenhum visual_feats válido em {folder}")

    feats = np.concatenate(chunks, axis=0)[:n_images]
    return feats, total_available


def compute_id_per_image(
    feats: np.ndarray,
    discard_fraction: float = 0.1,
    progress: bool = True,
) -> np.ndarray:
    """
    Aplica TwoNN em cada imagem independentemente.

    Parameters
    ----------
    feats:
        `[N_images, P, D]` — N_images matrizes de P patches em D dimensões.

    Returns
    -------
    ids:
        `[N_images]` array de dimensões intrínsecas.
    """
    n = feats.shape[0]
    ids = np.full(n, np.nan, dtype=np.float64)

    for i in range(n):
        ids[i] = twonn_id(feats[i], discard_fraction=discard_fraction)
        if progress and (i + 1) % max(1, n // 20) == 0:
            print(f"    {i+1}/{n}  ID atual={ids[i]:.3f}  μ_id={np.nanmean(ids[:i+1]):.3f}")

    return ids


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_id_distributions(
    ids_a: np.ndarray,
    ids_b: np.ndarray,
    label_a: str,
    label_b: str,
    save_path: str,
) -> None:
    """
    Histograma + boxplot + scatter pareado das IDs.
    """
    valid_a = ids_a[~np.isnan(ids_a)]
    valid_b = ids_b[~np.isnan(ids_b)]
    paired_mask = ~np.isnan(ids_a) & ~np.isnan(ids_b)
    paired_a = ids_a[paired_mask]
    paired_b = ids_b[paired_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Histograma sobreposto
    bins = np.linspace(
        min(valid_a.min(), valid_b.min()),
        max(valid_a.max(), valid_b.max()),
        40,
    )
    axes[0].hist(valid_a, bins=bins, alpha=0.6, label=label_a, color="#4A90D9")
    axes[0].hist(valid_b, bins=bins, alpha=0.6, label=label_b, color="#FF6B35")
    axes[0].axvline(valid_a.mean(), color="#4A90D9", linestyle="--", lw=2,
                    label=f"{label_a} μ={valid_a.mean():.2f}")
    axes[0].axvline(valid_b.mean(), color="#FF6B35", linestyle="--", lw=2,
                    label=f"{label_b} μ={valid_b.mean():.2f}")
    axes[0].set_xlabel("Intrinsic Dimension (TwoNN)")
    axes[0].set_ylabel("Image count")
    axes[0].set_title("ID Distribution per Image")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # Boxplot
    bp = axes[1].boxplot(
        [valid_a, valid_b],
        labels=[label_a, label_b],
        patch_artist=True,
        showmeans=True,
    )
    for patch, color in zip(bp["boxes"], ["#4A90D9", "#FF6B35"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[1].set_ylabel("Intrinsic Dimension")
    axes[1].set_title("Comparative Boxplot")
    axes[1].grid(alpha=0.3, axis="y")

    # Scatter pareado (imagem por imagem)
    if len(paired_a) > 0:
        axes[2].scatter(paired_a, paired_b, s=10, alpha=0.4, color="#7B5BC9")
        lim_min = min(paired_a.min(), paired_b.min())
        lim_max = max(paired_a.max(), paired_b.max())
        axes[2].plot([lim_min, lim_max], [lim_min, lim_max], "r--", lw=1, label="y = x")
        axes[2].set_xlabel(f"ID {label_a}")
        axes[2].set_ylabel(f"ID {label_b}")
        delta = (paired_b - paired_a).mean()
        axes[2].set_title(
            f"Paired ID per image\nAvg Δ ({label_b} − {label_a}) = {delta:+.3f}"
        )
        axes[2].legend()
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_analysis(
    folder_a: str,
    folder_b: str,
    label_a: str,
    label_b: str,
    n_images: int,
    output_dir: str,
    discard_fraction: float = 0.1,
    upsampled: str = "a",
) -> dict:
    """
    Executa análise de ID local em N imagens de cada conjunto.

    `upsampled` indica qual dos dois conjuntos passou por upsampling ("a" ou "b").
    É usado apenas na interpretação automática: o Δ reportado permanece sempre
    ID(b) − ID(a), independentemente desse parâmetro.
    """
    if upsampled not in ("a", "b"):
        raise ValueError(f"upsampled deve ser 'a' ou 'b', recebido: {upsampled!r}")
    print(f"\n{'═' * 70}")
    print(f"  LOCAL ID: {label_a}  vs  {label_b}")
    print(f"{'═' * 70}")

    print(f"\n[1/3] Loading {n_images} patch matrices from each set...")
    feats_a, total_a = load_visual_feats(folder_a, n_images)
    feats_b, total_b = load_visual_feats(folder_b, n_images)
    print(f"  {label_a}: {feats_a.shape}  (P={feats_a.shape[1]} patches, total={total_a:,})")
    print(f"  {label_b}: {feats_b.shape}  (P={feats_b.shape[1]} patches, total={total_b:,})")

    if feats_a.shape[0] != feats_b.shape[0]:
        n = min(feats_a.shape[0], feats_b.shape[0])
        print(f"  [WARN] Truncando para {n} imagens")
        feats_a = feats_a[:n]
        feats_b = feats_b[:n]

    print(f"\n[2/3] Estimating ID via TwoNN for {label_a} ({feats_a.shape[1]} patches/image)...")
    ids_a = compute_id_per_image(feats_a, discard_fraction=discard_fraction)

    print(f"\n      Estimating ID via TwoNN for {label_b} ({feats_b.shape[1]} patches/image)...")
    ids_b = compute_id_per_image(feats_b, discard_fraction=discard_fraction)

    print(f"\n[3/3] Statistics and visualization...")
    stats_a = _id_stats(ids_a)
    stats_b = _id_stats(ids_b)

    paired_mask = ~np.isnan(ids_a) & ~np.isnan(ids_b)
    paired_delta = (ids_b - ids_a)[paired_mask]

    report = {
        "label_a": label_a,
        "label_b": label_b,
        "folder_a": folder_a,
        "folder_b": folder_b,
        "n_images": int(feats_a.shape[0]),
        "patches_a": int(feats_a.shape[1]),
        "patches_b": int(feats_b.shape[1]),
        "id_stats_a": stats_a,
        "id_stats_b": stats_b,
        "paired": {
            "n_valid": int(paired_mask.sum()),
            "delta_mean": float(paired_delta.mean()) if paired_delta.size else float("nan"),
            "delta_std":  float(paired_delta.std())  if paired_delta.size else float("nan"),
            "delta_median": float(np.median(paired_delta)) if paired_delta.size else float("nan"),
            "frac_b_higher": float((paired_delta > 0).mean()) if paired_delta.size else float("nan"),
        },
    }

    # Print resumido
    print(f"\n  {label_a}  (P={feats_a.shape[1]}): ID μ={stats_a['mean']:.3f} ± {stats_a['std']:.3f}  "
          f"[mediana={stats_a['median']:.3f}]")
    print(f"  {label_b}  (P={feats_b.shape[1]}): ID μ={stats_b['mean']:.3f} ± {stats_b['std']:.3f}  "
          f"[mediana={stats_b['median']:.3f}]")
    if paired_delta.size:
        print(f"  Paired Δ ({label_b}−{label_a}): μ={paired_delta.mean():+.3f}  "
              f"({(paired_delta > 0).mean()*100:.1f}% of images have higher ID in {label_b})")

    # Interpretação automática — efeito DO UPSAMPLING sobre o ID.
    # Δ = ID(b) − ID(a). Se o conjunto upsampled for o `a`, o efeito do
    # upsampling é −Δ; caso contrário é +Δ.
    print(f"\n{'─' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'─' * 70}")
    delta = report["paired"]["delta_mean"]
    up_label, base_label = (label_a, label_b) if upsampled == "a" else (label_b, label_a)
    up_patches = feats_a.shape[1] if upsampled == "a" else feats_b.shape[1]
    base_patches = feats_b.shape[1] if upsampled == "a" else feats_a.shape[1]
    delta_up = -delta if upsampled == "a" else delta

    if abs(delta_up) < 1.0:
        verdict = (f"ID virtually EQUAL → the {up_patches} patches of {up_label} "
                   f"lie on the SAME submanifold as the {base_patches} patches of "
                   f"{base_label}; upsampling is geometrically redundant")
    elif delta_up > 5.0:
        verdict = (f"ID much HIGHER with upsampling ({delta_up:+.1f}) → upsampling added "
                   f"real complexity (microtextures / high frequency)")
    elif delta_up > 0:
        verdict = (f"ID MODERATELY higher with upsampling ({delta_up:+.1f}) → upsampling "
                   f"refined local structure but preserves smooth topology")
    else:
        verdict = (f"ID LOWER with upsampling ({delta_up:+.1f}) → upsampling SMOOTHED the "
                   f"patch cloud (implicit anti-aliasing / redundant patches)")
    print(f"  Δ ({label_b}−{label_a}) = {delta:+.3f}   "
          f"|   upsampling effect on ID = {delta_up:+.3f}")
    print(f"  →  {verdict}")

    report["upsampled_label"] = up_label
    report["paired"]["delta_upsampling"] = float(delta_up)

    # Salva
    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"id_{label_a}_vs_{label_b}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  [report] {json_path}")

    # Salva arrays brutos pra reanalise depois
    np_path = os.path.join(output_dir, f"id_{label_a}_vs_{label_b}.npz")
    np.savez(np_path, ids_a=ids_a, ids_b=ids_b)
    print(f"  [arrays] {np_path}")

    png_path = os.path.join(output_dir, f"id_{label_a}_vs_{label_b}.png")
    plot_id_distributions(ids_a, ids_b, label_a, label_b, png_path)

    return report


def _id_stats(ids: np.ndarray) -> dict:
    valid = ids[~np.isnan(ids)]
    if valid.size == 0:
        return {k: float("nan") for k in ("n_valid", "mean", "std", "median", "min", "max",
                                          "q25", "q75")}
    return {
        "n_valid": int(valid.size),
        "mean":    float(valid.mean()),
        "std":     float(valid.std()),
        "median":  float(np.median(valid)),
        "min":     float(valid.min()),
        "max":     float(valid.max()),
        "q25":     float(np.percentile(valid, 25)),
        "q75":     float(np.percentile(valid, 75)),
    }


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ID local TwoNN por imagem (com vs sem AnyUp)")
    parser.add_argument("--n_images", type=int, default=5000,
                        help="Quantas imagens analisar por conjunto (default: 500)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all",
                        help="Split a analisar (default: test)")
    parser.add_argument("--folder_anyup", type=str, default=None,
                        help="Override do diretório com AnyUp")
    parser.add_argument("--folder_noup", type=str, default=None,
                        help="Override do diretório sem AnyUp")
    parser.add_argument("--output_dir", type=str, default="results/id_local",
                        help="Diretório de saída")
    parser.add_argument("--discard", type=float, default=0.1,
                        help="Fração superior de μ a descartar no TwoNN (default: 0.1)")
    args = parser.parse_args()

    defaults = {
        "train": ("F:/COYO/embeds/train_anyup/", "E:/COYO/embeds/train_noup/"),
        "val":   ("G:/coyo/embeds/val_anyup/",   "E:/COYO/embeds/val_noup/"),
        "test":  ("G:/coyo/embeds/test_anyup/",  "E:/COYO/embeds/test_noup/"),
    }

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_reports = {}

    for split in splits_to_run:
        f_anyup, f_noup = defaults[split]
        if args.folder_anyup:
            f_anyup = args.folder_anyup
        if args.folder_noup:
            f_noup = args.folder_noup

        if not os.path.isdir(f_anyup):
            print(f"\n[SKIP] {split}: {f_anyup} does not exist")
            continue
        if not os.path.isdir(f_noup):
            print(f"\n[SKIP] {split}: {f_noup} does not exist")
            continue

        report = run_analysis(
            folder_a=f_anyup,
            folder_b=f_noup,
            label_a=f"{split}_anyup",
            label_b=f"{split}_noup",
            n_images=args.n_images,
            output_dir=args.output_dir,
            discard_fraction=args.discard,
            upsampled="a",          # folder_a = AnyUp
        )
        all_reports[split] = report

    if len(all_reports) > 1:
        print(f"\n\n{'═' * 70}")
        print(f"  FINAL SUMMARY — Local ID per Set")
        print(f"{'═' * 70}")
        print(f"  {'Split':<10} {'N imgs':>8} {'ID (anyup)':>14} {'ID (noup)':>14} "
              f"{'Δ (noup−anyup)':>16} {'% noup higher':>15}")
        print(f"  {'-'*10} {'-'*8} {'-'*14} {'-'*14} {'-'*16} {'-'*15}")
        for split, rep in all_reports.items():
            n = rep["n_images"]
            a = rep["id_stats_a"]["mean"]
            b = rep["id_stats_b"]["mean"]
            d = rep["paired"]["delta_mean"]
            s = rep["paired"]["delta_std"]
            f = rep["paired"]["frac_b_higher"] * 100.0
            print(f"  {split:<10} {n:>8,} {a:>14.3f} {b:>14.3f} "
                  f"{f'{d:+.3f} ± {s:.3f}':>16} {f:>14.1f}%")
