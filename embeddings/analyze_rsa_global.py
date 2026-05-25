"""
analyze_rsa_global.py
---------------------
Representational Similarity Analysis (RSA) sobre os embeddings globais
(visual_global / token CLS do DINOv3) dos shards COYO.

Compara a geometria intra-visual entre dois conjuntos:
  A. Embeddings extraídos COM AnyUp upsampling
  B. Embeddings extraídos SEM upsampling (LR nativo)

Pipeline:
  1. Carrega visual_global de N shards de cada conjunto
  2. Constrói RDM (Representational Dissimilarity Matrix) [N_samples, N_samples]
     usando 1 − cosine_similarity
  3. Calcula correlações entre as duas RDMs:
        - Spearman (rank-based, padrão em neurociência computacional)
        - Pearson  (linear)
        - Kendall tau (opcional, mais robusto a outliers)
  4. Gera estatísticas de magnitude (norm) e dispersão dos embeddings
  5. Salva relatório JSON + visualizações PNG

Interpretação:
  - corr ≈ 1.00 → upsampling preserva geometria global (semanticamente seguro)
  - corr ≈ 0.90 → preservação parcial; mudanças localizadas
  - corr ≤ 0.70 → upsampling reorganiza significativamente o espaço

Uso:
    python embeddings/analyze_rsa_global.py
    python embeddings/analyze_rsa_global.py --n_samples 5000 --split test
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
# Carregamento de embeddings
# ---------------------------------------------------------------------------

def load_visual_global(
    folder: str,
    n_samples: int,
    seed: int = 42,
) -> Tuple[np.ndarray, int]:
    """
    Carrega `n_samples` embeddings globais dos shards do diretório.

    Lê sequencialmente os shards até atingir n_samples. Retorna também o total
    de amostras disponíveis (útil para diagnóstico).

    Returns
    -------
    embeddings:
        Array `[n_samples, 768]` float32.
    total_available:
        Total de amostras disponíveis no diretório.
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum shard em {folder}")

    total_available = 0
    all_emb: list[np.ndarray] = []
    collected = 0

    for f in files:
        if collected >= n_samples:
            break
        try:
            with h5py.File(f, "r") as h5:
                if "visual_global" not in h5:
                    print(f"  [WARN] '{os.path.basename(f)}' sem visual_global — pulado")
                    continue
                vg = h5["visual_global"][:]   # [N, 768]
                total_available += vg.shape[0]
                take = min(vg.shape[0], n_samples - collected)
                all_emb.append(vg[:take].astype(np.float32))
                collected += take
        except Exception as e:
            print(f"  [WARN] Falha ao ler {f}: {e}")

    if not all_emb:
        raise RuntimeError(f"Nenhum visual_global válido encontrado em {folder}")

    embeddings = np.concatenate(all_emb, axis=0)[:n_samples]

    # Conta o total real (pode passar de n_samples)
    for f in files:
        try:
            with h5py.File(f, "r") as h5:
                if "visual_global" in h5 and f not in (g[0] for g in all_emb):
                    pass  # já contado acima
        except Exception:
            pass

    return embeddings, total_available


# ---------------------------------------------------------------------------
# RDM construction
# ---------------------------------------------------------------------------

def build_rdm_cosine(embeddings: np.ndarray) -> np.ndarray:
    """
    Constrói Representational Dissimilarity Matrix usando 1 − cosine.

    RDM[i, j] = 1 − (e_i · e_j) / (||e_i|| ||e_j||)

    Parameters
    ----------
    embeddings:
        `[N, D]` array.

    Returns
    -------
    rdm:
        `[N, N]` matriz simétrica com zeros na diagonal, valores em [0, 2].
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    normed = embeddings / norms
    sim = normed @ normed.T               # cosine similarity
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Extrai o triângulo superior (sem diagonal) como vetor 1D."""
    iu = np.triu_indices_from(matrix, k=1)
    return matrix[iu]


# ---------------------------------------------------------------------------
# Correlações entre RDMs
# ---------------------------------------------------------------------------

def rsa_correlate(rdm_a: np.ndarray, rdm_b: np.ndarray) -> dict:
    """
    Calcula correlações Pearson/Spearman/Kendall entre dois RDMs.

    Usa apenas o triângulo superior para evitar correlação inflada por
    simetria/diagonal.

    Returns
    -------
    dict com 'pearson', 'spearman' e (se scipy disponível) 'kendall'.
    """
    va = upper_triangle(rdm_a)
    vb = upper_triangle(rdm_b)

    # Pearson via numpy (rápido)
    pearson = float(np.corrcoef(va, vb)[0, 1])

    # Spearman: rank manual evita dependência opcional do scipy
    rank_a = _rank_data(va)
    rank_b = _rank_data(vb)
    spearman = float(np.corrcoef(rank_a, rank_b)[0, 1])

    out = {"pearson": pearson, "spearman": spearman, "n_pairs": int(len(va))}

    try:
        from scipy.stats import kendalltau
        # Kendall tau é O(N²) — só rodamos em subset se N for grande
        if len(va) > 200_000:
            idx = np.random.default_rng(42).choice(len(va), 200_000, replace=False)
            tau, _ = kendalltau(va[idx], vb[idx])
            out["kendall_subset"] = float(tau)
        else:
            tau, _ = kendalltau(va, vb)
            out["kendall"] = float(tau)
    except ImportError:
        pass

    return out


def _rank_data(x: np.ndarray) -> np.ndarray:
    """Retorna ranks (1-based) lidando com empates por average."""
    order = np.argsort(x)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1)
    # Empates: average rank
    sorted_x = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2
            ranks[order[i:j + 1]] = avg
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Estatísticas geométricas
# ---------------------------------------------------------------------------

def geometric_stats(embeddings: np.ndarray) -> dict:
    """
    Estatísticas sobre magnitude e dispersão dos embeddings.

    Returns
    -------
    dict com norm_mean/std/min/max, centroid_norm, mean_pairwise_cosine.
    """
    norms = np.linalg.norm(embeddings, axis=1)
    centroid = embeddings.mean(axis=0)

    normed = embeddings / np.clip(norms, 1e-12, None)[:, None]
    # Cosine pairwise medio (sample pra evitar O(N²) em RAM)
    n = embeddings.shape[0]
    if n > 2000:
        idx = np.random.default_rng(42).choice(n, 2000, replace=False)
        sample = normed[idx]
    else:
        sample = normed
    sim = sample @ sample.T
    triu = sim[np.triu_indices_from(sim, k=1)]

    return {
        "n_samples":    int(n),
        "dim":          int(embeddings.shape[1]),
        "norm_mean":    float(norms.mean()),
        "norm_std":     float(norms.std()),
        "norm_min":     float(norms.min()),
        "norm_max":     float(norms.max()),
        "centroid_norm":     float(np.linalg.norm(centroid)),
        "mean_pairwise_cos": float(triu.mean()),
        "std_pairwise_cos":  float(triu.std()),
    }


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_results(
    rdm_a: np.ndarray,
    rdm_b: np.ndarray,
    corr: dict,
    label_a: str,
    label_b: str,
    save_path: str,
    subsample: int = 500,
) -> None:
    """
    Gera figura com:
      - RDM A
      - RDM B
      - Scatter plot rank-rank entre A e B
    """
    n = rdm_a.shape[0]
    if n > subsample:
        idx = np.random.default_rng(42).choice(n, subsample, replace=False)
        idx.sort()
        rdm_a_vis = rdm_a[np.ix_(idx, idx)]
        rdm_b_vis = rdm_b[np.ix_(idx, idx)]
    else:
        rdm_a_vis = rdm_a
        rdm_b_vis = rdm_b

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    vmax = max(rdm_a_vis.max(), rdm_b_vis.max())

    im0 = axes[0].imshow(rdm_a_vis, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title(f"RDM {label_a}\n({rdm_a_vis.shape[0]}×{rdm_a_vis.shape[0]})")
    axes[0].set_xlabel("image j")
    axes[0].set_ylabel("image i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(rdm_b_vis, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title(f"RDM {label_b}\n({rdm_b_vis.shape[0]}×{rdm_b_vis.shape[0]})")
    axes[1].set_xlabel("image j")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Scatter plot dos triângulos superiores (subset)
    va = upper_triangle(rdm_a_vis)
    vb = upper_triangle(rdm_b_vis)
    if len(va) > 20_000:
        s_idx = np.random.default_rng(42).choice(len(va), 20_000, replace=False)
        va, vb = va[s_idx], vb[s_idx]
    axes[2].scatter(va, vb, s=2, alpha=0.3)
    axes[2].plot([0, vmax], [0, vmax], "r--", lw=1, label="y = x")
    axes[2].set_xlabel(f"dissimilarity ({label_a})")
    axes[2].set_ylabel(f"dissimilarity ({label_b})")
    axes[2].set_title(
        f"Pearson r = {corr['pearson']:.4f}\n"
        f"Spearman ρ = {corr['spearman']:.4f}"
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
    n_samples: int,
    output_dir: str,
) -> dict:
    """Executa RSA completa entre dois conjuntos de embeddings globais."""
    print(f"\n{'═' * 70}")
    print(f"  RSA: {label_a}  vs  {label_b}")
    print(f"{'═' * 70}")

    print(f"\n[1/4] Loading {n_samples} embeddings from each set...")
    emb_a, total_a = load_visual_global(folder_a, n_samples)
    emb_b, total_b = load_visual_global(folder_b, n_samples)
    print(f"  {label_a}: {emb_a.shape}  (total available: {total_a:,})")
    print(f"  {label_b}: {emb_b.shape}  (total available: {total_b:,})")

    if emb_a.shape[0] != emb_b.shape[0]:
        n = min(emb_a.shape[0], emb_b.shape[0])
        print(f"  [WARN] Truncando para {n} amostras (igualar conjuntos)")
        emb_a = emb_a[:n]
        emb_b = emb_b[:n]

    print(f"\n[2/4] Geometric statistics of embeddings...")
    stats_a = geometric_stats(emb_a)
    stats_b = geometric_stats(emb_b)
    print(f"  {label_a}: norm μ={stats_a['norm_mean']:.4f}±{stats_a['norm_std']:.4f}  "
          f"cos μ={stats_a['mean_pairwise_cos']:.4f}")
    print(f"  {label_b}: norm μ={stats_b['norm_mean']:.4f}±{stats_b['norm_std']:.4f}  "
          f"cos μ={stats_b['mean_pairwise_cos']:.4f}")

    print(f"\n[3/4] Construindo RDMs [{emb_a.shape[0]}×{emb_a.shape[0]}]...")
    rdm_a = build_rdm_cosine(emb_a)
    rdm_b = build_rdm_cosine(emb_b)
    print(f"  RDM {label_a}: μ={rdm_a.mean():.4f} std={rdm_a.std():.4f}")
    print(f"  RDM {label_b}: μ={rdm_b.mean():.4f} std={rdm_b.std():.4f}")

    print(f"\n[4/4] Computing correlation between RDMs...")
    corr = rsa_correlate(rdm_a, rdm_b)

    print(f"\n{'─' * 70}")
    print(f"  RSA CORRELATIONS  ({corr['n_pairs']:,} dissimilarity pairs)")
    print(f"{'─' * 70}")
    print(f"  Pearson r  = {corr['pearson']:.6f}")
    print(f"  Spearman ρ = {corr['spearman']:.6f}")
    if "kendall" in corr:
        print(f"  Kendall τ  = {corr['kendall']:.6f}")
    elif "kendall_subset" in corr:
        print(f"  Kendall τ  = {corr['kendall_subset']:.6f}  (subset 200k pares)")

    # Diagnóstico se algum valor for NaN
    if not np.isfinite(corr["pearson"]) or not np.isfinite(corr["spearman"]):
        print(f"\n  [WARN] NaN correlation detected — checking RDMs:")
        va = upper_triangle(rdm_a)
        vb = upper_triangle(rdm_b)
        print(f"    var(RDM_{label_a}) = {va.var():.6e}")
        print(f"    var(RDM_{label_b}) = {vb.var():.6e}")
        print(f"    nan in RDM_{label_a}: {np.isnan(va).any()}  inf: {np.isinf(va).any()}")
        print(f"    nan in RDM_{label_b}: {np.isnan(vb).any()}  inf: {np.isinf(vb).any()}")

    # Salva relatório
    report = {
        "label_a": label_a,
        "label_b": label_b,
        "folder_a": folder_a,
        "folder_b": folder_b,
        "n_samples": int(emb_a.shape[0]),
        "embedding_dim": int(emb_a.shape[1]),
        "stats_a": stats_a,
        "stats_b": stats_b,
        "rdm_stats_a": {
            "mean": float(rdm_a.mean()),
            "std":  float(rdm_a.std()),
            "min":  float(rdm_a.min()),
            "max":  float(rdm_a.max()),
        },
        "rdm_stats_b": {
            "mean": float(rdm_b.mean()),
            "std":  float(rdm_b.std()),
            "min":  float(rdm_b.min()),
            "max":  float(rdm_b.max()),
        },
        "correlation": corr,
    }

    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"rsa_{label_a}_vs_{label_b}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  [report] {json_path}")

    png_path = os.path.join(output_dir, f"rsa_{label_a}_vs_{label_b}.png")
    plot_results(rdm_a, rdm_b, corr, label_a, label_b, png_path)

    # Interpretação automática
    print(f"\n{'─' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'─' * 70}")
    rho = corr["spearman"]
    if rho >= 0.95:
        verdict = "GLOBAL geometry highly preserved — upsampling is semantically safe"
    elif rho >= 0.85:
        verdict = "GLOBAL geometry well preserved — minor local reorganizations"
    elif rho >= 0.70:
        verdict = "MODERATE preservation — upsampling causes visible changes"
    else:
        verdict = "SIGNIFICANT space reorganization — upsampling alters semantics"
    print(f"  Spearman ρ = {rho:.4f}  →  {verdict}")

    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RSA global entre embeddings com/sem AnyUp")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Amostras por conjunto para o RDM (default: 5000)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test",
                        help="Split a analisar (default: test)")
    parser.add_argument("--folder_anyup", type=str, default=None,
                        help="Override do diretório do conjunto com AnyUp")
    parser.add_argument("--folder_noup", type=str, default=None,
                        help="Override do diretório do conjunto sem AnyUp")
    parser.add_argument("--output_dir", type=str, default="results/rsa_global",
                        help="Diretório para JSON + PNGs")
    args = parser.parse_args()

    # Paths padrão por split (consistentes com os outros scripts)
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
            n_samples=args.n_samples,
            output_dir=args.output_dir,
        )
        all_reports[split] = report

    # Sumário final
    if len(all_reports) > 1:
        print(f"\n\n{'═' * 70}")
        print(f"  FINAL SUMMARY")
        print(f"{'═' * 70}")
        print(f"  {'Split':<10} {'N':>8} {'Pearson r':>12} {'Spearman ρ':>14}")
        print(f"  {'-'*10} {'-'*8} {'-'*12} {'-'*14}")
        for split, rep in all_reports.items():
            n = rep["n_samples"]
            r = rep["correlation"]["pearson"]
            rho = rep["correlation"]["spearman"]
            print(f"  {split:<10} {n:>8,} {r:>12.4f} {rho:>14.4f}")
