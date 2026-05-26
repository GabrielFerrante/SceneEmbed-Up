"""
analyze_rsa_global.py
---------------------
Representational Similarity Analysis (RSA) sobre embeddings globais
(visual_global / token CLS do DINOv3) usando a biblioteca `rsatoolbox`.

https://rsatoolbox.readthedocs.io/en/stable/

Compara a geometria intra-visual entre dois conjuntos:
  A. Embeddings extraídos COM AnyUp upsampling
  B. Embeddings extraídos SEM upsampling (LR nativo)

Pipeline:
  1. Carrega visual_global de N shards de cada conjunto
  2. Envelopa cada matriz como `rsatoolbox.data.Dataset`
  3. Computa RDMs com `rsatoolbox.rdm.calc_rdm(method=...)`
     Métodos suportados: euclidean | correlation | mahalanobis | crossnobis | poisson
  4. Compara as RDMs com `rsatoolbox.rdm.compare(method=...)`
     Métodos suportados: cosine | corr | cosine_cov | corr_cov | tau-a | rho-a
  5. Salva relatório JSON + visualização PNG (via rsatoolbox.vis quando possível)

Interpretação:
  - sim ≈ 1.00 → upsampling preserva geometria global
  - sim ≈ 0.90 → preservação parcial
  - sim ≤ 0.70 → upsampling reorganiza significativamente o espaço

Uso:
    python embeddings/analyze_rsa_global.py
    python embeddings/analyze_rsa_global.py --n_samples 5000 --split test \
        --rdm_method correlation --compare_method rho-a

Dependência:
    pip install rsatoolbox
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

# rsatoolbox — núcleo da análise
import rsatoolbox
import rsatoolbox.data as rsd
import rsatoolbox.rdm as rsr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io_utils import ensure_dir


# ---------------------------------------------------------------------------
# Carregamento de embeddings
# ---------------------------------------------------------------------------

def load_visual_global(
    folder: str,
    n_samples: int,
) -> Tuple[np.ndarray, int]:
    """
    Carrega `n_samples` embeddings globais dos shards.

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
                vg = h5["visual_global"][:]
                total_available += vg.shape[0]
                take = min(vg.shape[0], n_samples - collected)
                all_emb.append(vg[:take].astype(np.float32))
                collected += take
        except Exception as e:
            print(f"  [WARN] Falha ao ler {f}: {e}")

    if not all_emb:
        raise RuntimeError(f"Nenhum visual_global válido em {folder}")

    embeddings = np.concatenate(all_emb, axis=0)[:n_samples]
    return embeddings, total_available


# ---------------------------------------------------------------------------
# Construção de Dataset e RDM via rsatoolbox
# ---------------------------------------------------------------------------

def build_dataset(
    embeddings: np.ndarray,
    name: str,
) -> rsd.Dataset:
    """
    Envelopa `[N, D]` num `rsatoolbox.data.Dataset`.

    Adiciona descriptors para rastreabilidade: cada observação ganha um índice
    e o dataset recebe um nome (útil para batch processing).
    """
    n = embeddings.shape[0]
    return rsd.Dataset(
        measurements=embeddings,
        descriptors={"name": name},
        obs_descriptors={"index": np.arange(n)},
    )


def compute_rdm(
    dataset: rsd.Dataset,
    method: str = "correlation",
) -> rsr.RDMs:
    """
    Calcula RDM via rsatoolbox.

    Parameters
    ----------
    method:
        - 'correlation' → 1 - Pearson (padrão histórico em RSA)
        - 'euclidean'   → distância euclidiana normalizada por canais
        - 'mahalanobis' → requer noise precision matrix (não usado aqui)
        - 'crossnobis'  → distância unbiased entre runs (não aplicável)
        - 'poisson'     → KL-divergence simétrica (não aplicável)
    """
    return rsr.calc_rdm(dataset, method=method)


def compare_rdms(
    rdm_a: rsr.RDMs,
    rdm_b: rsr.RDMs,
    method: str = "rho-a",
) -> float:
    """
    Compara duas RDMs via rsatoolbox.rdm.compare.

    `rsatoolbox.rdm.compare` retorna matriz `[n_rdms_a, n_rdms_b]` com
    similaridades pairwise. Como temos 1 RDM por lado, extraímos `[0, 0]`.

    Parameters
    ----------
    method:
        - 'cosine'     → cosine similarity
        - 'corr'       → Pearson (correlação de Pearson entre RDMs)
        - 'cosine_cov' → whitened cosine
        - 'corr_cov'   → whitened correlation
        - 'tau-a'      → Kendall tau-a
        - 'rho-a'      → Spearman rho-a (recomendado pela doc)
    """
    sim_matrix = rsr.compare(rdm_a, rdm_b, method=method)
    return float(sim_matrix[0, 0])


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def _rdm_to_square(rdm: rsr.RDMs) -> np.ndarray:
    """Converte RDM rsatoolbox em matriz quadrada N×N para plotagem."""
    return rdm.get_matrices()[0]   # primeira (e única) RDM


def plot_results(
    rdm_a: rsr.RDMs,
    rdm_b: rsr.RDMs,
    similarities: dict,
    label_a: str,
    label_b: str,
    save_path: str,
    subsample: int = 500,
) -> None:
    """
    Figura com 3 painéis: RDM_A | RDM_B | scatter dissimilaridades.
    """
    mat_a = _rdm_to_square(rdm_a)
    mat_b = _rdm_to_square(rdm_b)
    n = mat_a.shape[0]

    if n > subsample:
        idx = np.random.default_rng(42).choice(n, subsample, replace=False)
        idx.sort()
        mat_a_vis = mat_a[np.ix_(idx, idx)]
        mat_b_vis = mat_b[np.ix_(idx, idx)]
    else:
        mat_a_vis = mat_a
        mat_b_vis = mat_b

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(mat_a_vis.max(), mat_b_vis.max())

    im0 = axes[0].imshow(mat_a_vis, cmap="viridis", vmin=0, vmax=vmax)
    axes[0].set_title(f"RDM {label_a}\n({mat_a_vis.shape[0]}x{mat_a_vis.shape[0]})")
    axes[0].set_xlabel("image j")
    axes[0].set_ylabel("image i")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(mat_b_vis, cmap="viridis", vmin=0, vmax=vmax)
    axes[1].set_title(f"RDM {label_b}\n({mat_b_vis.shape[0]}x{mat_b_vis.shape[0]})")
    axes[1].set_xlabel("image j")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # Scatter sobre triângulo superior das RDMs visualizadas
    iu = np.triu_indices_from(mat_a_vis, k=1)
    va, vb = mat_a_vis[iu], mat_b_vis[iu]
    if len(va) > 20_000:
        s_idx = np.random.default_rng(42).choice(len(va), 20_000, replace=False)
        va, vb = va[s_idx], vb[s_idx]
    axes[2].scatter(va, vb, s=2, alpha=0.3)
    axes[2].plot([0, vmax], [0, vmax], "r--", lw=1, label="y = x")
    axes[2].set_xlabel(f"dissimilarity ({label_a})")
    axes[2].set_ylabel(f"dissimilarity ({label_b})")

    sim_strs = [f"{k}: {v:.4f}" for k, v in similarities.items()]
    axes[2].set_title("RDM similarity\n" + "  |  ".join(sim_strs))
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
    rdm_method: str = "correlation",
    compare_methods: Tuple[str, ...] = ("cosine", "corr", "rho-a", "tau-a"),
) -> dict:
    """Executa RSA via rsatoolbox entre dois conjuntos de embeddings globais."""
    print(f"\n{'=' * 70}")
    print(f"  RSA (rsatoolbox): {label_a}  vs  {label_b}")
    print(f"  rdm_method={rdm_method}   compare_methods={list(compare_methods)}")
    print(f"{'=' * 70}")

    print(f"\n[1/4] Loading {n_samples} embeddings from each set...")
    emb_a, total_a = load_visual_global(folder_a, n_samples)
    emb_b, total_b = load_visual_global(folder_b, n_samples)
    print(f"  {label_a}: {emb_a.shape}  (total available: {total_a:,})")
    print(f"  {label_b}: {emb_b.shape}  (total available: {total_b:,})")

    if emb_a.shape[0] != emb_b.shape[0]:
        n = min(emb_a.shape[0], emb_b.shape[0])
        print(f"  [WARN] Truncating to {n} samples to match sets")
        emb_a, emb_b = emb_a[:n], emb_b[:n]

    print(f"\n[2/4] Building rsatoolbox Datasets...")
    ds_a = build_dataset(emb_a, name=label_a)
    ds_b = build_dataset(emb_b, name=label_b)
    print(f"  {label_a}: {ds_a}")
    print(f"  {label_b}: {ds_b}")

    print(f"\n[3/4] Computing RDMs (method={rdm_method})...")
    rdm_a = compute_rdm(ds_a, method=rdm_method)
    rdm_b = compute_rdm(ds_b, method=rdm_method)
    mat_a = _rdm_to_square(rdm_a)
    mat_b = _rdm_to_square(rdm_b)
    print(f"  RDM {label_a}: shape={mat_a.shape}  mean={mat_a.mean():.4f}  std={mat_a.std():.4f}")
    print(f"  RDM {label_b}: shape={mat_b.shape}  mean={mat_b.mean():.4f}  std={mat_b.std():.4f}")

    print(f"\n[4/4] Comparing RDMs with rsatoolbox.rdm.compare...")
    similarities: dict = {}
    for cmp in compare_methods:
        try:
            sim = compare_rdms(rdm_a, rdm_b, method=cmp)
            similarities[cmp] = sim
            print(f"  {cmp:<12} = {sim:.6f}")
        except Exception as e:
            print(f"  {cmp:<12} = FAILED ({e})")
            similarities[cmp] = float("nan")

    # ── Interpretação ────────────────────────────────────────────────────
    print(f"\n{'-' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'-' * 70}")
    ref = similarities.get("rho-a") or similarities.get("corr") or next(iter(similarities.values()))
    if ref >= 0.95:
        verdict = "Geometry highly PRESERVED -- upsampling is semantically safe"
    elif ref >= 0.85:
        verdict = "Geometry well preserved -- small local reorganizations"
    elif ref >= 0.70:
        verdict = "MODERATE preservation -- upsampling causes visible changes"
    else:
        verdict = "SIGNIFICANT reorganization -- upsampling alters the semantics"
    print(f"  Reference similarity = {ref:.4f}  ->  {verdict}")

    # ── Salva relatório ─────────────────────────────────────────────────
    report = {
        "label_a": label_a,
        "label_b": label_b,
        "folder_a": folder_a,
        "folder_b": folder_b,
        "n_samples": int(emb_a.shape[0]),
        "embedding_dim": int(emb_a.shape[1]),
        "rdm_method": rdm_method,
        "rdm_stats_a": {
            "mean": float(mat_a.mean()),
            "std":  float(mat_a.std()),
            "min":  float(mat_a.min()),
            "max":  float(mat_a.max()),
        },
        "rdm_stats_b": {
            "mean": float(mat_b.mean()),
            "std":  float(mat_b.std()),
            "min":  float(mat_b.min()),
            "max":  float(mat_b.max()),
        },
        "similarities": similarities,
        "rsatoolbox_version": getattr(rsatoolbox, "__version__", "unknown"),
    }

    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"rsa_{label_a}_vs_{label_b}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  [report] saved to {json_path}")

    png_path = os.path.join(output_dir, f"rsa_{label_a}_vs_{label_b}.png")
    plot_results(rdm_a, rdm_b, similarities, label_a, label_b, png_path)

    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="RSA global via rsatoolbox: anyup vs noup"
    )
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Samples per set for the RDM (default: 5000)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--folder_anyup", type=str, default=None,
                        help="Override directory of AnyUp set")
    parser.add_argument("--folder_noup", type=str, default=None,
                        help="Override directory of no-up set")
    parser.add_argument("--output_dir", type=str, default="results/rsa_global")
    parser.add_argument("--rdm_method", type=str, default="correlation",
                        choices=["correlation", "euclidean"],
                        help="rsatoolbox method for calc_rdm (default: correlation)")
    parser.add_argument("--compare_methods", nargs="+",
                        default=["cosine", "corr", "rho-a", "tau-a"],
                        help="rsatoolbox compare methods (default: cosine corr rho-a tau-a)")
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
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            rdm_method=args.rdm_method,
            compare_methods=tuple(args.compare_methods),
        )
        all_reports[split] = report

    if len(all_reports) > 1:
        print(f"\n\n{'=' * 70}")
        print(f"  FINAL SUMMARY (rsatoolbox)")
        print(f"{'=' * 70}")
        first = next(iter(all_reports.values()))
        cmp_keys = list(first["similarities"].keys())
        header = f"  {'Split':<10} {'N':>8} " + "".join(f"{k:>12}" for k in cmp_keys)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for split, rep in all_reports.items():
            n = rep["n_samples"]
            row = f"  {split:<10} {n:>8,} " + "".join(
                f"{rep['similarities'].get(k, float('nan')):>12.4f}" for k in cmp_keys
            )
            print(row)
