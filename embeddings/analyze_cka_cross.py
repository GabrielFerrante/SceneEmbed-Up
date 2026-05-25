"""
analyze_cka_cross.py
--------------------
Centered Kernel Alignment (CKA) entre embeddings visuais e textuais.

CKA mede aderência geométrica entre dois espaços de representação MESMO
quando suas dimensões nativas são diferentes (768 vs 4096). Olha para as
relações entre N amostras, não para as dimensões de cada amostra.

Pipeline:
  A. Constroi Gram matrices K (visual) e L (textual), ambas [N, N]
  B. Centraliza ambas: K' = H K H, com H = I - (1/N) 11ᵀ
  C. Calcula HSIC(K', L') = trace(K' L') / (N-1)²
  D. Normaliza: CKA = HSIC(K,L) / sqrt(HSIC(K,K) · HSIC(L,L))

Interpretação:
  - 1.0 → geometrias idênticas (mesmo arranjo relativo das N amostras)
  - 0.0 → arranjos totalmente independentes
  - >0.5 → forte alinhamento (típico em modelos bem treinados)

Comparações executadas (texto é único — Qwen é determinístico):
  1. Visual_global (anyup) ↔ Text                  — alinhamento bruto com upsampling
  2. Visual_global (noup)  ↔ Text                  — alinhamento bruto sem upsampling
  3. Visual_global (anyup) ↔ Visual_global (noup)  — preservação geométrica do CLS

Uso:
    python embeddings/analyze_cka_cross.py
    python embeddings/analyze_cka_cross.py --n_samples 5000 --kernel rbf
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
# Kernels
# ---------------------------------------------------------------------------

def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Gram matrix com produto escalar: K = X Xᵀ"""
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: float | None = None) -> np.ndarray:
    """
    Gram matrix com kernel RBF (gaussiano).

    sigma=None → bandwidth via mediana das distâncias (heurística padrão)
    """
    norms = (X * X).sum(axis=1)
    sq_d = np.maximum(norms[:, None] + norms[None, :] - 2.0 * (X @ X.T), 0.0)

    if sigma is None:
        # Mediana dos elementos não-diagonais
        n = X.shape[0]
        mask = ~np.eye(n, dtype=bool)
        sigma = np.sqrt(np.median(sq_d[mask]) / 2.0)
        sigma = max(sigma, 1e-12)

    K = np.exp(-sq_d / (2.0 * sigma * sigma))
    return K


# ---------------------------------------------------------------------------
# HSIC e CKA
# ---------------------------------------------------------------------------

def center_gram(K: np.ndarray) -> np.ndarray:
    """
    Centraliza Gram matrix: K' = H K H, onde H = I - (1/N) 11ᵀ.

    Computacionalmente equivalente a subtrair média de linhas/colunas
    e adicionar média global.
    """
    n = K.shape[0]
    means_row = K.mean(axis=0, keepdims=True)         # [1, N]
    means_col = K.mean(axis=1, keepdims=True)         # [N, 1]
    mean_all = K.mean()
    return K - means_row - means_col + mean_all


def hsic(K_c: np.ndarray, L_c: np.ndarray) -> float:
    """
    HSIC com matrizes já centralizadas: trace(K_c L_c) / (N-1)²

    Implementação via produto Frobenius: sum(K_c * L_c) = trace(K_c L_c)
    para matrizes simétricas centralizadas.
    """
    n = K_c.shape[0]
    return float(np.sum(K_c * L_c) / ((n - 1) ** 2))


def cka(X: np.ndarray, Y: np.ndarray, kernel: str = "linear") -> float:
    """
    Centered Kernel Alignment entre X [N, Dx] e Y [N, Dy].

    Parameters
    ----------
    X, Y:
        Matrizes de features com mesmo número de amostras N.
    kernel:
        'linear' ou 'rbf'.

    Returns
    -------
    float em [0, 1]
    """
    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X)
        L = rbf_kernel(Y)
    else:
        raise ValueError(f"Kernel desconhecido: {kernel}")

    K_c = center_gram(K)
    L_c = center_gram(L)

    h_xy = hsic(K_c, L_c)
    h_xx = hsic(K_c, K_c)
    h_yy = hsic(L_c, L_c)

    denom = np.sqrt(h_xx * h_yy)
    if denom < 1e-12:
        return float("nan")
    return float(h_xy / denom)


# ---------------------------------------------------------------------------
# Carregamento dos shards
# ---------------------------------------------------------------------------

def load_embeddings(
    folder: str,
    n_samples: int,
    keys: Tuple[str, ...] = ("visual_global", "text_feats"),
) -> dict:
    """
    Carrega N amostras emparelhadas de visual_global + text_feats.

    Returns
    -------
    dict com:
        visual_global: [N, 768]  float32
        text_feats:    [N, 4096] float32 (squeezed)
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum shard em {folder}")

    chunks = {k: [] for k in keys}
    collected = 0

    for f in files:
        if collected >= n_samples:
            break
        try:
            with h5py.File(f, "r") as h5:
                missing = [k for k in keys if k not in h5]
                if missing:
                    print(f"  [WARN] {os.path.basename(f)} sem {missing} — pulado")
                    continue

                # Determina take pelo primeiro key
                n_avail = h5[keys[0]].shape[0]
                take = min(n_avail, n_samples - collected)

                for k in keys:
                    arr = h5[k][:take].astype(np.float32)
                    if k == "text_feats" and arr.ndim == 3:
                        arr = arr.squeeze(1)            # [N, 1, 4096] → [N, 4096]
                    chunks[k].append(arr)
                collected += take
        except Exception as e:
            print(f"  [WARN] Falha ao ler {f}: {e}")

    out = {k: np.concatenate(chunks[k], axis=0)[:n_samples] for k in keys}
    return out


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_cka_matrix(
    results: dict,
    save_path: str,
) -> None:
    """
    Heatmap das CKAs computadas + barplot horizontal.
    """
    labels = list(results.keys())
    values = [results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color="#4A90D9", alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CKA")
    ax.set_title("Centered Kernel Alignment")
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="strong alignment >= 0.5")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_cka_analysis(
    folder_anyup: str,
    folder_noup: str,
    n_samples: int,
    output_dir: str,
    split: str,
    kernel: str = "linear",
) -> dict:
    """
    Computa CKA entre os pares relevantes:
      - visual_anyup ↔ text                  (alinhamento bruto com AnyUp)
      - visual_noup  ↔ text                  (alinhamento bruto sem AnyUp)
      - visual_anyup ↔ visual_noup           (preservação geométrica do CLS)

    Nota: o text_feats é determinístico (Qwen sobre o mesmo caption), então
    usa-se UM SÓ tensor de texto. Antes, verifica-se que os dois shards
    contêm o mesmo texto via norma da diferença (sanity check).
    """
    print(f"\n{'═' * 70}")
    print(f"  CKA  —  split={split}  kernel={kernel}  N={n_samples}")
    print(f"{'═' * 70}")

    print(f"\n[1/3] Loading embeddings...")
    data_anyup = load_embeddings(folder_anyup, n_samples)
    data_noup  = load_embeddings(folder_noup,  n_samples)

    v_anyup = data_anyup["visual_global"]
    v_noup  = data_noup["visual_global"]
    t_anyup = data_anyup["text_feats"]
    t_noup  = data_noup["text_feats"]

    # Truncar para a menor cardinalidade
    n = min(v_anyup.shape[0], v_noup.shape[0], t_anyup.shape[0], t_noup.shape[0])
    v_anyup, v_noup = v_anyup[:n], v_noup[:n]
    t_anyup, t_noup = t_anyup[:n], t_noup[:n]

    print(f"  visual_anyup: {v_anyup.shape}")
    print(f"  visual_noup : {v_noup.shape}")
    print(f"  text_anyup  : {t_anyup.shape}")
    print(f"  text_noup   : {t_noup.shape}")
    print(f"  Effective N : {n}")

    # Sanity check: textos devem ser idênticos (mesmo caption → mesmo Qwen)
    diff = float(np.linalg.norm(t_anyup - t_noup) / np.linalg.norm(t_anyup))
    print(f"\n  [sanity] ||t_anyup - t_noup|| / ||t_anyup|| = {diff:.6e}")
    if diff < 1e-4:
        print(f"  OK: Texts are identical -- shards aligned per sample. Using t_anyup.")
        t = t_anyup
    else:
        print(f"  [WARN] Texts diverge by {diff:.4f} -- shards NOT aligned per sample!")
        print(f"         Splits were shuffled in different orders during extraction.")
        print(f"         visual<->text CKA will be biased. Consider regenerating with the same seed.")
        t = t_anyup  # usa anyup como referência mesmo assim

    print(f"\n[2/3] Computing CKAs ({kernel})...")

    pairs = {
        "visual_anyup <-> text":         (v_anyup, t),
        "visual_noup  <-> text":         (v_noup,  t),
        "visual_anyup <-> visual_noup":  (v_anyup, v_noup),
    }

    results: dict = {}
    for name, (X, Y) in pairs.items():
        score = cka(X, Y, kernel=kernel)
        results[name] = score
        print(f"  {name:<32}  CKA = {score:.6f}")

    # Interpretação
    print(f"\n{'-' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'-' * 70}")

    cka_v_cross = results["visual_anyup <-> visual_noup"]
    if cka_v_cross >= 0.95:
        v_msg = "Visual_global is PRACTICALLY IDENTICAL between the two extractions"
    elif cka_v_cross >= 0.85:
        v_msg = "Visual_global has HIGH alignment with small differences"
    elif cka_v_cross >= 0.70:
        v_msg = "Visual_global has MODERATE alignment -- AnyUp visibly alters the CLS"
    else:
        v_msg = "Visual_global DIVERGES substantially between extractions"
    print(f"  visual_anyup <-> visual_noup = {cka_v_cross:.4f}  ->  {v_msg}")

    a_vt = results["visual_anyup <-> text"]
    n_vt = results["visual_noup  <-> text"]
    delta = n_vt - a_vt
    if abs(delta) < 0.01:
        align_msg = "RAW visual<->text alignment is practically the same"
    elif delta > 0:
        align_msg = (f"No-AnyUp has higher RAW alignment by +{delta:.4f} "
                     f"-> confirms that AnyUp introduces noise")
    else:
        align_msg = (f"AnyUp has higher RAW alignment by +{abs(delta):.4f} "
                     f"-> upsampling helps in the raw space")
    print(f"  Delta alignment (noup - anyup) = {delta:+.4f}  ->  {align_msg}")

    # Salva relatório
    report = {
        "split": split,
        "kernel": kernel,
        "n_samples": int(n),
        "folder_anyup": folder_anyup,
        "folder_noup": folder_noup,
        "cka": results,
    }

    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"cka_{split}_{kernel}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  [report] saved to {json_path}")

    png_path = os.path.join(output_dir, f"cka_{split}_{kernel}.png")
    plot_cka_matrix(results, png_path)

    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKA entre visual e text embeddings")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Amostras pareadas por conjunto (default: 5000)")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="linear",
                        help="Kernel para Gram matrices (default: linear)")
    parser.add_argument("--folder_anyup", type=str, default=None)
    parser.add_argument("--folder_noup", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/cka")
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

        report = run_cka_analysis(
            folder_anyup=f_anyup,
            folder_noup=f_noup,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            split=split,
            kernel=args.kernel,
        )
        all_reports[split] = report

    # Sumário final
    if len(all_reports) > 1:
        print(f"\n\n{'═' * 70}")
        print(f"  FINAL SUMMARY -- CKA per split")
        print(f"{'═' * 70}")
        pairs_keys = list(next(iter(all_reports.values()))["cka"].keys())
        header = f"  {'Split':<8} " + "".join(f"{k[:24]:>26}" for k in pairs_keys)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for split, rep in all_reports.items():
            row = f"  {split:<8} " + "".join(f"{rep['cka'][k]:>26.4f}" for k in pairs_keys)
            print(row)
