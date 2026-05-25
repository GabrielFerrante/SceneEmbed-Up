"""
analyze_embeddings.py
---------------------
Analisa as propriedades dos embeddings pré-computados nos shards H5.

Métricas geradas
----------------
Visual (patches):
  - Norma L2 por patch e por amostra
  - Distribuição de ativações (valores absolutos médios)
  - Variância entre patches de uma mesma imagem (diversidade local)
  - Dead features: % de dimensões zeradas por amostra

Texto:
  - Norma L2 por embedding
  - Distribuição de ativações

Alinhamento:
  - Distribuição de similaridade coseno visual–texto (par correto vs. aleatório)
  - Gap: sim(correto) − sim(aleatório)
  - Matriz de similaridade de uma sub-amostra (visualizada como heatmap)

Global (CLS token, se disponível):
  - Norma L2 e distribuição de ativações

PCA 2D:
  - Projeção de N amostras visuais (tokens globais) para visualização

Saída
-----
  results/dataset_analysis_embeddings.json
  results/dataset_analysis_embeddings.png

Uso
---
  python data/analyze_embeddings.py
  python data/analyze_embeddings.py --shard_dir G:/coyo/embeds/val_anyup --max_samples 5000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ---------------------------------------------------------------------------
# Carregamento de amostras dos shards H5
# ---------------------------------------------------------------------------

def _load_samples(
    shard_dir: str,
    max_samples: int,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Carrega amostras aleatórias dos shards H5.

    Parameters
    ----------
    shard_dir:
        Diretório com arquivos .h5 (busca recursiva).
    max_samples:
        Número máximo de amostras a carregar.

    Returns
    -------
    visual_feats:
        float32 array [N, 1024, 768] ou [N, P, D].
    text_feats:
        float32 array [N, 1, 4096] ou [N, T, D].
    global_feats:
        float32 array [N, 768] se visual_global existir, None caso contrário.
    """
    pattern = os.path.join(shard_dir, "**", "*.h5")
    shard_files = sorted(glob.glob(pattern, recursive=True))

    if not shard_files:
        raise RuntimeError(f"Nenhum arquivo .h5 encontrado em {shard_dir}")

    print(f"  {len(shard_files)} shards encontrados.")

    visuals: List[np.ndarray] = []
    texts:   List[np.ndarray] = []
    globals_: List[np.ndarray] = []
    has_global = True
    collected = 0

    for path in tqdm(shard_files, desc="Carregando shards"):
        if collected >= max_samples:
            break
        try:
            with h5py.File(path, "r") as f:
                n = f["visual_feats"].shape[0]
                remaining = max_samples - collected
                take = min(n, remaining)

                # Amostragem aleatória dentro do shard
                idxs = np.sort(np.random.choice(n, take, replace=False))
                visuals.append(f["visual_feats"][idxs].astype(np.float32))
                texts.append(f["text_feats"][idxs].astype(np.float32))

                if has_global and "visual_global" in f:
                    globals_.append(f["visual_global"][idxs].astype(np.float32))
                else:
                    has_global = False

                collected += take
        except Exception as e:
            print(f"  [WARN] Shard ignorado: {path} — {e}")

    if not visuals:
        raise RuntimeError("Nenhuma amostra válida carregada.")

    V = np.concatenate(visuals, axis=0)
    T = np.concatenate(texts,   axis=0)
    G = np.concatenate(globals_, axis=0) if has_global and globals_ else None

    print(f"  Amostras carregadas: {len(V):,}")
    print(f"  Shapes — Visual: {V.shape}, Texto: {T.shape}"
          + (f", Global: {G.shape}" if G is not None else ""))
    return V, T, G


# ---------------------------------------------------------------------------
# Métricas de embeddings
# ---------------------------------------------------------------------------

def _compute_norms(arr: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Norma L2 ao longo de um eixo.

    Parameters
    ----------
    arr:
        Array de qualquer shape.
    axis:
        Eixo ao longo do qual calcular a norma.

    Returns
    -------
    np.ndarray com shape reduzido em `axis`.
    """
    return np.linalg.norm(arr, axis=axis)


def _cosine_similarity_diag(
    A: np.ndarray,
    B: np.ndarray,
    chunk: int = 512,
) -> np.ndarray | None:
    """
    Similaridade coseno entre pares correspondentes (diagonal).
    Retorna None se A e B tiverem dimensões incompatíveis.
    """
    if A.shape[1] != B.shape[1]:
        print(f"  [INFO] Incompatible dimensions ({A.shape[1]} vs {B.shape[1]}) — "
              "cross-modal similarity requires the Aligner. Skipping.")
        return None

    N = A.shape[0]
    sims = np.zeros(N, dtype=np.float32)
    for start in range(0, N, chunk):
        end = min(start + chunk, N)
        a = A[start:end]
        b = B[start:end]
        a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        sims[start:end] = (a_norm * b_norm).sum(axis=1)
    return sims

def _cosine_similarity_intra(
    X: np.ndarray,
    n_pairs: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Similaridade coseno entre pares ALEATÓRIOS dentro do mesmo espaço.
    Usado para medir colapso / diversidade dos embeddings.

    Parameters
    ----------
    X:
        [N, D] embeddings.
    n_pairs:
        Número de pares aleatórios a amostrar.

    Returns
    -------
    np.ndarray [n_pairs] com similaridades intra-modais.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    idxA = rng.choice(N, n_pairs, replace=True)
    idxB = rng.choice(N, n_pairs, replace=True)
    same = idxA == idxB
    idxB[same] = (idxB[same] + 1) % N

    a = X[idxA]
    b = X[idxB]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=1)


def _cosine_similarity_random_pairs(
    A: np.ndarray,
    B: np.ndarray,
    n_pairs: int = 5_000,
    seed: int = 42,
) -> np.ndarray:
    """
    Similaridade coseno entre pares ALEATÓRIOS (não correspondentes).

    Serve como baseline de distribuição de pares negativos.

    Parameters
    ----------
    A:
        [N, D] embeddings visuais.
    B:
        [N, D] embeddings textuais.
    n_pairs:
        Número de pares aleatórios a amostrar.
    seed:
        Seed para reprodutibilidade.

    Returns
    -------
    np.ndarray [n_pairs] com similaridades de pares negativos.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    idxA = rng.choice(N, n_pairs, replace=True)
    idxB = rng.choice(N, n_pairs, replace=True)
    # Garante que não são pares corretos
    same = idxA == idxB
    idxB[same] = (idxB[same] + 1) % N

    a = A[idxA]
    b = B[idxB]
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return (a_norm * b_norm).sum(axis=1)


def _dead_feature_ratio(arr: np.ndarray, threshold: float = 1e-4) -> float:
    """
    Fração de dimensões com ativação média abaixo de threshold.

    Parameters
    ----------
    arr:
        [N, D] embeddings.
    threshold:
        Limiar de ativação para considerar "morta".

    Returns
    -------
    float entre 0 e 1.
    """
    mean_act = np.abs(arr).mean(axis=0)  # [D]
    return float((mean_act < threshold).mean())


def _intra_image_variance(visual: np.ndarray, n_sample: int = 500) -> np.ndarray:
    """
    Variância média entre patches de uma mesma imagem.

    Um valor alto indica que a imagem contém regiões visualmente diversas.

    Parameters
    ----------
    visual:
        [N, P, D] patches visuais.
    n_sample:
        Número de imagens a amostrar para eficiência.

    Returns
    -------
    np.ndarray [n_sample] com a variância média por imagem.
    """
    idxs = np.random.choice(len(visual), min(n_sample, len(visual)), replace=False)
    return visual[idxs].var(axis=1).mean(axis=1)  # [n_sample]


# ---------------------------------------------------------------------------
# Plotagem
# ---------------------------------------------------------------------------

def _plot_results(
    V: np.ndarray,
    T: np.ndarray,
    G: Optional[np.ndarray],
    sims_pos: np.ndarray,
    sims_neg: np.ndarray,
    out_path: str,
    cross_modal_available: bool = False,  
    n_heatmap: int = 100,
) -> None:
    """
    Gera painel completo de análise de embeddings.

    Parameters
    ----------
    V:
        [N, P, D] patches visuais.
    T:
        [N, 1 ou T, D] embeddings textuais.
    G:
        [N, D] tokens globais (CLS), ou None.
    sims_pos:
        [N] similaridades de pares corretos.
    sims_neg:
        [K] similaridades de pares aleatórios.
    out_path:
        Caminho de saída do PNG.
    n_heatmap:
        Número de amostras para a sub-matriz de similaridade.
    """
    n_rows = 3 if G is not None else 2
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 6 * n_rows))
    fig.suptitle("COYO Embeddings — Distribution Analysis", fontsize=15, fontweight="bold")

    V_flat = V.reshape(len(V), -1)     # [N, P*D]
    T_flat = T.reshape(len(T), -1)     # [N, T*D]
    V_mean = V.mean(axis=1)            # [N, D] — média dos patches por imagem

    # ── Linha 1: Normas ─────────────────────────────────────────────────────
    v_patch_norms = _compute_norms(V, axis=-1).mean(axis=1)    # norma média por imagem
    t_norms       = _compute_norms(T.squeeze(1), axis=-1)      # [N]

    axes[0, 0].hist(v_patch_norms, bins=60, color="#3498db", edgecolor="white", linewidth=0.3)
    axes[0, 0].set_title("L2 Norm — Visual (avg patches/img)")
    axes[0, 0].set_xlabel("L2 Norm")

    axes[0, 1].hist(t_norms, bins=60, color="#e74c3c", edgecolor="white", linewidth=0.3)
    axes[0, 1].set_title("L2 Norm — Text")
    axes[0, 1].set_xlabel("L2 Norm")

    # ── Variância intra-imagem ───────────────────────────────────────────────
    intra_var = _intra_image_variance(V, n_sample=min(1000, len(V)))
    axes[0, 2].hist(intra_var, bins=60, color="#9b59b6", edgecolor="white", linewidth=0.3)
    axes[0, 2].set_title("Intra-Image Variance\n(patch diversity)")
    axes[0, 2].set_xlabel("Avg variance across patches")

    # ── Dead features ────────────────────────────────────────────────────────
    dead_v = _dead_feature_ratio(V_mean)
    dead_t = _dead_feature_ratio(T.squeeze(1))
    categories = ["Visual\n(patches)", "Texto"]
    dead_vals  = [dead_v * 100, dead_t * 100]
    bars = axes[0, 3].bar(categories, dead_vals, color=["#3498db", "#e74c3c"],
                          edgecolor="white", width=0.5)
    for bar, val in zip(bars, dead_vals):
        axes[0, 3].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", fontsize=10)
    axes[0, 3].set_title("Dead Features\n(dims with activation < 1e-4)")
    axes[0, 3].set_ylabel("% dimensions")
    axes[0, 3].set_ylim(0, max(dead_vals) * 1.3 + 1)

    # ── Linha 2: Similaridade e alinhamento ──────────────────────────────────
    axes[1, 0].hist(sims_pos, bins=60, color="#2ecc71", alpha=0.8, label="Correct pair",  edgecolor="white", linewidth=0.3)
    axes[1, 0].hist(sims_neg, bins=60, color="#e74c3c", alpha=0.6, label="Random pair", edgecolor="white", linewidth=0.3)
    axes[1, 0].set_title("Cosine Similarity Distribution\nVisual–Text")
    axes[1, 0].set_xlabel("Coseno")
    axes[1, 0].legend()

    gap = sims_pos.mean() - sims_neg.mean()
    axes[1, 1].bar(["Correct", "Random"],
                   [sims_pos.mean(), sims_neg.mean()],
                   color=["#2ecc71", "#e74c3c"],
                   yerr=[sims_pos.std(), sims_neg.std()],
                   capsize=8, edgecolor="white")
    axes[1, 1].set_title(f"Average Similarity\nGap = {gap:.4f}")
    axes[1, 1].set_ylabel("Avg cosine")

    # Heatmap de sub-matriz de similaridade
    n_h = min(n_heatmap, len(V))
    idxs_h = np.random.choice(len(V), n_h, replace=False)
    v_sub = V_mean[idxs_h]
    t_sub = T.squeeze(1)[idxs_h]
    v_sub_n = v_sub / (np.linalg.norm(v_sub, axis=1, keepdims=True) + 1e-8)
    t_sub_n = t_sub / (np.linalg.norm(t_sub, axis=1, keepdims=True) + 1e-8)

    if v_sub_n.shape[1] == t_sub_n.shape[1]:
        # Cross-modal disponível
        sim_sub = v_sub_n @ t_sub_n.T
        heatmap_title = f"Heatmap Similaridade V↔T\n({n_h}×{n_h} sub-amostras)"
    else:
        # Intra-visual: imagem vs imagem
        sim_sub = v_sub_n @ v_sub_n.T
        heatmap_title = f"Heatmap Similaridade V↔V\n({n_h}×{n_h} sub-amostras, intra-visual)"

    im = axes[1, 2].imshow(sim_sub, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1.0)
    axes[1, 2].set_title(heatmap_title)
    axes[1, 2].set_xlabel("Imagem j")
    axes[1, 2].set_ylabel("Imagem i")
    plt.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    # Percentis da distribuição de similaridade
    percentis = [5, 10, 25, 50, 75, 90, 95]
    vals_pct = [float(np.percentile(sims_pos, p)) for p in percentis]
    axes[1, 3].plot(percentis, vals_pct, "o-", color="#2980b9", linewidth=2)
    axes[1, 3].axhline(0, color="gray", linestyle="--", linewidth=1)

    # Label dinâmico dependendo do modo
    pos_label = "Sim V↔T (correto)" if cross_modal_available else "Sim V↔V (intra-visual)"
    axes[1, 3].set_title(f"Percentis — {pos_label}")
    axes[1, 3].set_xlabel("Percentil")
    axes[1, 3].set_ylabel("Coseno")
    axes[1, 3].grid(alpha=0.3)

    # ── Linha 3: Global (CLS) e PCA ─────────────────────────────────────────
    if G is not None:
        g_norms = _compute_norms(G, axis=-1)
        axes[2, 0].hist(g_norms, bins=60, color="#f39c12", edgecolor="white", linewidth=0.3)
        axes[2, 0].set_title("L2 Norm — CLS Token (Global)")
        axes[2, 0].set_xlabel("L2 Norm")

        # Distribuição de ativações do CLS
        g_mean_act = np.abs(G).mean(axis=0)
        axes[2, 1].plot(sorted(g_mean_act, reverse=True), color="#d35400", linewidth=0.8)
        axes[2, 1].set_title("Avg Dimension Activation\n(CLS token, sorted)")
        axes[2, 1].set_xlabel("Dimension (sorted)"); axes[2, 1].set_ylabel("Avg activation")
        axes[2, 1].set_yscale("log")

        # PCA 2D do CLS
        n_pca = min(3000, len(G))
        idxs_pca = np.random.choice(len(G), n_pca, replace=False)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(G[idxs_pca])
        axes[2, 2].scatter(coords[:, 0], coords[:, 1], s=3, alpha=0.4, c="#8e44ad")
        var = pca.explained_variance_ratio_
        axes[2, 2].set_title(f"PCA 2D — CLS Token\n(exp var: {100*var[0]:.1f}% + {100*var[1]:.1f}%)")
        axes[2, 2].set_xlabel("PC1"); axes[2, 2].set_ylabel("PC2")

        # Correlação V_mean × G
        v_mean_n = V_mean / (np.linalg.norm(V_mean, axis=1, keepdims=True) + 1e-8)
        g_n      = G       / (np.linalg.norm(G,       axis=1, keepdims=True) + 1e-8)
        corr = (v_mean_n * g_n).sum(axis=1)   # coseno entre média dos patches e CLS
        axes[2, 3].hist(corr, bins=60, color="#16a085", edgecolor="white", linewidth=0.3)
        axes[2, 3].set_title("Corr. CLS vs. Avg Patches\n(how representative is the CLS)")
        axes[2, 3].set_xlabel("Coseno")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(
    shard_dir: str,
    max_samples: int,
    output_dir: str,
    n_neg_pairs: int,
    n_heatmap: int,
) -> None:
    """
    Pipeline completo de análise dos embeddings H5.

    Parameters
    ----------
    shard_dir:
        Diretório com shards H5.
    max_samples:
        Número máximo de amostras a carregar.
    output_dir:
        Diretório de saída.
    n_neg_pairs:
        Número de pares negativos para distribuição de similaridade.
    n_heatmap:
        Tamanho da sub-matriz para heatmap.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nLoading up to {max_samples:,} samples from {shard_dir} ...")
    V, T, G = _load_samples(shard_dir, max_samples)

    V_mean = V.mean(axis=1)          # [N, D] — representação por imagem
    T_flat = T.squeeze(1)            # [N, D]

    print("\nCalculando similaridades positivas (pares corretos)...")
    sims_pos = _cosine_similarity_diag(V_mean, T_flat)

    # Cross-modal não disponível sem Aligner — usa intra-modal como alternativa
    if sims_pos is None:
        print("  → Computing intra-visual similarity (patch diversity)...")
        sims_pos = _cosine_similarity_intra(V_mean, n_pairs=min(10_000, len(V_mean)))
        print("  → Computing intra-text similarity...")
        sims_neg = _cosine_similarity_intra(T_flat, n_pairs=min(10_000, len(T_flat)))
        cross_modal_available = False
    else:
        print("  → Computing negative cross-modal similarity...")
        sims_neg = _cosine_similarity_random_pairs(V_mean, T_flat, n_pairs=n_neg_pairs)
        cross_modal_available = True


    gap = float(sims_pos.mean()) - float(sims_neg.mean())

    # ── Sumário ───────────────────────────────────────────────────────────
    def _stats(arr: np.ndarray, name: str) -> Dict:
        return {
            name: {
                "mean":  float(arr.mean()),
                "std":   float(arr.std()),
                "p25":   float(np.percentile(arr, 25)),
                "p50":   float(np.percentile(arr, 50)),
                "p75":   float(np.percentile(arr, 75)),
                "p5":    float(np.percentile(arr, 5)),
                "p95":   float(np.percentile(arr, 95)),
            }
        }

    summary: Dict = {
        "timestamp":   datetime.now().isoformat(),
        "shard_dir":   shard_dir,
        "n_samples":   int(len(V)),
        "visual_shape": list(V.shape[1:]),
        "text_shape":   list(T.shape[1:]),
        "has_global":   G is not None,
        "alignment_gap_cosine": gap,
        "dead_features_visual": _dead_feature_ratio(V_mean),
        "dead_features_text":   _dead_feature_ratio(T_flat),
    }
    summary.update(_stats(_compute_norms(V, axis=-1).mean(axis=1), "visual_norm"))
    summary.update(_stats(_compute_norms(T_flat, axis=-1), "text_norm"))
    summary.update(_stats(sims_pos, "sim_positive"))
    summary.update(_stats(sims_neg, "sim_negative"))

    if G is not None:
        summary.update(_stats(_compute_norms(G, axis=-1), "global_norm"))

    json_path = os.path.join(output_dir, "dataset_analysis_embeddings.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nJSON saved to: {json_path}")

    png_path = os.path.join(output_dir, "dataset_analysis_embeddings.png")
    _plot_results(V, T, G, sims_pos, sims_neg, png_path,cross_modal_available = False, n_heatmap=n_heatmap)

    print("\n" + "=" * 55)
    print("EMBEDDINGS ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Samples             : {len(V):>10,}")

    if cross_modal_available:
        print(f"  Sim+ avg (V↔T correct)  : {sims_pos.mean():>10.4f}")
        print(f"  Sim- avg (V↔T random)   : {sims_neg.mean():>10.4f}")
        print(f"  Alignment gap           : {gap:>10.4f}")
    else:
        print(f"  Intra-visual sim (diversity): {sims_pos.mean():>10.4f}")
        print(f"  Intra-text sim   (diversity): {sims_neg.mean():>10.4f}")
        print(f"  (Cross-modal requires the Aligner projecting 768→4096)")
        
        
    print(f"  Alignment gap       : {gap:>10.4f}")
    print(f"  Dead features visual: {_dead_feature_ratio(V_mean)*100:>9.1f}%")
    print(f"  Dead features text  : {_dead_feature_ratio(T_flat)*100:>9.1f}%")
    print("=" * 55)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise dos embeddings H5 do COYO.")
    parser.add_argument("--shard_dir",   default="F:/COYO/embeds/train_anyup",  help="Diretório com shards H5")
    parser.add_argument("--max_samples", type=int, default=5_000,             help="Máximo de amostras a carregar")
    parser.add_argument("--output_dir",  default="results",                    help="Diretório de saída")
    parser.add_argument("--n_neg_pairs", type=int, default=10_000,            help="Pares negativos para distribuição")
    parser.add_argument("--n_heatmap",   type=int, default=100,                help="Sub-amostras para heatmap")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        shard_dir=args.shard_dir,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        n_neg_pairs=args.n_neg_pairs,
        n_heatmap=args.n_heatmap,
    )
