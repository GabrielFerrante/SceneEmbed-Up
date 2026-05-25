"""
analyze_images.py
-----------------
Analisa as propriedades das imagens e captions brutas do dataset COYO extraído.

Métricas geradas
----------------
Imagens:
  - Distribuição de resoluções (largura × altura)
  - Distribuição de aspect ratios
  - Estatísticas de canais RGB (média/desvio por canal)
  - Distribuição de tamanhos em disco (KB)
  - Percentual de imagens corrompidas / sem caption

Captions:
  - Distribuição de comprimento (palavras e caracteres)
  - Vocabulário: top-N palavras mais frequentes
  - Percentual de captions vazias

Saída
-----
  results/dataset_analysis_images.json  — dados brutos
  results/dataset_analysis_images.png   — painel de gráficos

Uso
---
  python data/analyze_images.py
  python data/analyze_images.py --root F:/COYO/coyo/extracted --max_samples 50000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Coleta de amostras
# ---------------------------------------------------------------------------

def _collect_image_paths(root_dir: str, max_samples: int) -> List[Path]:
    """
    Varre recursivamente root_dir e retorna até max_samples paths de imagem.

    Parameters
    ----------
    root_dir:
        Diretório raiz com imagens extraídas do img2dataset.
    max_samples:
        Limite de amostras para análise (0 = sem limite).

    Returns
    -------
    list[Path]
        Lista de paths de imagem encontrados.
    """
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    paths: List[Path] = []

    print(f"Varrendo {root_dir} ...")
    for p in Path(root_dir).rglob("*"):
        if p.suffix.lower() in extensions:
            paths.append(p)
            if max_samples > 0 and len(paths) >= max_samples:
                break

    print(f"  {len(paths):,} imagens encontradas.")
    return paths


# ---------------------------------------------------------------------------
# Análise de imagens
# ---------------------------------------------------------------------------

def _analyze_images(
    paths: List[Path],
) -> Dict:
    """
    Extrai propriedades visuais e textuais de um conjunto de imagens.

    Parameters
    ----------
    paths:
        Lista de paths de imagem.

    Returns
    -------
    dict com listas de métricas brutas para plotagem posterior.
    """
    widths: List[int] = []
    heights: List[int] = []
    aspect_ratios: List[float] = []
    file_sizes_kb: List[float] = []
    r_means: List[float] = []
    g_means: List[float] = []
    b_means: List[float] = []
    r_stds: List[float] = []
    g_stds: List[float] = []
    b_stds: List[float] = []
    caption_lengths_words: List[int] = []
    caption_lengths_chars: List[int] = []
    word_counter: Counter = Counter()

    n_corrupt = 0
    n_missing_caption = 0
    n_empty_caption = 0

    for p in tqdm(paths, desc="Analisando imagens", unit="img"):
        # ── Tamanho em disco ────────────────────────────────────────────────
        try:
            file_sizes_kb.append(p.stat().st_size / 1024)
        except OSError:
            file_sizes_kb.append(0.0)

        # ── Imagem ──────────────────────────────────────────────────────────
        try:
            with Image.open(p) as img:
                img_rgb = img.convert("RGB")
                w, h = img_rgb.size
                widths.append(w)
                heights.append(h)
                aspect_ratios.append(w / max(h, 1))

                arr = np.array(img_rgb, dtype=np.float32) / 255.0
                r_means.append(float(arr[:, :, 0].mean()))
                g_means.append(float(arr[:, :, 1].mean()))
                b_means.append(float(arr[:, :, 2].mean()))
                r_stds.append(float(arr[:, :, 0].std()))
                g_stds.append(float(arr[:, :, 1].std()))
                b_stds.append(float(arr[:, :, 2].std()))
        except Exception:
            n_corrupt += 1
            # Mantém as outras listas com valores sentinela para não desalinhar contagens
            widths.append(0)
            heights.append(0)
            aspect_ratios.append(0.0)
            r_means.append(0.0); g_means.append(0.0); b_means.append(0.0)
            r_stds.append(0.0);  g_stds.append(0.0);  b_stds.append(0.0)

        # ── Caption ─────────────────────────────────────────────────────────
        txt_path = p.with_suffix(".txt")
        if not txt_path.exists():
            n_missing_caption += 1
            caption_lengths_words.append(0)
            caption_lengths_chars.append(0)
        else:
            try:
                caption = txt_path.read_text(encoding="utf-8", errors="ignore").strip()
                if not caption:
                    n_empty_caption += 1
                words = caption.split()
                caption_lengths_words.append(len(words))
                caption_lengths_chars.append(len(caption))
                word_counter.update(w.lower().strip(".,!?;:\"'") for w in words)
            except Exception:
                n_missing_caption += 1
                caption_lengths_words.append(0)
                caption_lengths_chars.append(0)

    return {
        "widths": widths,
        "heights": heights,
        "aspect_ratios": aspect_ratios,
        "file_sizes_kb": file_sizes_kb,
        "r_means": r_means, "g_means": g_means, "b_means": b_means,
        "r_stds": r_stds,   "g_stds": g_stds,   "b_stds": b_stds,
        "caption_lengths_words": caption_lengths_words,
        "caption_lengths_chars": caption_lengths_chars,
        "word_counter": word_counter,
        "n_total": len(paths),
        "n_corrupt": n_corrupt,
        "n_missing_caption": n_missing_caption,
        "n_empty_caption": n_empty_caption,
    }


# ---------------------------------------------------------------------------
# Sumário estatístico
# ---------------------------------------------------------------------------

def _summary_stats(values: List[float], name: str) -> Dict:
    """
    Calcula percentis e momentos básicos de uma lista numérica.

    Parameters
    ----------
    values:
        Lista de valores numéricos.
    name:
        Nome da variável (usado como chave no dicionário de saída).

    Returns
    -------
    dict com min, max, mean, std e percentis 5/25/50/75/95.
    """
    arr = np.array([v for v in values if v > 0], dtype=np.float64)
    if len(arr) == 0:
        return {name: {}}
    return {
        name: {
            "count": int(len(arr)),
            "mean":  float(arr.mean()),
            "std":   float(arr.std()),
            "min":   float(arr.min()),
            "p5":    float(np.percentile(arr, 5)),
            "p25":   float(np.percentile(arr, 25)),
            "p50":   float(np.percentile(arr, 50)),
            "p75":   float(np.percentile(arr, 75)),
            "p95":   float(np.percentile(arr, 95)),
            "max":   float(arr.max()),
        }
    }


# ---------------------------------------------------------------------------
# Plotagem
# ---------------------------------------------------------------------------

def _plot_results(data: Dict, out_path: str, top_words: int = 30) -> None:
    """
    Gera painel de gráficos com a distribuição das propriedades do dataset.

    Parameters
    ----------
    data:
        Dicionário retornado por _analyze_images().
    out_path:
        Caminho de saída do PNG.
    top_words:
        Número de palavras mais frequentes a exibir.
    """
    fig, axes = plt.subplots(3, 4, figsize=(22, 16))
    fig.suptitle("COYO Dataset — Distribution Analysis", fontsize=16, fontweight="bold")

    def _clean(lst: List, sentinel: float = 0.0) -> np.ndarray:
        return np.array([v for v in lst if v != sentinel], dtype=np.float64)

    # ── Linha 1: Geometria ───────────────────────────────────────────────────
    axes[0, 0].hist(_clean(data["widths"]),   bins=60, color="#3498db", edgecolor="white", linewidth=0.3)
    axes[0, 0].set_title("Width Distribution (px)")
    axes[0, 0].set_xlabel("Width"); axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(_clean(data["heights"]),  bins=60, color="#2ecc71", edgecolor="white", linewidth=0.3)
    axes[0, 1].set_title("Height Distribution (px)")
    axes[0, 1].set_xlabel("Height")

    ar = _clean(data["aspect_ratios"])
    ar_clipped = np.clip(ar, 0.2, 5.0)
    axes[0, 2].hist(ar_clipped, bins=60, color="#9b59b6", edgecolor="white", linewidth=0.3)
    axes[0, 2].set_title("Aspect Ratio (W/H)")
    axes[0, 2].set_xlabel("W/H (clipped 0.2–5)")
    axes[0, 2].axvline(1.0, color="red", linestyle="--", linewidth=1.2, label="1:1")
    axes[0, 2].legend()

    axes[0, 3].hist(_clean(data["file_sizes_kb"]), bins=60, color="#e67e22", edgecolor="white", linewidth=0.3)
    axes[0, 3].set_title("File Size on Disk (KB)")
    axes[0, 3].set_xlabel("KB")

    # ── Linha 2: Cores RGB ───────────────────────────────────────────────────
    for ax, vals, label, color in [
        (axes[1, 0], data["r_means"], "Mean R", "#e74c3c"),
        (axes[1, 1], data["g_means"], "Mean G", "#27ae60"),
        (axes[1, 2], data["b_means"], "Mean B", "#2980b9"),
    ]:
        arr = np.array(vals, dtype=np.float32)
        arr = arr[arr > 0]
        ax.hist(arr, bins=60, color=color, edgecolor="white", linewidth=0.3)
        ax.set_title(f"Distribution {label}")
        ax.set_xlabel("Value [0,1]")

    # Std RGB sobreposto
    for vals, label, color in [
        (data["r_stds"], "Std R", "#e74c3c"),
        (data["g_stds"], "Std G", "#27ae60"),
        (data["b_stds"], "Std B", "#2980b9"),
    ]:
        arr = np.array(vals, dtype=np.float32)
        arr = arr[arr > 0]
        axes[1, 3].hist(arr, bins=60, color=color, alpha=0.5, label=label, edgecolor="white", linewidth=0.2)
    axes[1, 3].set_title("RGB Standard Deviation")
    axes[1, 3].set_xlabel("Std [0,1]")
    axes[1, 3].legend()

    # ── Linha 3: Captions ───────────────────────────────────────────────────
    cw = np.array([v for v in data["caption_lengths_words"] if v > 0], dtype=np.int32)
    axes[2, 0].hist(cw, bins=60, color="#1abc9c", edgecolor="white", linewidth=0.3)
    axes[2, 0].set_title("Caption Length (words)")
    axes[2, 0].set_xlabel("Number of words")

    cc = np.array([v for v in data["caption_lengths_chars"] if v > 0], dtype=np.int32)
    axes[2, 1].hist(cc, bins=60, color="#f39c12", edgecolor="white", linewidth=0.3)
    axes[2, 1].set_title("Caption Length (characters)")
    axes[2, 1].set_xlabel("Number of characters")

    # Top palavras
    top = data["word_counter"].most_common(top_words)
    words_labels = [w for w, _ in top]
    counts = [c for _, c in top]
    axes[2, 2].barh(words_labels[::-1], counts[::-1], color="#8e44ad", edgecolor="white", linewidth=0.3)
    axes[2, 2].set_title(f"Top {top_words} Words")
    axes[2, 2].set_xlabel("Frequency")
    axes[2, 2].tick_params(axis="y", labelsize=7)

    # Qualidade do dataset
    n = data["n_total"]
    quality_labels = ["Valid", "Corrupted", "No caption", "Empty caption"]
    quality_vals = [
        n - data["n_corrupt"] - data["n_missing_caption"],
        data["n_corrupt"],
        data["n_missing_caption"],
        data["n_empty_caption"],
    ]
    quality_vals = [max(v, 0) for v in quality_vals]
    colors_q = ["#2ecc71", "#e74c3c", "#f39c12", "#95a5a6"]
    axes[2, 3].pie(quality_vals, labels=quality_labels, colors=colors_q,
                   autopct="%1.1f%%", startangle=140, textprops={"fontsize": 9})
    axes[2, 3].set_title("Dataset Quality")

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved to: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def analyze(root_dir: str, max_samples: int, output_dir: str, top_words: int) -> None:
    """
    Pipeline completo: coleta → análise → sumário → plot → JSON.

    Parameters
    ----------
    root_dir:
        Diretório raiz do dataset extraído.
    max_samples:
        Limite de amostras (0 = todas).
    output_dir:
        Diretório de saída para JSON e PNG.
    top_words:
        Número de palavras mais frequentes a salvar/plotar.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = _collect_image_paths(root_dir, max_samples)
    if not paths:
        print("Nenhuma imagem encontrada. Verifique o caminho.")
        return

    data = _analyze_images(paths)

    # ── Sumário estatístico ────────────────────────────────────────────────
    summary: Dict = {
        "timestamp": datetime.now().isoformat(),
        "root_dir": root_dir,
        "n_analyzed": len(paths),
        "n_corrupt": data["n_corrupt"],
        "n_missing_caption": data["n_missing_caption"],
        "n_empty_caption": data["n_empty_caption"],
        "corruption_rate": data["n_corrupt"] / max(len(paths), 1),
    }
    for key in ("widths", "heights", "aspect_ratios", "file_sizes_kb",
                "r_means", "g_means", "b_means", "r_stds", "g_stds", "b_stds",
                "caption_lengths_words", "caption_lengths_chars"):
        summary.update(_summary_stats(data[key], key))

    summary["top_words"] = {w: c for w, c in data["word_counter"].most_common(top_words)}

    # ── Persistência ───────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, "dataset_analysis_images.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"JSON saved to: {json_path}")

    png_path = os.path.join(output_dir, "dataset_analysis_images.png")
    _plot_results(data, png_path, top_words=top_words)

    # ── Impressão rápida ───────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("IMAGE ANALYSIS SUMMARY")
    print("=" * 55)
    print(f"  Samples analyzed    : {len(paths):>10,}")
    print(f"  Corrupted images    : {data['n_corrupt']:>10,}  "
          f"({100 * data['n_corrupt'] / max(len(paths), 1):.2f}%)")
    print(f"  No caption          : {data['n_missing_caption']:>10,}")
    w_arr = np.array([v for v in data["widths"]  if v > 0])
    h_arr = np.array([v for v in data["heights"] if v > 0])
    if len(w_arr):
        print(f"  Median resolution   : {int(np.median(w_arr))} × {int(np.median(h_arr))} px")
    cw_arr = np.array([v for v in data["caption_lengths_words"] if v > 0])
    if len(cw_arr):
        print(f"  Median caption      : {int(np.median(cw_arr))} words")
    print("=" * 55)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Análise de imagens e captions do COYO extraído.")
    parser.add_argument("--root",        default="F:/COYO/coyo/extracted",  help="Diretório raiz do dataset")
    parser.add_argument("--max_samples", type=int, default=0,          help="Limite de amostras (0=todas)")
    parser.add_argument("--output_dir",  default="results",                  help="Diretório de saída")
    parser.add_argument("--top_words",   type=int, default=30,               help="Top N palavras mais frequentes")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    analyze(
        root_dir=args.root,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        top_words=args.top_words,
    )
