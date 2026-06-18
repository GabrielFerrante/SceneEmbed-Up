"""
panel_query_ablation.py
------------------------
Gera um painel tabular (4 aligners × 6 colunas) com a imagem top-1 de cada
combinacao aligner × query × modo (dense / reranked), a partir do
summary.json produzido por eval_sg_query_ablation.py.

Layout:
             full              half             dominant
          Dense  Reranked   Dense  Reranked   Dense  Reranked
  GT   ──────────────── [imagem ground truth] ──────────────
  noup    [img]  [img]     [img]  [img]      [img]  [img]
  anyup   [img]  [img]     [img]  [img]      [img]  [img]
  ats     [img]  [img]     [img]  [img]      [img]  [img]
  pca     [img]  [img]     [img]  [img]      [img]  [img]

Borda verde = imagem e o ground truth.

Uso:
    python viz/panel_query_ablation.py
    python viz/panel_query_ablation.py --summary results/sg_query_ablation/summary.json --image_dir F:/COYO/coyo/extracted/00000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io_utils import ensure_dir

ALIGNERS = ["noup", "anyup", "anyup_ats", "anyup_pca"]
ALIGNER_LABELS = ["No Upsampler", "AnyUp", "AnyUp + ATS", "AnyUp + PCA"]
QUERY_NAMES = ["full", "half", "dominant"]

COLUMNS = [
    ("full", "dense"),
    ("full", "reranked"),
    ("half", "dense"),
    ("half", "reranked"),
    ("dominant", "dense"),
    ("dominant", "reranked"),
]

COL_HEADERS = [
    "Full\nDense",
    "Full\nReranked",
    "Half\nDense",
    "Half\nReranked",
    "Dominant\nDense",
    "Dominant\nReranked",
]

BG_COLOR = "#1a1a2e"
GT_BORDER_COLOR = "#00e676"
DEFAULT_BORDER_COLOR = "#444466"


def load_top1_info(summary: dict) -> list[list[dict]]:
    """
    Retorna grid[aligner_idx][col_idx] = {filename, score, is_gt}.
    """
    gt_file = summary.get("gt_image")
    results = summary["results"]

    grid: list[list[dict]] = []
    for aligner in ALIGNERS:
        row = []
        for q_name, mode in COLUMNS:
            data = results.get(aligner, {}).get(q_name, {})

            if mode == "dense":
                paths = data.get("top_k_paths_dense", [])
                scores = data.get("scores_dense", [])
                if paths:
                    row.append({
                        "filename": paths[0],
                        "score": scores[0] if scores else 0.0,
                        "is_gt": paths[0] == gt_file,
                    })
                else:
                    row.append(None)
            else:
                paths = data.get("top_k_paths_reranked", [])
                scores_rr = data.get("scores_reranked", [])
                if paths:
                    row.append({
                        "filename": paths[0],
                        "score": scores_rr[0]["final"] if scores_rr else 0.0,
                        "is_gt": paths[0] == gt_file,
                    })
                else:
                    row.append(None)
        grid.append(row)
    return grid


def make_panel(summary: dict, image_dir: str, save_path: str) -> None:
    gt_file = summary.get("gt_image")
    queries = summary.get("queries", {})
    grid = load_top1_info(summary)

    n_rows = len(ALIGNERS) + 1  # +1 for GT row
    n_cols = len(COLUMNS)

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 3.8), facecolor=BG_COLOR)
    gs = fig.add_gridspec(
        n_rows, n_cols,
        height_ratios=[0.85] + [1.0] * len(ALIGNERS),
        hspace=0.35, wspace=0.12,
        left=0.06, right=0.98, top=0.92, bottom=0.02,
    )

    # Suptitle
    query_full_text = queries.get("full", "")[:55]
    query_half_text = queries.get("half", "")[:30]
    fig.suptitle(
        f"Query Ablation — Top-1 Retrieval per Aligner\n"
        f"full: \"{query_full_text}\"  |  half: \"{query_half_text}\"  |  dominant: \"{queries.get('dominant', '')}\"",
        color="white", fontsize=13, fontweight="bold", y=0.97,
    )

    # ── Row 0: Ground truth reference ─────────────────────────────────────
    if gt_file:
        gt_path = os.path.join(image_dir, gt_file)
        if os.path.exists(gt_path):
            gt_img = np.array(Image.open(gt_path).convert("RGB").resize((256, 256)))
        else:
            gt_img = np.full((256, 256, 3), 40, dtype=np.uint8)
    else:
        gt_img = np.full((256, 256, 3), 40, dtype=np.uint8)

    # Show GT in center columns, use side cells for label
    for j in range(n_cols):
        ax = fig.add_subplot(gs[0, j])
        ax.set_facecolor(BG_COLOR)

        if j == 2 or j == 3:
            ax.imshow(gt_img)
            for spine in ax.spines.values():
                spine.set_edgecolor(GT_BORDER_COLOR)
                spine.set_linewidth(3)
                spine.set_visible(True)
            if j == 2:
                ax.set_title(f"Ground Truth: {gt_file or '?'}", color=GT_BORDER_COLOR, fontsize=10, fontweight="bold")
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    # ── Rows 1-4: Aligner results ────────────────────────────────────────
    for i, (aligner, label) in enumerate(zip(ALIGNERS, ALIGNER_LABELS)):
        for j, (q_name, mode) in enumerate(COLUMNS):
            ax = fig.add_subplot(gs[i + 1, j])
            ax.set_facecolor(BG_COLOR)

            info = grid[i][j]
            if info is None:
                ax.text(0.5, 0.5, "N/A", color="gray", fontsize=14, ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
                continue

            img_path = os.path.join(image_dir, info["filename"])
            if os.path.exists(img_path):
                img = np.array(Image.open(img_path).convert("RGB").resize((256, 256)))
            else:
                img = np.full((256, 256, 3), 40, dtype=np.uint8)

            ax.imshow(img)

            border_color = GT_BORDER_COLOR if info["is_gt"] else DEFAULT_BORDER_COLOR
            border_width = 3 if info["is_gt"] else 1.5
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(border_width)
                spine.set_visible(True)

            score_text = f"{info['score']:.4f}"
            fn_text = info["filename"]
            gt_tag = "  [GT]" if info["is_gt"] else ""

            ax.text(
                0.5, -0.02, f"{fn_text}{gt_tag}\nscore: {score_text}",
                color="white" if not info["is_gt"] else GT_BORDER_COLOR,
                fontsize=8, ha="center", va="top",
                transform=ax.transAxes, family="monospace",
            )

            ax.set_xticks([])
            ax.set_yticks([])

            # Column header (only first aligner row)
            if i == 0:
                ax.set_title(COL_HEADERS[j], color="white", fontsize=10, fontweight="bold", pad=8)

            # Row label (only first column)
            if j == 0:
                ax.set_ylabel(label, color="white", fontsize=11, fontweight="bold", rotation=90, labelpad=12)

    ensure_dir(os.path.dirname(save_path) or ".")
    fig.savefig(save_path, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    print(f"[panel] Saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Panel visualization of query ablation top-1 results")
    parser.add_argument("--summary", default="results/sg_query_ablation/summary.json")
    parser.add_argument("--image_dir", default="F:/COYO/coyo/extracted/00000")
    parser.add_argument("--output", default="results/sg_query_ablation/panel_top1.png")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"[ERROR] summary.json nao encontrado: {args.summary}")
        sys.exit(1)

    with open(args.summary, "r", encoding="utf-8") as f:
        summary = json.load(f)

    make_panel(summary, args.image_dir, args.output)
