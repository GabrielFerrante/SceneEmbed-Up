"""
compare_anyup_visualization.py
------------------------------
Comparação qualitativa entre extrações COM e SEM AnyUp para uma única imagem.

Para a mesma imagem COYO, executa o DINOv3 em dois modos:
  1. upsampler="none"  → patches LR nativos  [196, 768] (14x14)
  2. upsampler="anyup" → patches HR upsampled [B, 768, H_hr, W_hr]

Gera uma figura com 4 painéis comparativos:
  A. Imagem original
  B. PCA-RGB dos patches LR sobreposto na imagem (sem AnyUp)
  C. PCA-RGB dos patches HR sobreposto na imagem (com AnyUp)
  D. Lado a lado das normas L2 por patch como heatmap

A projeção PCA leva os patches 768-d → 3 componentes principais, depois
normaliza para [0, 1] e usa como RGB. Patches semanticamente similares
ficam com cores parecidas — permite ver o quão fina é a granularidade que
cada modo captura.

Uso:
    python embeddings/compare_anyup_visualization.py --image_path image.jpg
    python embeddings/compare_anyup_visualization.py --image_path foo.jpg --image_size 256
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.encoders.dinov3_extrator import DinoSceneEncoder
from utils.io_utils import ensure_dir


_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------------------------------------------------------------------------
# PCA para visualização RGB
# ---------------------------------------------------------------------------

def patches_to_pca_rgb(
    patches_hw: np.ndarray,
    pca_basis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Projeta patches [H, W, D] em 3 componentes PCA e normaliza para RGB [0, 1].

    Parameters
    ----------
    patches_hw : np.ndarray
        Patches reshapeados como grade espacial `[H, W, D]`.
    pca_basis : np.ndarray | None
        Se fornecido, usa esses 3 vetores como projeção (compartilha base
        entre múltiplas imagens). Caso contrário, calcula PCA local.

    Returns
    -------
    rgb : `[H, W, 3]` array em [0, 1]
    basis : `[D, 3]` componentes principais usados
    """
    H, W, D = patches_hw.shape
    flat = patches_hw.reshape(-1, D)                       # [H*W, D]

    # Centraliza
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    if pca_basis is None:
        # SVD via covariance compacto (D~768 cabe em qualquer máquina)
        cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
        # Top-3 autovetores
        eigvals, eigvecs = np.linalg.eigh(cov)
        basis = eigvecs[:, -3:][:, ::-1].copy()             # [D, 3] desc
    else:
        basis = pca_basis

    proj = centered @ basis                                # [H*W, 3]

    # Normaliza cada canal para [0, 1] independentemente (estética padrão p/ feature vis)
    lo = proj.min(axis=0, keepdims=True)
    hi = proj.max(axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-12)
    rgb = ((proj - lo) / rng).reshape(H, W, 3)

    return rgb, basis


def patches_to_norm_heatmap(patches_hw: np.ndarray) -> np.ndarray:
    """
    Calcula a norma L2 de cada patch e devolve `[H, W]` normalizado em [0, 1].

    Patches com norma alta sinalizam regiões "interessantes" para o modelo.
    """
    norms = np.linalg.norm(patches_hw, axis=-1)            # [H, W]
    lo, hi = norms.min(), norms.max()
    rng = max(hi - lo, 1e-12)
    return (norms - lo) / rng


# ---------------------------------------------------------------------------
# Extração com e sem AnyUp
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_both(image: Image.Image, device: str) -> dict:
    """
    Extrai features com upsampler='none' e upsampler='anyup' para a mesma imagem.

    Returns
    -------
    dict com:
        noup_patches  : [H_lr, W_lr, 768]  numpy float32
        noup_cls      : [768]              numpy float32
        anyup_patches : [H_hr, W_hr, 768]  numpy float32
        anyup_cls     : [768]              numpy float32
    """
    img_t = _IMG_TRANSFORM(image).unsqueeze(0).to(device)

    # ── sem upsampler ────────────────────────────────────────────────────
    dino_lr = DinoSceneEncoder(device=device, upsampler="none")
    cls_lr, lr_feat = dino_lr.extract_features(img_t)         # [1,768,h,w]
    noup_patches = lr_feat.squeeze(0).cpu().float().numpy()   # [768, h, w]
    noup_patches = noup_patches.transpose(1, 2, 0)             # [h, w, 768]
    noup_cls = cls_lr.squeeze(0).cpu().float().numpy()

    # Libera VRAM
    del dino_lr, cls_lr, lr_feat
    torch.cuda.empty_cache()

    # ── com AnyUp ────────────────────────────────────────────────────────
    dino_hr = DinoSceneEncoder(device=device, upsampler="anyup")
    cls_hr, hr_feat = dino_hr.extract_features(img_t)         # [1,768,H,W]
    anyup_patches = hr_feat.squeeze(0).cpu().float().numpy()
    anyup_patches = anyup_patches.transpose(1, 2, 0)
    anyup_cls = cls_hr.squeeze(0).cpu().float().numpy()

    del dino_hr, cls_hr, hr_feat
    torch.cuda.empty_cache()

    return {
        "noup_patches": noup_patches,
        "noup_cls": noup_cls,
        "anyup_patches": anyup_patches,
        "anyup_cls": anyup_cls,
    }


# ---------------------------------------------------------------------------
# Comparação CLS
# ---------------------------------------------------------------------------

def cls_diagnostics(cls_a: np.ndarray, cls_b: np.ndarray, label_a: str, label_b: str) -> dict:
    """Estatísticas de similaridade entre os CLS tokens (global descriptors)."""
    ca = cls_a / max(np.linalg.norm(cls_a), 1e-12)
    cb = cls_b / max(np.linalg.norm(cls_b), 1e-12)
    cos = float(ca @ cb)
    l2 = float(np.linalg.norm(cls_a - cls_b))
    return {
        f"norm_{label_a}": float(np.linalg.norm(cls_a)),
        f"norm_{label_b}": float(np.linalg.norm(cls_b)),
        "cosine_sim":     cos,
        "l2_distance":    l2,
    }


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def make_comparison_figure(
    image: Image.Image,
    data: dict,
    save_path: str,
    cls_stats: dict,
) -> None:
    """
    Figura 2x3 comparativa:

        Row 1: imagem original (col 0) | PCA-RGB noup (col 1) | PCA-RGB anyup (col 2)
        Row 2: (vazio)                 | norm heatmap noup   | norm heatmap anyup
    """
    np_img = np.array(image.resize((256, 256)))

    noup_p = data["noup_patches"]                              # [h_lr, w_lr, 768]
    anyup_p = data["anyup_patches"]                            # [h_hr, w_hr, 768]

    # PCA — usamos base independente para cada um (cada modo tem distribuição própria)
    rgb_noup, _ = patches_to_pca_rgb(noup_p)
    rgb_anyup, _ = patches_to_pca_rgb(anyup_p)

    norm_noup = patches_to_norm_heatmap(noup_p)
    norm_anyup = patches_to_norm_heatmap(anyup_p)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="#1a1a2e")

    # Row 1
    axes[0, 0].imshow(np_img)
    axes[0, 0].set_title("Original image (256x256)", color="white", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_noup, interpolation="nearest")
    axes[0, 1].set_title(
        f"PCA-RGB — NO upsampler\nshape {noup_p.shape[0]}x{noup_p.shape[1]} patches",
        color="white", fontsize=10,
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_anyup, interpolation="nearest")
    axes[0, 2].set_title(
        f"PCA-RGB — AnyUp\nshape {anyup_p.shape[0]}x{anyup_p.shape[1]} patches",
        color="white", fontsize=10,
    )
    axes[0, 2].axis("off")

    # Row 2 — heatmaps
    axes[1, 0].axis("off")
    axes[1, 0].text(
        0.5, 0.5,
        "CLS comparison\n\n"
        f"cosine sim: {cls_stats['cosine_sim']:.4f}\n"
        f"L2 distance: {cls_stats['l2_distance']:.3f}\n\n"
        f"norm noup:  {cls_stats['norm_noup']:.3f}\n"
        f"norm anyup: {cls_stats['norm_anyup']:.3f}\n\n"
        f"Patches:\n"
        f"  noup:  {noup_p.shape[0]*noup_p.shape[1]} ({noup_p.shape[0]}x{noup_p.shape[1]})\n"
        f"  anyup: {anyup_p.shape[0]*anyup_p.shape[1]} ({anyup_p.shape[0]}x{anyup_p.shape[1]})",
        ha="center", va="center",
        color="white", fontsize=11,
        transform=axes[1, 0].transAxes,
        family="monospace",
    )

    im_lo = axes[1, 1].imshow(norm_noup, cmap="inferno", interpolation="nearest")
    axes[1, 1].set_title("L2 norm per patch — NO upsampler", color="white", fontsize=10)
    axes[1, 1].axis("off")
    plt.colorbar(im_lo, ax=axes[1, 1], fraction=0.046)

    im_hi = axes[1, 2].imshow(norm_anyup, cmap="inferno", interpolation="nearest")
    axes[1, 2].set_title("L2 norm per patch — AnyUp", color="white", fontsize=10)
    axes[1, 2].axis("off")
    plt.colorbar(im_hi, ax=axes[1, 2], fraction=0.046)

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comparação qualitativa AnyUp vs LR nativo para uma imagem COYO"
    )
    parser.add_argument("--image_path", required=True, type=str,
                        help="Caminho da imagem (ex: F:/COYO/coyo/extracted/00000/abc.jpg)")
    parser.add_argument("--output_dir", default="results/anyup_compare", type=str)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {args.image_path}")

    print(f"[load] image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")
    print(f"  size original: {image.size}")

    print(f"\n[extract] Running DINOv3 with and without AnyUp on device={args.device}...")
    data = extract_both(image, device=args.device)
    print(f"  patches no-up : shape {data['noup_patches'].shape}")
    print(f"  patches anyup : shape {data['anyup_patches'].shape}")

    print(f"\n[cls] Comparing CLS tokens (global descriptors)...")
    cls_stats = cls_diagnostics(data["noup_cls"], data["anyup_cls"], "noup", "anyup")
    for k, v in cls_stats.items():
        print(f"  {k:<14} = {v:.4f}")

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"compare_{base_name}.png")
    print(f"\n[viz] Generating comparison figure...")
    make_comparison_figure(image, data, save_path, cls_stats)

    print(f"\n[done] Visual comparison saved to {save_path}")
