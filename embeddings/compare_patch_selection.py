"""
compare_patch_selection.py
---------------------------
Visualização rápida comparando os patches de uma mesma imagem em 5 cenários:

  A. Sem upsampler            — 196 patches LR nativos do DINOv3   (14x14)
  B. Com AnyUp                 — 1024 patches HR upsampled         (32x32)
  C. AnyUp + ATS (caption real)     — 196/1024 patches (AdaptiveTokenSampler)
  D. AnyUp + PCA (variância)        — 196/1024 patches (PCAVarianceSampler)
  E. AnyUp + ATS (caption aleatória) — 196/1024 patches (AdaptiveTokenSampler)

Não treina nenhum modelo: roda apenas o encoder DINOv3 (com/sem AnyUp) e
aplica as técnicas de seleção de patches já implementadas em
train_aligner_ats_h5.py / train_aligner_pca_h5.py.

Para o ATS, tenta carregar o sampler treinado de --ats_checkpoint
(default: checkpoints/best_ats_aligner.pth); se não existir, usa pesos
aleatórios (aviso impresso). Os painéis C e E usam o MESMO sampler treinado —
a única diferença é a query textual:
  - C usa o embedding real (Qwen3-Embedding-8B) da caption do par
    image-caption (--caption, ou auto-detectado em "<imagem>.txt")
  - E usa um vetor aleatório fixo (placeholder, sem relação com a imagem)

Comparar C vs E isola o efeito da query textual no ATS treinado, e C vs D
(painel de overlap) compara seleção aprendida vs critério clássico.

A 3ª linha da figura projeta para o grid nativo 14x14 (sem upsampler):
  - B → A: o grid de features do AnyUp (32x32) com average-pool para 14x14,
    visualizado em PCA-RGB — compara a representação densa do AnyUp (pooled)
    com a representação nativa do DINOv3 (painel A) na mesma resolução.
  - C/D/E → A: as máscaras de seleção (32x32) projetadas via nearest-neighbor,
    mostrando onde cada seleção cairia na resolução nativa.

Uso:
    python embeddings/compare_patch_selection.py --image_path embeddings/image1.jpg
    python embeddings/compare_patch_selection.py --image_path embeddings/image1.jpg --caption "a baseball player fielding a ground ball"
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from train_aligner_ats_h5 import AdaptiveTokenSampler
from train_aligner_pca_h5 import PCAVarianceSampler
from utils.io_utils import ensure_dir


_IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

K_PATCHES = 196  # mesmo K usado em train_aligner_ats_h5.py / train_aligner_pca_h5.py
ANYUP_GRID = 32  # 32x32 = 1024 patches (igual generate_shards.py)
NOUP_GRID = 14   # 14x14 = 196 patches


# ---------------------------------------------------------------------------
# PCA-RGB para visualização
# ---------------------------------------------------------------------------

def patches_to_pca_rgb(patches_hw: np.ndarray) -> np.ndarray:
    """Projeta patches [H, W, D] em 3 componentes PCA e normaliza para RGB [0, 1]."""
    H, W, D = patches_hw.shape
    flat = patches_hw.reshape(-1, D)
    mean = flat.mean(axis=0, keepdims=True)
    centered = flat - mean

    cov = centered.T @ centered / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    basis = eigvecs[:, -3:][:, ::-1].copy()  # [D, 3] desc

    proj = centered @ basis  # [H*W, 3]
    lo = proj.min(axis=0, keepdims=True)
    hi = proj.max(axis=0, keepdims=True)
    rng = np.maximum(hi - lo, 1e-12)
    return ((proj - lo) / rng).reshape(H, W, 3)


def resize_mask(mask: np.ndarray, out_size: int) -> np.ndarray:
    """Redimensiona uma máscara booleana [gh, gw] para [out_size, out_size] (nearest + threshold 0.5)."""
    t = torch.from_numpy(mask.astype(np.float32))[None, None]
    resized = torch.nn.functional.interpolate(t, size=(out_size, out_size), mode="nearest")
    return resized[0, 0].numpy() > 0.5


def overlay_selection_mask(image_arr: np.ndarray, mask_grid: np.ndarray) -> np.ndarray:
    """
    Escurece regiões não-selecionadas da imagem para evidenciar a seleção.

    image_arr : [H, W, 3] em [0, 1]
    mask_grid : [gh, gw] bool — True = patch selecionado
    """
    H, W = image_arr.shape[:2]
    t = torch.from_numpy(mask_grid.astype(np.float32))[None, None]
    mask_full = torch.nn.functional.interpolate(t, size=(H, W), mode="nearest")[0, 0].numpy()[..., None]
    return image_arr * (0.2 + 0.8 * mask_full)


def overlap_rgb_map(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Mapa de sobreposição entre duas máscaras booleanas [gh, gw].

    Retorna (rgb [gh, gw, 3], jaccard).
    Verde = só A | Azul = só B | Amarelo = ambos | Cinza-escuro = nenhum.
    """
    both     = mask_a & mask_b
    only_a   = mask_a & ~mask_b
    only_b   = mask_b & ~mask_a

    rgb = np.full((*mask_a.shape, 3), 0.08, dtype=np.float32)
    rgb[only_a] = [0.20, 0.85, 0.30]
    rgb[only_b] = [0.25, 0.55, 1.00]
    rgb[both]   = [1.00, 0.85, 0.10]

    jaccard = both.sum() / max((mask_a | mask_b).sum(), 1)
    return rgb, float(jaccard)


# ---------------------------------------------------------------------------
# Extração dos cenários
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_scenarios(image: Image.Image, device: str, ats_checkpoint: str | None, caption: str | None = None) -> dict:
    img_t = _IMG_TRANSFORM(image).unsqueeze(0).to(device)

    # ── A. Sem upsampler — 196 patches (14x14) ──────────────────────────────
    dino_lr = DinoSceneEncoder(device=device, upsampler="none")
    _, lr_feat = dino_lr.extract_features(img_t)               # [1, 768, 14, 14]
    noup_grid = lr_feat.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # [14,14,768]
    del dino_lr, lr_feat
    torch.cuda.empty_cache()

    # ── B. Com AnyUp — pool para 1024 patches (32x32) ───────────────────────
    dino_hr = DinoSceneEncoder(device=device, upsampler="anyup")
    _, hr_feat = dino_hr.extract_features(img_t)                # [1, 768, H, W]
    hr_small = torch.nn.functional.adaptive_avg_pool2d(hr_feat, (ANYUP_GRID, ANYUP_GRID))
    anyup_grid = hr_small.squeeze(0).permute(1, 2, 0).float().cpu().numpy()  # [32,32,768]
    anyup_flat = hr_small.flatten(2).permute(0, 2, 1).float()   # [1, 1024, 768]
    del dino_hr, hr_feat, hr_small
    torch.cuda.empty_cache()

    # ── Sampler ATS (mesmo para os painéis C e E) ───────────────────────────
    ats_sampler = AdaptiveTokenSampler(visual_dim=768, text_dim=4096, k=K_PATCHES).to(device)
    if ats_checkpoint and os.path.exists(ats_checkpoint):
        state = torch.load(ats_checkpoint, map_location=device)
        sampler_state = {k[len("sampler."):]: v for k, v in state.items() if k.startswith("sampler.")}
        ats_sampler.load_state_dict(sampler_state)
        ats_sampler = ats_sampler.float()
        print(f"[ATS] sampler treinado carregado de {ats_checkpoint}")
    else:
        print(f"[ATS] checkpoint não encontrado ({ats_checkpoint}) — usando pesos aleatórios")
    ats_sampler.eval()

    # ── C. Query real (caption do par image-caption) ────────────────────────
    if caption:
        qwen = QwenSceneEmbedder(device=device)
        text_query_real = qwen.embed_components([caption]).to(device)  # [1, 1, 4096]
        print(f"[ATS] painel C — query textual real: '{caption}'")
        del qwen
        torch.cuda.empty_cache()
    else:
        print("[ATS] painel C — nenhuma caption disponível, usando query aleatória (seed=42)")
        torch.manual_seed(42)
        text_query_real = F.normalize(torch.randn(1, 1, 4096, device=device), p=2, dim=-1)

    # ── E. Query aleatória (placeholder, sem relação com a imagem) ──────────
    # Normalizada (norma=1) para igualar a magnitude do embedding Qwen3 real
    # (embed_components usa normalize=True) — sem isso, text_score teria
    # escalas muito diferentes entre C e E e a comparação não seria justa.
    torch.manual_seed(123)
    text_query_random = F.normalize(torch.randn(1, 1, 4096, device=device), p=2, dim=-1)

    _, _, ats_idx_real   = ats_sampler(anyup_flat.to(device), text_query_real)
    _, _, ats_idx_random = ats_sampler(anyup_flat.to(device), text_query_random)

    # ── D. AnyUp + PCA (variância) ───────────────────────────────────────────
    pca_sampler = PCAVarianceSampler(k=K_PATCHES, n_components=64)
    _, _, pca_idx = pca_sampler(anyup_flat.to(device))

    return {
        "noup_grid": noup_grid,
        "anyup_grid": anyup_grid,
        "ats_idx": ats_idx_real.squeeze(0).cpu().numpy(),
        "ats_idx_random": ats_idx_random.squeeze(0).cpu().numpy(),
        "pca_idx": pca_idx.squeeze(0).cpu().numpy(),
    }


def idx_to_mask(idx: np.ndarray, grid_size: int) -> np.ndarray:
    mask = np.zeros(grid_size * grid_size, dtype=bool)
    mask[idx] = True
    return mask.reshape(grid_size, grid_size)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def make_figure(image: Image.Image, data: dict, save_path: str, caption: str | None = None) -> None:
    np_img = np.array(image.resize((256, 256))).astype(np.float32) / 255.0

    rgb_noup = patches_to_pca_rgb(data["noup_grid"])
    rgb_anyup = patches_to_pca_rgb(data["anyup_grid"])

    ats_mask_real = idx_to_mask(data["ats_idx"], ANYUP_GRID)
    ats_mask_rand = idx_to_mask(data["ats_idx_random"], ANYUP_GRID)
    pca_mask = idx_to_mask(data["pca_idx"], ANYUP_GRID)

    overlap_ap_rgb, jaccard_ap = overlap_rgb_map(ats_mask_real, pca_mask)
    overlap_ar_rgb, jaccard_ar = overlap_rgb_map(ats_mask_real, ats_mask_rand)

    # Projeção das máscaras (32x32, espaço AnyUp) para o grid nativo (14x14, sem upsampler)
    ats_mask_real_14 = resize_mask(ats_mask_real, NOUP_GRID)
    pca_mask_14       = resize_mask(pca_mask,       NOUP_GRID)
    ats_mask_rand_14 = resize_mask(ats_mask_rand, NOUP_GRID)

    # B → A: grid de features do AnyUp (32x32) pooled para 14x14, em PCA-RGB
    anyup_t = torch.from_numpy(data["anyup_grid"]).permute(2, 0, 1)[None]  # [1,768,32,32]
    anyup_grid_14 = torch.nn.functional.adaptive_avg_pool2d(anyup_t, (NOUP_GRID, NOUP_GRID))
    rgb_anyup_14 = patches_to_pca_rgb(anyup_grid_14[0].permute(1, 2, 0).numpy())

    fig, axes = plt.subplots(3, 5, figsize=(25, 15), facecolor="#1a1a2e")

    caption_title = f"\n\"{caption[:50]}{'...' if caption and len(caption) > 50 else ''}\"" if caption else "\n(sem caption — query aleatória)"

    # Row 1
    axes[0, 0].imshow(np_img)
    axes[0, 0].set_title("Original image (256x256)", color="white", fontsize=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(rgb_noup, interpolation="nearest")
    axes[0, 1].set_title(f"A. Without upsampler\n{NOUP_GRID}x{NOUP_GRID} = 196 patches", color="white", fontsize=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(rgb_anyup, interpolation="nearest")
    axes[0, 2].set_title(f"B. AnyUp (pool 32x32)\n{ANYUP_GRID}x{ANYUP_GRID} = 1024 patches", color="white", fontsize=10)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(overlay_selection_mask(np_img, ats_mask_real))
    axes[0, 3].set_title(f"C. AnyUp + ATS (real caption)\n{ats_mask_real.sum()}/1024 selected{caption_title}", color="white", fontsize=9)
    axes[0, 3].axis("off")

    axes[0, 4].axis("off")

    # Row 2
    axes[1, 0].imshow(overlay_selection_mask(np_img, pca_mask))
    axes[1, 0].set_title(f"D. AnyUp + PCA (variance)\n{pca_mask.sum()}/1024 selected", color="white", fontsize=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(overlay_selection_mask(np_img, ats_mask_rand))
    axes[1, 1].set_title(f"E. AnyUp + ATS (random caption)\n{ats_mask_rand.sum()}/1024 selected", color="white", fontsize=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(overlap_ap_rgb, interpolation="nearest")
    axes[1, 2].set_title(
        f"Overlap C vs D (ATS vs PCA)\n"
        f"Green=only ATS | Blue=only PCA | Yellow=Both\nJaccard={jaccard_ap:.3f}",
        color="white", fontsize=9,
    )
    axes[1, 2].axis("off")

    axes[1, 3].imshow(overlap_ar_rgb, interpolation="nearest")
    axes[1, 3].set_title(
        f"Overlap C vs E (real vs random caption)\n"
        f"Green=only real | Blue=only random | Yellow=Both\nJaccard={jaccard_ar:.3f}",
        color="white", fontsize=9,
    )
    axes[1, 3].axis("off")

    axes[1, 4].axis("off")

    # Row 3 — A/B/C/D/E projetados no grid nativo 14x14 (sem upsampler)
    axes[2, 0].imshow(rgb_noup, interpolation="nearest")
    axes[2, 0].set_title(f"A. Without upsampler (reference)\n{NOUP_GRID}x{NOUP_GRID} = 196 patches", color="white", fontsize=10)
    axes[2, 0].axis("off")

    axes[2, 1].imshow(rgb_anyup_14, interpolation="nearest")
    axes[2, 1].set_title(f"B → A: AnyUp pooled\nProjection for{NOUP_GRID}x{NOUP_GRID}", color="white", fontsize=10)
    axes[2, 1].axis("off")

    axes[2, 2].imshow(overlay_selection_mask(np_img, ats_mask_real_14))
    axes[2, 2].set_title(f"C → A: ATS (real caption)\nProjection for {NOUP_GRID}x{NOUP_GRID}", color="white", fontsize=10)
    axes[2, 2].axis("off")

    axes[2, 3].imshow(overlay_selection_mask(np_img, pca_mask_14))
    axes[2, 3].set_title(f"D → A: PCA (variance)\nProjection for {NOUP_GRID}x{NOUP_GRID}", color="white", fontsize=10)
    axes[2, 3].axis("off")

    axes[2, 4].imshow(overlay_selection_mask(np_img, ats_mask_rand_14))
    axes[2, 4].set_title(f"E → A: ATS (random caption)\nProjection for {NOUP_GRID}x{NOUP_GRID}", color="white", fontsize=10)
    axes[2, 4].axis("off")

    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path) or ".")
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _resolve_caption(image_path: str, explicit_caption: str | None) -> str | None:
    """Usa --caption se fornecida; senão tenta carregar '<imagem>.txt' ao lado da imagem."""
    if explicit_caption:
        return explicit_caption
    txt_path = os.path.splitext(image_path)[0] + ".txt"
    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Noup vs anyup vs anyup+ATS(real) vs anyup+PCA vs anyup+ATS(random)")
    parser.add_argument("--image_path", default="embeddings/image1.jpg", type=str)
    parser.add_argument("--output_dir", default="embeddings/results/patch_selection_compare", type=str)
    parser.add_argument("--ats_checkpoint", default="checkpoints/best_ats_aligner.pth", type=str)
    parser.add_argument("--caption", default=None, type=str, help="Caption usada como query textual real do ATS (Qwen3-Embedding-8B). Se omitida, tenta '<imagem>.txt'")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Imagem não encontrada: {args.image_path}")

    print(f"[load] image: {args.image_path}")
    image = Image.open(args.image_path).convert("RGB")

    caption = _resolve_caption(args.image_path, args.caption)
    print(f"[load] caption: {caption!r}")

    print(f"\n[extract] Rodando os cenários em device={args.device}...")
    data = extract_scenarios(image, device=args.device, ats_checkpoint=args.ats_checkpoint, caption=caption)

    base_name = os.path.splitext(os.path.basename(args.image_path))[0]
    save_path = os.path.join(args.output_dir, f"compare_{base_name}.png")
    make_figure(image, data, save_path, caption=caption)

    print(f"\n[done] Comparação salva em {save_path}")
