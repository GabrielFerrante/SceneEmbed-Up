from __future__ import annotations

import os
import sys
from typing import Iterable, Tuple

import h5py
import torch
from tqdm import tqdm

sys.path.append("..")

from data.data_utils_pytorch import create_all_dataloaders  # noqa: E402
from models.encoders.dinov3_extrator import DinoSceneEncoder  # noqa: E402
from models.encoders.qwen3_extrator import QwenSceneEmbedder  # noqa: E402
from utils.io_utils import ensure_dir, get_next_sample_index  # noqa: E402


def _validate_h5_integrity(path: str) -> None:
    """
    Valida integridade básica de um arquivo `.h5` recém escrito.

    Verifica existência e shape mínimo de:
    - `visual_feats`  -> `[N_patches, 768]`
    - `text_feats`    -> `[1, 4096]`
    """
    try:
        with h5py.File(path, "r") as f:
            assert "visual_feats" in f and "text_feats" in f, "Datasets obrigatórios ausentes."
            v_shape = f["visual_feats"].shape
            t_shape = f["text_feats"].shape
            if len(v_shape) != 2 or v_shape[1] != 768:
                raise ValueError(f"Shape inesperado para visual_feats: {v_shape}")
            if len(t_shape) != 2 or t_shape[1] != 4096:
                raise ValueError(f"Shape inesperado para text_feats: {t_shape}")
    except Exception as e:
        print(f"[WARN] Falha na validação de integridade do arquivo {path}: {e}")


def export_embeddings(
    dataloader: Iterable[Tuple[torch.Tensor, Iterable[str]]],
    dino: DinoSceneEncoder,
    qwen: QwenSceneEmbedder,
    output_dir: str,
) -> None:
    """
    Exporta embeddings de imagem/texto para arquivos `.h5` shardados.

    Shapes
    ------
    imgs:
        `[B, 3, H, W]`.
    globais:
        `[B, 768]` — token CLS do DINO.
    locais:
        `[B, 768, H_l, W_l]` — mapa de features espaciais.
    visual_feats (salvo):
        `[1024, 768]` — patches 32×32 achatados.
    text_feats (salvo):
        `[1, 4096]` — embedding textual do Qwen por legenda.
    """
    ensure_dir(output_dir)
    start_idx = get_next_sample_index(output_dir)
    print(f"Iniciando/Retomando do índice: {start_idx}")

    idx = 0
    samples_per_folder = 1000

    dino.model.eval()
    qwen.model.eval()

    with torch.no_grad():
        for imgs, texts in tqdm(dataloader, desc="Exportando"):
            batch_size = imgs.shape[0]

            if idx + batch_size <= start_idx:
                idx += batch_size
                continue

            try:
                globais, locais = dino.extract_features(imgs.to("cuda"))
                locais_small = torch.nn.functional.adaptive_avg_pool2d(locais, (32, 32))

                formatted_texts = [[t] for t in texts]
                t_feat = qwen.embed_components(formatted_texts, normalize=False)

                if t_feat.dim() == 2:
                    t_feat = t_feat.unsqueeze(1)

                for b in range(batch_size):
                    folder_idx = str(idx // samples_per_folder).zfill(5)
                    target_folder = ensure_dir(os.path.join(output_dir, folder_idx))

                    sample_filename = os.path.join(target_folder, f"sample_{idx}.h5")

                    feat_b = locais_small[b]
                    c, h, w = feat_b.shape
                    flat_feat = feat_b.reshape(c, -1).transpose(0, 1)

                    with h5py.File(sample_filename, "w") as f:
                        f.create_dataset(
                            "visual_feats",
                            data=flat_feat.cpu().numpy().astype("float16"),
                            compression="gzip",
                        )
                        f.create_dataset(
                            "text_feats",
                            data=t_feat[b].cpu().numpy().astype("float16"),
                            compression="gzip",
                        )
                        f.create_dataset(
                            "visual_global",
                            data=globais[b].cpu().numpy().astype("float16"),
                            compression="gzip",
                        )

                    _validate_h5_integrity(sample_filename)
                    idx += 1
            except KeyboardInterrupt:
                print(f"\nInterrupção detectada! Processamento parado no índice {idx}. Pode fechar agora.")
                return
            except RuntimeError as e:
                if "CUDA" in str(e).upper() and torch.cuda.is_available():
                    print(f"Erro de GPU durante exportação: {e}")
                    print(torch.cuda.memory_summary())
                raise


if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="anyup")
    qwen_embedder = QwenSceneEmbedder(device="cuda")

    train_dataloader, val_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="train"
    )
    test_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="test"
    )

    export_embeddings(val_dataloader, dino_encoder, qwen_embedder, output_dir="G:/coyo/embeds/val_anyup")
    export_embeddings(test_dataloader, dino_encoder, qwen_embedder, output_dir="G:/coyo/embeds/test_anyup")

