"""
generate_shards_no_up.py
------------------------
Gera embeddings pré-computados em shards .h5 SEM upsampler (AnyUp/FeatUp).
Usa patches LR nativos do DINOv3 — o processor redimensiona para 224×224,
gerando grid 14×14 com patch 16: [B, 196, 768].

Shapes de saída por shard (samples_per_shard=5000):
    visual_feats  : [5000, 196, 768]   float16  — patches LR do DINO (sem upsampling)
    text_feats    : [5000, 1,   4096]  float16  — embedding textual do Qwen
    visual_global : [5000,      768]   float16  — token CLS do DINO
    image_paths   : [5000]             string   — caminho absoluto da imagem origem

Vantagens vs generate_shards.py (com AnyUp):
- ~4× menor em disco (256 patches vs 1024)
- ~2-3× mais rápido (sem AnyUp na extração)
- Menos VRAM/RAM no carregamento

Uso:
    python embeddings/generate_shards_no_up.py

Retomada automática:
    Se interrompido, basta rodar novamente — o ShardWriter detecta
    o último shard incompleto e continua de onde parou.
"""

from __future__ import annotations

import os
import sys
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from data.data_utils_pytorch import create_all_dataloaders
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from utils.io_utils import ensure_dir

# ---------------------------------------------------------------------------
# Constantes de shape — diferem do generate_shards.py (1024 → 256 patches)
# ---------------------------------------------------------------------------
VISUAL_DIM     = 768   # dim do patch DINO
VISUAL_PATCHES = 196   # 14×14 LR (DINOv3 processor resize para 224, patch 16)
TEXT_DIM       = 4096  # dim do Qwen
GLOBAL_DIM     = 768   # dim do CLS token DINO


# ---------------------------------------------------------------------------
# Helpers de shard
# ---------------------------------------------------------------------------

def _shard_path(output_dir: str, shard_idx: int) -> str:
    """Retorna o path de um shard pelo índice."""
    return os.path.join(output_dir, f"shard_{str(shard_idx).zfill(6)}.h5")


def _get_resume_state(output_dir: str) -> Tuple[int, int]:
    """
    Detecta o último shard existente para retomada após crash.

    Returns
    -------
    (shard_idx, n_samples_no_shard_atual)
    """
    existing = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0, 0

    last      = existing[-1]
    shard_idx = int(last.replace("shard_", "").replace(".h5", ""))
    path      = _shard_path(output_dir, shard_idx)

    try:
        with h5py.File(path, "r") as f:
            n = f["visual_feats"].shape[0]
    except Exception:
        n = 0

    return shard_idx, n


def _count_global_samples(output_dir: str, samples_per_shard: int) -> int:
    """Conta o total de amostras já exportadas para retomada."""
    existing = sorted(
        f for f in os.listdir(output_dir)
        if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0

    total = (len(existing) - 1) * samples_per_shard

    path = os.path.join(output_dir, existing[-1])
    try:
        with h5py.File(path, "r") as f:
            total += f["visual_feats"].shape[0]
    except Exception:
        pass

    return total


# ---------------------------------------------------------------------------
# Validação de shard
# ---------------------------------------------------------------------------

def _validate_shard(path: str, expected_n: Optional[int] = None) -> None:
    """
    Valida integridade de um shard após escrita.

    Verifica shapes esperados:
        visual_feats  : [N, 196, 768]
        text_feats    : [N, 1,   4096]
        visual_global : [N,      768]   (opcional)
    """
    try:
        with h5py.File(path, "r") as f:

            for key in ("visual_feats", "text_feats"):
                assert key in f, f"Dataset '{key}' ausente."

            v_shape = f["visual_feats"].shape
            t_shape = f["text_feats"].shape
            n       = v_shape[0]

            assert t_shape[0] == n, (
                f"Eixo-0 inconsistente: visual={n}, text={t_shape[0]}"
            )

            assert len(v_shape) == 3 and v_shape[1] == VISUAL_PATCHES and v_shape[2] == VISUAL_DIM, \
                f"Shape inesperado para visual_feats: {v_shape}"
            assert len(t_shape) == 3 and t_shape[2] == TEXT_DIM, \
                f"Shape inesperado para text_feats: {t_shape}"

            if "visual_global" in f:
                g_shape = f["visual_global"].shape
                assert g_shape[0] == n and g_shape[1] == GLOBAL_DIM, \
                    f"Shape inesperado para visual_global: {g_shape}"

            if "image_paths" in f:
                p_shape = f["image_paths"].shape
                assert p_shape[0] == n, \
                    f"Eixo-0 inconsistente em image_paths: {p_shape[0]} vs {n}"

            if expected_n is not None:
                assert n == expected_n, f"Esperava {expected_n} amostras, encontrou {n}."

        print(f"  [OK] {os.path.basename(path)} — {n} amostras, shape visual={v_shape}")

    except Exception as e:
        print(f"  [WARN] Validation failed for {path}: {e}")


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Escreve amostras em shards .h5 com datasets resizáveis (versão sem AnyUp).

    Cada shard acumula até `samples_per_shard` amostras e então rotaciona
    automaticamente para o próximo arquivo. Suporta retomada após crash.

    Shapes escritos por shard:
        visual_feats  : [N, 196, 768]   float16
        text_feats    : [N, 1,   4096]  float16
        visual_global : [N,      768]   float16
        image_paths   : [N]             vlen str
    """

    def __init__(
        self,
        output_dir: str,
        samples_per_shard: int = 5_000,
        resume: bool = True,
    ):
        self.output_dir        = output_dir
        self.samples_per_shard = samples_per_shard

        ensure_dir(output_dir)

        if resume:
            self._shard_idx, self._in_shard = _get_resume_state(output_dir)
        else:
            self._shard_idx, self._in_shard = 0, 0

        self._file      : Optional[h5py.File]    = None
        self._ds_visual : Optional[h5py.Dataset] = None
        self._ds_text   : Optional[h5py.Dataset] = None
        self._ds_global : Optional[h5py.Dataset] = None
        self._ds_paths  : Optional[h5py.Dataset] = None

        if self._in_shard > 0:
            self._open_shard(new=False)

        self.total_written = _count_global_samples(output_dir, samples_per_shard)
        print(
            f"[ShardWriter] shard={self._shard_idx} | "
            f"amostras no shard atual={self._in_shard} | "
            f"total já exportado={self.total_written:,}"
        )

    # ------------------------------------------------------------------
    # Gerenciamento de arquivo
    # ------------------------------------------------------------------

    def _open_shard(self, new: bool = True) -> None:
        """Abre ou cria um shard."""
        if self._file is not None:
            self._close_shard()

        path = _shard_path(self.output_dir, self._shard_idx)

        if new or not os.path.exists(path):
            self._file = h5py.File(path, "w")
            self._ds_visual = self._file.create_dataset(
                "visual_feats",
                shape=(0, VISUAL_PATCHES, VISUAL_DIM),
                maxshape=(None, VISUAL_PATCHES, VISUAL_DIM),
                dtype="float16",
                compression="gzip",
                chunks=(64, VISUAL_PATCHES, VISUAL_DIM),
            )
            self._ds_text = self._file.create_dataset(
                "text_feats",
                shape=(0, 1, TEXT_DIM),
                maxshape=(None, 1, TEXT_DIM),
                dtype="float16",
                compression="gzip",
                chunks=(64, 1, TEXT_DIM),
            )
            self._ds_global = self._file.create_dataset(
                "visual_global",
                shape=(0, GLOBAL_DIM),
                maxshape=(None, GLOBAL_DIM),
                dtype="float16",
                compression="gzip",
                chunks=(256, GLOBAL_DIM),
            )
            self._ds_paths = self._file.create_dataset(
                "image_paths",
                shape=(0,),
                maxshape=(None,),
                dtype=h5py.string_dtype(encoding="utf-8"),
                chunks=(256,),
            )
        else:
            self._file      = h5py.File(path, "a")
            self._ds_visual = self._file["visual_feats"]
            self._ds_text   = self._file["text_feats"]
            self._ds_global = self._file["visual_global"]
            self._ds_paths  = self._file["image_paths"] if "image_paths" in self._file else None

    def _close_shard(self) -> None:
        """Fecha e valida o shard atual."""
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            _validate_shard(_shard_path(self.output_dir, self._shard_idx))

    def _rotate_shard(self) -> None:
        """Fecha shard cheio e abre o próximo."""
        self._close_shard()
        self._shard_idx += 1
        self._in_shard   = 0
        self._open_shard(new=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def write_batch(
        self,
        visual_feats: np.ndarray,   # [B, 196, 768]  float16
        text_feats  : np.ndarray,   # [B, 1,   4096] float16
        global_feats: np.ndarray,   # [B,      768]  float16
        image_paths : list[str],    # [B] caminhos absolutos das imagens
    ) -> None:
        """Escreve um batch, quebrando em sub-lotes se ultrapassar o shard."""
        b      = visual_feats.shape[0]
        offset = 0

        while offset < b:
            if self._file is None:
                self._open_shard(new=(self._in_shard == 0))

            slots    = self.samples_per_shard - self._in_shard
            n        = min(slots, b - offset)
            end      = offset + n
            new_size = self._in_shard + n

            self._ds_visual.resize(new_size, axis=0)
            self._ds_text.resize(new_size, axis=0)
            self._ds_global.resize(new_size, axis=0)

            self._ds_visual[self._in_shard:new_size] = visual_feats[offset:end]
            self._ds_text  [self._in_shard:new_size] = text_feats  [offset:end]
            self._ds_global[self._in_shard:new_size] = global_feats[offset:end]

            if self._ds_paths is not None:
                self._ds_paths.resize(new_size, axis=0)
                self._ds_paths[self._in_shard:new_size] = image_paths[offset:end]

            self._in_shard     = new_size
            self.total_written += n
            offset             = end

            if self._in_shard >= self.samples_per_shard:
                self._rotate_shard()

    def close(self) -> None:
        self._close_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Exportação principal
# ---------------------------------------------------------------------------

def export_embeddings_sharded(
    dataloader        : Iterable[Tuple[torch.Tensor, Iterable[str]]],
    dino              : DinoSceneEncoder,
    qwen              : QwenSceneEmbedder,
    output_dir        : str,
    samples_per_shard : int  = 5_000,
    resume            : bool = True,
) -> None:
    """
    Extrai e salva embeddings em shards .h5 (sem upsampling).

    Shapes gerados:
        visual_feats  : [samples_per_shard, 196, 768]
        text_feats    : [samples_per_shard, 1,   4096]
        visual_global : [samples_per_shard,      768]
        image_paths   : [samples_per_shard]               vlen utf-8

    Parameters
    ----------
    dataloader:
        Emite (imgs [B,3,H,W], texts [B], paths [B]).
    dino:
        Encoder visual DINOv3 (deve estar instanciado com upsampler="none").
    qwen:
        Embedder textual Qwen.
    output_dir:
        Diretório de saída.
    samples_per_shard:
        Amostras por arquivo .h5 (padrão: 5.000).
    resume:
        Se True, continua de onde parou após crash.
    """
    dino.model.eval()
    qwen.model.eval()

    with ShardWriter(output_dir, samples_per_shard=samples_per_shard, resume=resume) as writer:
        skipped    = 0
        global_idx = writer.total_written

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Exportando → {output_dir}"):
                # CoyoCollate sem tokenizer retorna (imgs, captions, paths)
                if len(batch) == 3:
                    imgs, texts, paths = batch
                else:
                    imgs, texts = batch
                    paths = [""] * imgs.shape[0]

                batch_size = imgs.shape[0]

                # ── Skip de batches já exportados ────────────────────────
                if skipped < global_idx:
                    remaining = global_idx - skipped
                    if batch_size <= remaining:
                        skipped += batch_size
                        continue
                    imgs       = imgs[remaining:]
                    texts      = list(texts)[remaining:]
                    paths      = list(paths)[remaining:]
                    skipped   += remaining
                    batch_size = imgs.shape[0]

                try:
                    # ── Extração de features (LR direto, sem upsampling) ──
                    globais, patches = dino.extract_patches_seq(imgs.to("cuda"))
                    # patches: [B, 196, 768]

                    # Embeddings textuais [B, 1, 4096]
                    t_feat = qwen.embed_components([[t] for t in texts], normalize=False)
                    if t_feat.dim() == 2:
                        t_feat = t_feat.unsqueeze(1)

                    # ── Conversão para numpy float16 ──────────────────────
                    v_np = patches.cpu().numpy().astype("float16")   # [B, 196, 768]
                    t_np = t_feat.cpu().numpy().astype("float16")    # [B, 1,   4096]
                    g_np = globais.cpu().numpy().astype("float16")   # [B,      768]

                    writer.write_batch(v_np, t_np, g_np, list(paths))

                except KeyboardInterrupt:
                    print(f"\nInterrupted! Total exported: {writer.total_written:,}. "
                          "Safe to close.")
                    return

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(f"GPU error: {e}")
                        print(torch.cuda.memory_summary())
                    raise

    print(f"\nExport complete — {writer.total_written:,} samples in {output_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="none")
    qwen_embedder = QwenSceneEmbedder(device="cuda")

    # Uma varredura única do dataset — random_split com seed=42 é determinístico
    train_dl, val_dl, test_dl = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=8, num_workers=8, t="all"
    )
    """
    export_embeddings_sharded(
        train_dl, dino_encoder, qwen_embedder,
        output_dir        = "E:/COYO/embeds/train_noup",
        samples_per_shard = 5_000,
    )
    """
    export_embeddings_sharded(
        val_dl, dino_encoder, qwen_embedder,
        output_dir        = "E:/COYO/embeds/val_noup",
        samples_per_shard = 5_000,
    )
    export_embeddings_sharded(
        test_dl, dino_encoder, qwen_embedder,
        output_dir        = "E:/COYO/embeds/test_noup",
        samples_per_shard = 5_000,
    )
