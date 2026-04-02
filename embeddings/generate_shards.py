"""
generate_shards.py
------------------
Gera embeddings pré-computados em shards .h5 prontos para treino.

Shapes de saída por shard (samples_per_shard=5000):
    visual_feats  : [5000, 1024, 768]  float16  — patches 32×32 do DINO
    text_feats    : [5000, 1,    4096] float16  — embedding textual do Qwen
    visual_global : [5000,       768]  float16  — token CLS do DINO

Uso:
    python generate_shards.py

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
# Constantes de shape
# ---------------------------------------------------------------------------
VISUAL_DIM    = 768    # dim do patch DINO
VISUAL_PATCHES = 1024  # 32×32
TEXT_DIM      = 4096   # dim do Qwen
GLOBAL_DIM    = 768    # dim do CLS token DINO


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
        shard_idx: índice do último shard (completo ou parcial)
        n_samples: quantas amostras já foram escritas nesse shard
    """
    # Busca apenas arquivos shard_*.h5 direto na raiz (sem subpastas)
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

    total = (len(existing) - 1) * samples_per_shard  # shards completos

    # Último shard pode estar incompleto
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
        visual_feats  : [N, 1024, 768]
        text_feats    : [N, 1,    4096]
        visual_global : [N,       768]   (opcional)
    """
    try:
        with h5py.File(path, "r") as f:

            # Datasets obrigatórios
            for key in ("visual_feats", "text_feats"):
                assert key in f, f"Dataset '{key}' ausente."

            v_shape = f["visual_feats"].shape  # [N, 1024, 768]
            t_shape = f["text_feats"].shape    # [N, 1,    4096]
            n       = v_shape[0]

            # Valida consistência do eixo-0
            assert t_shape[0] == n, (
                f"Eixo-0 inconsistente: visual={n}, text={t_shape[0]}"
            )

            # Valida shapes internos
            assert len(v_shape) == 3 and v_shape[1] == VISUAL_PATCHES and v_shape[2] == VISUAL_DIM, \
                f"Shape inesperado para visual_feats: {v_shape}"
            assert len(t_shape) == 3 and t_shape[2] == TEXT_DIM, \
                f"Shape inesperado para text_feats: {t_shape}"

            # visual_global é opcional
            if "visual_global" in f:
                g_shape = f["visual_global"].shape
                assert g_shape[0] == n and g_shape[1] == GLOBAL_DIM, \
                    f"Shape inesperado para visual_global: {g_shape}"

            # Valida contagem esperada
            if expected_n is not None:
                assert n == expected_n, f"Esperava {expected_n} amostras, encontrou {n}."

        print(f"  [OK] {os.path.basename(path)} — {n} amostras, shape visual={v_shape}")

    except Exception as e:
        print(f"  [WARN] Falha na validação de {path}: {e}")


# ---------------------------------------------------------------------------
# ShardWriter
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Escreve amostras em shards .h5 com datasets resizáveis.

    Cada shard acumula até `samples_per_shard` amostras e então rotaciona
    automaticamente para o próximo arquivo. Suporta retomada após crash.

    Shapes escritos por shard:
        visual_feats  : [N, 1024, 768]  float16
        text_feats    : [N, 1,    4096] float16
        visual_global : [N,       768]  float16
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

        # Reabre shard parcial se necessário
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
        else:
            # Reabre em modo append
            self._file      = h5py.File(path, "a")
            self._ds_visual = self._file["visual_feats"]
            self._ds_text   = self._file["text_feats"]
            self._ds_global = self._file["visual_global"]

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
        visual_feats: np.ndarray,  # [B, 1024, 768] float16
        text_feats  : np.ndarray,  # [B, 1,    4096] float16
        global_feats: np.ndarray,  # [B,       768]  float16
    ) -> None:
        """
        Escreve um batch, quebrando em sub-lotes se ultrapassar o shard.
        """
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
    Extrai e salva embeddings em shards .h5.

    Shapes gerados:
        visual_feats  : [samples_per_shard, 1024, 768]
        text_feats    : [samples_per_shard, 1,    4096]
        visual_global : [samples_per_shard,       768]

    Parameters
    ----------
    dataloader:
        Emite (imgs [B,3,H,W], texts [B]).
    dino:
        Encoder visual DINOv2.
    qwen:
        Embedder textual Qwen.
    output_dir:
        Diretório de saída. Shards salvos como shard_000000.h5, shard_000001.h5, ...
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
            for imgs, texts in tqdm(dataloader, desc=f"Exportando → {output_dir}"):
                batch_size = imgs.shape[0]

                # ── Skip de batches já exportados ────────────────────────
                if skipped < global_idx:
                    remaining = global_idx - skipped
                    if batch_size <= remaining:
                        skipped += batch_size
                        continue
                    # Borda de retomada: batch parcialmente processado
                    imgs       = imgs[remaining:]
                    texts      = list(texts)[remaining:]
                    skipped   += remaining
                    batch_size = imgs.shape[0]

                try:
                    # ── Extração de features ──────────────────────────────
                    globais, locais = dino.extract_features(imgs.to("cuda"))

                    # [B, 768, H, W] → pool → [B, 768, 32, 32]
                    locais_small = torch.nn.functional.adaptive_avg_pool2d(locais, (32, 32))

                    # [B, 768, 32, 32] → [B, 1024, 768]
                    b, c, h, w = locais_small.shape
                    flat = locais_small.reshape(b, c, h * w).permute(0, 2, 1)

                    # Embeddings textuais [B, 1, 4096]
                    t_feat = qwen.embed_components([[t] for t in texts], normalize=False)
                    if t_feat.dim() == 2:
                        t_feat = t_feat.unsqueeze(1)

                    # ── Conversão para numpy float16 ──────────────────────
                    v_np = flat.cpu().numpy().astype("float16")      # [B, 1024, 768]
                    t_np = t_feat.cpu().numpy().astype("float16")    # [B, 1,    4096]
                    g_np = globais.cpu().numpy().astype("float16")   # [B,       768]

                    writer.write_batch(v_np, t_np, g_np)

                except KeyboardInterrupt:
                    print(f"\nInterrupção! Total exportado: {writer.total_written:,}. "
                          "Seguro fechar.")
                    return

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(f"Erro de GPU: {e}")
                        print(torch.cuda.memory_summary())
                    raise

    print(f"\nExportação concluída — {writer.total_written:,} amostras em {output_dir}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="anyup")
    qwen_embedder = QwenSceneEmbedder(device="cuda")

    # Treino
    train_dl, val_dl = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="train"
    )
    test_dl = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="test"
    )

    export_embeddings_sharded(
        train_dl, dino_encoder, qwen_embedder,
        output_dir        = "F:/COYO/embeds/train_anyup",
        samples_per_shard = 5_000,
    )
    export_embeddings_sharded(
        val_dl, dino_encoder, qwen_embedder,
        output_dir        = "G:/coyo/embeds/val_anyup",
        samples_per_shard = 5_000,
    )
    export_embeddings_sharded(
        test_dl, dino_encoder, qwen_embedder,
        output_dir        = "G:/coyo/embeds/test_anyup",
        samples_per_shard = 5_000,
    )