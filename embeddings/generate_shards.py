from __future__ import annotations

import os
import sys
from typing import Iterable, Optional, Tuple

import h5py
import numpy as np
import torch
from tqdm import tqdm

sys.path.append("..")

from data.data_utils_pytorch import create_all_dataloaders  # noqa: E402
from models.encoders.dinov3_extrator import DinoSceneEncoder  # noqa: E402
from models.encoders.qwen3_extrator import QwenSceneEmbedder  # noqa: E402
from utils.io_utils import ensure_dir  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes de shape (centraliza para facilitar manutenção)
# ---------------------------------------------------------------------------
VISUAL_DIM = 768          # dim do patch DINO
VISUAL_PATCHES = 1024     # 32×32
TEXT_DIM = 4096           # dim do Qwen
GLOBAL_DIM = 768          # dim do CLS token DINO


# ---------------------------------------------------------------------------
# Helpers de shard
# ---------------------------------------------------------------------------

def _shard_path(output_dir: str, shard_idx: int) -> str:
    return os.path.join(output_dir, f"shard_{str(shard_idx).zfill(6)}.h5")


def _get_resume_state(output_dir: str) -> Tuple[int, int]:
    """
    Retorna (shard_idx, samples_já_escritos_no_shard_atual).

    Lógica:
    - Varre shards existentes em ordem.
    - O último shard pode estar incompleto (crash): usa o tamanho real.
    - Shards anteriores são considerados cheios (samples_per_shard).
    """
    existing = sorted(
        f for f in os.listdir(output_dir) if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0, 0

    last = existing[-1]
    shard_idx = int(last.replace("shard_", "").replace(".h5", ""))
    path = _shard_path(output_dir, shard_idx)
    try:
        with h5py.File(path, "r") as f:
            n = f["visual_feats"].shape[0]
    except Exception:
        n = 0
    return shard_idx, n


def _count_global_samples(output_dir: str, samples_per_shard: int) -> int:
    """Conta total de samples já exportados (shards completos + parcial)."""
    existing = sorted(
        f for f in os.listdir(output_dir) if f.startswith("shard_") and f.endswith(".h5")
    )
    if not existing:
        return 0
    total = 0
    for fname in existing[:-1]:
        total += samples_per_shard          # shards anteriores: assumidos cheios
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
    Valida integridade básica de um shard `.h5`.

    Verifica:
    - Datasets obrigatórios presentes.
    - Shapes consistentes: todos com a mesma dimensão-0 (N).
    - Dimensões internas corretas (VISUAL_DIM, TEXT_DIM, GLOBAL_DIM).
    - N == expected_n quando fornecido.
    """
    try:
        with h5py.File(path, "r") as f:
            required = ("visual_feats", "text_feats", "visual_global")
            for key in required:
                assert key in f, f"Dataset '{key}' ausente."

            v_shape = f["visual_feats"].shape   # [N, PATCHES, DIM]
            t_shape = f["text_feats"].shape     # [N, 1, TEXT_DIM]
            g_shape = f["visual_global"].shape  # [N, GLOBAL_DIM]

            n = v_shape[0]
            assert t_shape[0] == n and g_shape[0] == n, (
                f"Inconsistência no eixo-0: visual={n}, text={t_shape[0]}, global={g_shape[0]}"
            )

            if len(v_shape) != 3 or v_shape[1] != VISUAL_PATCHES or v_shape[2] != VISUAL_DIM:
                raise ValueError(f"Shape inesperado para visual_feats: {v_shape}")
            if len(t_shape) != 3 or t_shape[2] != TEXT_DIM:
                raise ValueError(f"Shape inesperado para text_feats: {t_shape}")
            if len(g_shape) != 2 or g_shape[1] != GLOBAL_DIM:
                raise ValueError(f"Shape inesperado para visual_global: {g_shape}")

            if expected_n is not None and n != expected_n:
                raise ValueError(f"Shard com {n} samples, esperado {expected_n}.")

    except Exception as e:
        print(f"[WARN] Falha na validação de integridade do shard {path}: {e}")


# ---------------------------------------------------------------------------
# ShardWriter: gerencia abertura/fechamento de shards
# ---------------------------------------------------------------------------

class ShardWriter:
    """
    Escreve samples em shards `.h5` com datasets resizáveis.

    Cada shard contém até `samples_per_shard` samples e expõe os datasets:
    - ``visual_feats``  : float16, shape ``[N, VISUAL_PATCHES, VISUAL_DIM]``
    - ``text_feats``    : float16, shape ``[N, 1,              TEXT_DIM]``
    - ``visual_global`` : float16, shape ``[N,                 GLOBAL_DIM]``

    Parameters
    ----------
    output_dir:
        Diretório de saída dos shards.
    samples_per_shard:
        Número máximo de samples por shard antes de rotacionar.
    resume:
        Se True, detecta o último shard incompleto e continua escrevendo nele.
    """

    def __init__(self, output_dir: str, samples_per_shard: int = 5_000, resume: bool = True):
        self.output_dir = output_dir
        self.samples_per_shard = samples_per_shard

        ensure_dir(output_dir)

        if resume:
            self._shard_idx, self._in_shard = _get_resume_state(output_dir)
        else:
            self._shard_idx, self._in_shard = 0, 0

        self._file: Optional[h5py.File] = None
        self._ds_visual: Optional[h5py.Dataset] = None
        self._ds_text: Optional[h5py.Dataset] = None
        self._ds_global: Optional[h5py.Dataset] = None

        # Reabre shard parcial se necessário
        if self._in_shard > 0:
            self._open_shard(new=False)

        self.total_written = _count_global_samples(output_dir, samples_per_shard)
        print(
            f"[ShardWriter] Retomando do shard {self._shard_idx}, "
            f"sample {self._in_shard} dentro do shard "
            f"(total global já exportado: {self.total_written})."
        )

    # ------------------------------------------------------------------
    # Gerenciamento de arquivo
    # ------------------------------------------------------------------

    def _open_shard(self, new: bool = True) -> None:
        """Abre (ou cria) um arquivo de shard."""
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
            # Reabre shard incompleto em modo append
            self._file = h5py.File(path, "a")
            self._ds_visual = self._file["visual_feats"]
            self._ds_text = self._file["text_feats"]
            self._ds_global = self._file["visual_global"]

    def _close_shard(self) -> None:
        if self._file is not None:
            self._file.flush()
            self._file.close()
            self._file = None
            _validate_shard(_shard_path(self.output_dir, self._shard_idx))

    def _rotate_shard(self) -> None:
        """Fecha shard atual e prepara o próximo."""
        self._close_shard()
        self._shard_idx += 1
        self._in_shard = 0
        self._open_shard(new=True)

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def write_batch(
        self,
        visual_feats: np.ndarray,   # [B, PATCHES, DIM]
        text_feats: np.ndarray,     # [B, 1, TEXT_DIM]
        global_feats: np.ndarray,   # [B, GLOBAL_DIM]
    ) -> None:
        """
        Escreve um batch de samples, rotacionando shards conforme necessário.
        Pode quebrar o batch em sub-lotes se ele ultrapassar o limite do shard.
        """
        b = visual_feats.shape[0]
        offset = 0

        while offset < b:
            if self._file is None:
                self._open_shard(new=(self._in_shard == 0))

            slots = self.samples_per_shard - self._in_shard
            n = min(slots, b - offset)

            end = offset + n
            v = visual_feats[offset:end]
            t = text_feats[offset:end]
            g = global_feats[offset:end]

            new_size = self._in_shard + n
            self._ds_visual.resize(new_size, axis=0)
            self._ds_text.resize(new_size, axis=0)
            self._ds_global.resize(new_size, axis=0)

            self._ds_visual[self._in_shard:new_size] = v
            self._ds_text[self._in_shard:new_size] = t
            self._ds_global[self._in_shard:new_size] = g

            self._in_shard = new_size
            self.total_written += n
            offset = end

            if self._in_shard >= self.samples_per_shard:
                self._rotate_shard()

    def close(self) -> None:
        self._close_shard()

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Função principal de exportação
# ---------------------------------------------------------------------------

def export_embeddings_sharded(
    dataloader: Iterable[Tuple[torch.Tensor, Iterable[str]]],
    dino: DinoSceneEncoder,
    qwen: QwenSceneEmbedder,
    output_dir: str,
    samples_per_shard: int = 5_000,
    resume: bool = True,
) -> None:
    """
    Exporta embeddings de imagem/texto para shards `.h5`.

    Cada shard reúne até `samples_per_shard` amostras nos datasets:

    - ``visual_feats``  : ``[N, 1024, 768]``  — patches 32×32 do DINO.
    - ``text_feats``    : ``[N, 1,    4096]`` — embedding textual do Qwen.
    - ``visual_global`` : ``[N,       768]``  — token CLS do DINO.

    Parameters
    ----------
    dataloader:
        Dataloader que emite tuplas ``(imgs [B,3,H,W], texts [B])``.
    dino:
        Encoder visual baseado em DINOv2.
    qwen:
        Embedder textual baseado em Qwen.
    output_dir:
        Diretório de destino dos shards.
    samples_per_shard:
        Número máximo de amostras por arquivo `.h5`.
    resume:
        Se True (padrão), detecta e continua de um export interrompido.
    """
    dino.model.eval()
    qwen.model.eval()

    with ShardWriter(output_dir, samples_per_shard=samples_per_shard, resume=resume) as writer:
        global_idx = writer.total_written      # amostras já exportadas (para skip)
        skipped = 0

        with torch.no_grad():
            for imgs, texts in tqdm(dataloader, desc="Exportando shards"):
                batch_size = imgs.shape[0]

                # ── Skip de batches já processados ───────────────────────
                if skipped < global_idx:
                    remaining_skip = global_idx - skipped
                    if batch_size <= remaining_skip:
                        skipped += batch_size
                        continue
                    # Batch parcialmente processado (borda de retomada)
                    skip_within = remaining_skip
                    imgs = imgs[skip_within:]
                    texts = list(texts)[skip_within:]
                    skipped += skip_within
                    batch_size = imgs.shape[0]

                try:
                    # ── Extração de features ──────────────────────────────
                    globais, locais = dino.extract_features(imgs.to("cuda"))

                    locais_small = torch.nn.functional.adaptive_avg_pool2d(
                        locais, (32, 32)
                    )  # [B, 768, 32, 32]

                    # [B, C, H, W] -> [B, H*W, C] == [B, 1024, 768]
                    b, c, h, w = locais_small.shape
                    flat = locais_small.reshape(b, c, h * w).permute(0, 2, 1)

                    formatted_texts = [[t] for t in texts]
                    t_feat = qwen.embed_components(formatted_texts, normalize=False)
                    if t_feat.dim() == 2:
                        t_feat = t_feat.unsqueeze(1)  # [B, 1, 4096]

                    # ── Conversão para numpy float16 ──────────────────────
                    v_np = flat.cpu().numpy().astype("float16")        # [B, 1024, 768]
                    t_np = t_feat.cpu().numpy().astype("float16")      # [B, 1,    4096]
                    g_np = globais.cpu().numpy().astype("float16")     # [B,       768]

                    writer.write_batch(v_np, t_np, g_np)

                except KeyboardInterrupt:
                    print(
                        f"\nInterrupção detectada! "
                        f"Total exportado até agora: {writer.total_written}. "
                        "Pode fechar agora com segurança."
                    )
                    return

                except RuntimeError as e:
                    if "CUDA" in str(e).upper() and torch.cuda.is_available():
                        print(f"Erro de GPU durante exportação: {e}")
                        print(torch.cuda.memory_summary())
                    raise

    print(f"\nExportação concluída. Total de amostras: {writer.total_written}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="anyup")
    qwen_embedder = QwenSceneEmbedder(device="cuda")

    train_dataloader, val_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="train"
    )
    test_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="test"
    )

    export_embeddings_sharded(
        val_dataloader,
        dino_encoder,
        qwen_embedder,
        output_dir="G:/coyo/embeds/val_anyup_sharded",
        samples_per_shard=5_000,
    )
    export_embeddings_sharded(
        test_dataloader,
        dino_encoder,
        qwen_embedder,
        output_dir="G:/coyo/embeds/test_anyup_sharded",
        samples_per_shard=5_000,
    )