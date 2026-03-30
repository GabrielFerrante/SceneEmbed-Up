"""
consolidate_shards.py
---------------------
Consolida arquivos `.h5` pequenos (um por amostra) em shards grandes,
sem precisar manter ambos no disco ao mesmo tempo.

Fluxo por pasta de origem:
    1. Lista todos os `sample_*.h5` dentro da pasta.
    2. Lê todos em memória (numpy arrays).
    3. Cria o arquivo de shard grande no destino.
    4. Exclui os arquivos pequenos (e a pasta se ficar vazia).

Shapes esperados nos arquivos pequenos:
    visual_feats  : [1024, 768]   float16
    text_feats    : [1, 4096]     float16
    visual_global : [768]         float16   (opcional, preservado se existir)

Shapes resultantes no shard:
    visual_feats  : [N, 1024, 768]
    text_feats    : [N, 1, 4096]
    visual_global : [N, 768]      (somente se presente em todos os arquivos)
    
    
# Substitui no mesmo disco (sem espaço extra) — seu caso
python small_to_shards.py G:/coyo/embeds/val_anyup

# Com saída separada (se tiver espaço)
python small_to_shards.py G:/coyo/embeds/val_anyup --output_dir H:/shards/val

# Modo teste — NÃO deleta os originais
python small_to_shards.py G:/coyo/embeds/val_anyup --no_delete

# Compressão mais rápida (menos CPU, arquivo levemente maior)
python small_to_shards.py G:/coyo/embeds/val_anyup --compression lzf
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_sample_files(folder: Path) -> List[Path]:
    """Retorna todos os `sample_*.h5` em `folder`, ordenados por índice."""
    files = sorted(
        folder.glob("sample_*.h5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return files


def _read_sample(path: Path) -> dict[str, np.ndarray]:
    """Lê um arquivo de amostra e devolve um dict de arrays numpy."""
    with h5py.File(path, "r") as f:
        data = {key: f[key][()] for key in f.keys()}
    return data


def _validate_shard(path: Path, expected_n: int) -> None:
    """Verificação básica de integridade do shard gerado."""
    try:
        with h5py.File(path, "r") as f:
            assert "visual_feats" in f and "text_feats" in f, "Datasets obrigatórios ausentes."
            vf = f["visual_feats"]
            tf = f["text_feats"]
            if vf.shape[0] != expected_n:
                raise ValueError(f"visual_feats tem {vf.shape[0]} amostras, esperava {expected_n}.")
            if tf.shape[0] != expected_n:
                raise ValueError(f"text_feats tem {tf.shape[0]} amostras, esperava {expected_n}.")
    except Exception as e:
        print(f"[WARN] Falha na validação do shard {path}: {e}")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def consolidate_folder(
    src_folder: Path,
    dst_folder: Path,
    shard_name: str,
    delete_originals: bool = True,
    compression: str = "gzip",
) -> int:
    """
    Lê todos os `sample_*.h5` de `src_folder`, agrupa em um único shard
    `dst_folder / shard_name.h5`, apaga os originais e devolve o nº de amostras.

    Retorna 0 se não houver arquivos para processar.
    """
    sample_files = _list_sample_files(src_folder)
    if not sample_files:
        return 0

    n = len(sample_files)
    print(f"  [{src_folder.name}] {n} amostras → {shard_name}.h5")

    # -----------------------------------------------------------------------
    # 1. Leitura em memória
    # -----------------------------------------------------------------------
    all_visual: list[np.ndarray] = []
    all_text: list[np.ndarray] = []
    all_global: list[np.ndarray] = []
    has_global = True

    for p in tqdm(sample_files, desc="  Lendo", leave=False):
        try:
            d = _read_sample(p)
        except Exception as e:
            print(f"[ERROR] Não foi possível ler {p}: {e} — abortando pasta.")
            return 0

        all_visual.append(d["visual_feats"])   # [1024, 768]
        all_text.append(d["text_feats"])        # [1, 4096]

        if has_global:
            if "visual_global" in d:
                all_global.append(d["visual_global"])  # [768]
            else:
                has_global = False  # pelo menos um arquivo sem visual_global

    # Stack → [N, ...]
    visual_arr = np.stack(all_visual, axis=0)   # [N, 1024, 768]
    text_arr   = np.stack(all_text,   axis=0)   # [N, 1,    4096]
    global_arr = np.stack(all_global, axis=0) if has_global else None  # [N, 768]

    # -----------------------------------------------------------------------
    # 2. Escrita do shard
    # -----------------------------------------------------------------------
    dst_folder.mkdir(parents=True, exist_ok=True)
    shard_path = dst_folder / f"{shard_name}.h5"

    with h5py.File(shard_path, "w") as f:
        f.create_dataset("visual_feats",  data=visual_arr,  compression=compression)
        f.create_dataset("text_feats",    data=text_arr,    compression=compression)
        if global_arr is not None:
            f.create_dataset("visual_global", data=global_arr, compression=compression)

        # Metadados úteis
        f.attrs["n_samples"]      = n
        f.attrs["source_folder"]  = str(src_folder)
        f.attrs["shard_name"]     = shard_name

    _validate_shard(shard_path, n)

    # -----------------------------------------------------------------------
    # 3. Remoção dos arquivos pequenos
    # -----------------------------------------------------------------------
    if delete_originals:
        for p in tqdm(sample_files, desc="  Deletando originais", leave=False):
            try:
                p.unlink()
            except OSError as e:
                print(f"[WARN] Não foi possível deletar {p}: {e}")

        # Remove a pasta de origem se estiver vazia (e for sub-pasta numérica)
        try:
            remaining = list(src_folder.iterdir())
            if not remaining:
                src_folder.rmdir()
                print(f"  Pasta vazia removida: {src_folder}")
        except OSError:
            pass  # pasta não vazia ou sem permissão — ignora

    return n


def consolidate_root(
    root_dir: str,
    output_dir: str | None,
    shard_prefix: str = "shard",
    delete_originals: bool = True,
    compression: str = "gzip",
) -> None:
    """
    Percorre `root_dir` procurando sub-pastas com arquivos `sample_*.h5`.

    Se `output_dir` for None, os shards são salvos dentro de `root_dir`
    (mesmo local, necessário quando não há espaço extra).

    Cada sub-pasta vira um arquivo:
        <output_dir>/<pasta_nome>/<shard_prefix>.h5
    """
    root = Path(root_dir)
    out_root = Path(output_dir) if output_dir else root

    # Encontra todas as pastas com sample_*.h5
    sub_folders = sorted(
        [p for p in root.iterdir() if p.is_dir() and any(p.glob("sample_*.h5"))]
    )

    if not sub_folders:
        # Talvez os arquivos estejam diretamente na raiz
        if any(root.glob("sample_*.h5")):
            sub_folders = [root]
        else:
            print(f"Nenhuma pasta com sample_*.h5 encontrada em: {root}")
            return

    total_samples = 0
    print(f"\nEncontradas {len(sub_folders)} pasta(s) para consolidar.\n")

    for folder in sub_folders:
        # Nome relativo preservado na saída
        rel = folder.relative_to(root) if folder != root else Path(".")
        dst = out_root / rel if folder != root else out_root

        n = consolidate_folder(
            src_folder=folder,
            dst_folder=dst,
            shard_name=f"{shard_prefix}_{folder.name}" if folder != root else shard_prefix,
            delete_originals=delete_originals,
            compression=compression,
        )
        total_samples += n

    print(f"\nConcluído. Total de amostras consolidadas: {total_samples}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida sample_*.h5 pequenos em shards grandes."
    )
    parser.add_argument(
        "root_dir",
        help="Diretório raiz com sub-pastas contendo sample_*.h5",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Diretório de saída para os shards. "
            "Se omitido, usa root_dir (substitui no mesmo lugar). "
            "Use apenas se houver espaço separado disponível."
        ),
    )
    parser.add_argument(
        "--shard_prefix",
        default="shard",
        help="Prefixo para o nome dos arquivos de shard (padrão: 'shard').",
    )
    parser.add_argument(
        "--no_delete",
        action="store_true",
        help="Não deleta os arquivos originais (útil para testes).",
    )
    parser.add_argument(
        "--compression",
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="Algoritmo de compressão HDF5 (padrão: gzip).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    compression = None if args.compression == "none" else args.compression

    consolidate_root(
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        shard_prefix=args.shard_prefix,
        delete_originals=not args.no_delete,
        compression=compression,
    )