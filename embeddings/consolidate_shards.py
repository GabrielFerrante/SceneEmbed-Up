"""
consolidate_shards.py
---------------------
Consolida arquivos sample_*.h5 pequenos em shards de 5k amostras,
com shuffle interno, sem precisar de espaço extra em disco.

Uso:
    # Substitui no mesmo disco
    python consolidate_shards.py F:/COYO/embeds/train_anyup

    # Modo teste — NÃO deleta os originais
    python consolidate_shards.py F:/COYO/embeds/train_anyup --no_delete

    # Compressão mais rápida
    python consolidate_shards.py F:/COYO/embeds/train_anyup --compression lzf
    
    ***
    
# Modo teste primeiro — não deleta nada
python consolidate_shards.py F:/COYO/embeds/train_anyup --no_delete

# Se estiver ok, roda de verdade
python consolidate_shards.py F:/COYO/embeds/train_anyup

# Validação e teste
python consolidate_shards.py G:/coyo/embeds/val_anyup
python consolidate_shards.py G:/coyo/embeds/test_anyup

# Compressão mais rápida se demorar muito
python consolidate_shards.py F:/COYO/embeds/train_anyup --compression lzf
```

***
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import h5py
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _list_sample_files(folder: Path) -> List[Path]:
    """Retorna todos os sample_*.h5 em folder, ordenados por índice."""
    return sorted(
        folder.glob("shard_*.h5"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )


def _read_sample(path: Path) -> dict[str, np.ndarray]:
    """Lê um arquivo de amostra e devolve um dict de arrays numpy."""
    with h5py.File(path, "r") as f:
        return {key: f[key][()] for key in f.keys()}


def _validate_shard(path: Path, expected_n: int) -> None:
    """Verificação básica de integridade do shard gerado."""
    try:
        with h5py.File(path, "r") as f:
            assert "visual_feats" in f and "text_feats" in f
            assert f["visual_feats"].shape[0] == expected_n
            assert f["text_feats"].shape[0]   == expected_n
        print(f"  [OK] {path.name} validado ({expected_n} amostras)")
    except Exception as e:
        print(f"  [WARN] Falha na validação de {path}: {e}")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def consolidate_groups(
    root_dir: str,
    folders_per_shard: int = 5,
    delete_originals: bool = True,
    compression: str = "gzip",
    seed: int = 42,
) -> None:
    """
    Agrupa folders_per_shard pastas de sample_*.h5 em um único shard,
    com shuffle interno, salvando no próprio root_dir.

    Shapes de entrada (por arquivo):
        visual_feats  : [1024, 768]  float16
        text_feats    : [1, 4096]    float16
        visual_global : [768]        float16  (opcional)

    Shapes de saída (por shard):
        visual_feats  : [N, 1024, 768]
        text_feats    : [N, 1, 4096]
        visual_global : [N, 768]          (se presente em todos)
    """
    root = Path(root_dir)
    rng  = np.random.default_rng(seed)

    # Lista todas as subpastas com sample_*.h5
    sub_folders = sorted([
        p for p in root.iterdir()
        if p.is_dir() and any(p.glob("shard_*.h5"))
    ])
    
    if not sub_folders:
        print(f"Nenhuma pasta com sample_*.h5 encontrada em: {root}")
        return

    n_groups = len(sub_folders) // folders_per_shard
    remainder = len(sub_folders) % folders_per_shard

    print(f"Pastas encontradas    : {len(sub_folders)}")
    print(f"Pastas por shard      : {folders_per_shard}")
    print(f"Shards completos      : {n_groups}")
    print(f"Pastas restantes      : {remainder} "
          f"({'incluídas no último shard' if remainder else 'nenhuma'})")
    print(f"Saída                 : {root}\n")

    total_samples = 0
    shard_idx     = 0

    for group_start in range(0, len(sub_folders), folders_per_shard):
        group   = sub_folders[group_start : group_start + folders_per_shard]
        all_vis, all_txt, all_glob = [], [], []
        has_global = True
        files_to_delete = []

        # ── 1. Leitura das N pastas do grupo ────────────────────────────────
        for folder in group:
            sample_files = _list_sample_files(folder)
            if not sample_files:
                continue

            print(f"  Lendo [{folder.name}] — {len(sample_files)} amostras...")
            for p in tqdm(sample_files, desc=f"  {folder.name}", leave=False):
                try:
                    d = _read_sample(p)
                    all_vis.append(d["visual_feats"])   # [1024, 768]
                    all_txt.append(d["text_feats"])      # [1, 4096]

                    if has_global:
                        if "visual_global" in d:
                            all_glob.append(d["visual_global"])  # [768]
                        else:
                            has_global = False

                    files_to_delete.append(p)

                except Exception as e:
                    print(f"  [WARN] Erro ao ler {p}: {e} — pulando amostra.")

        if not all_vis:
            print(f"  [WARN] Grupo {shard_idx} vazio, pulando.")
            continue

        # ── 2. Stack + shuffle ───────────────────────────────────────────────
        visual = np.stack(all_vis, axis=0)  # [N, 1024, 768]
        text   = np.stack(all_txt, axis=0)  # [N, 1, 4096]
        glob_v = np.stack(all_glob, axis=0) if has_global else None  # [N, 768]

        idx    = rng.permutation(len(visual))
        visual = visual[idx]
        text   = text[idx]
        if glob_v is not None:
            glob_v = glob_v[idx]

        n = len(visual)

        # ── 3. Escrita do shard ──────────────────────────────────────────────
        shard_path = root / f"consolidated_{shard_idx:04d}.h5"
        print(f"\n  Escrevendo {shard_path.name} ({n:,} amostras)...")

        with h5py.File(shard_path, "w") as f:
            f.create_dataset("visual_feats",  data=visual, compression=compression)
            f.create_dataset("text_feats",    data=text,   compression=compression)
            if glob_v is not None:
                f.create_dataset("visual_global", data=glob_v, compression=compression)
            f.attrs["n_samples"] = n

        _validate_shard(shard_path, n)

        # ── 4. Delete dos originais ──────────────────────────────────────────
        if delete_originals:
            print(f"  Deletando {len(files_to_delete):,} arquivos originais...")
            for p in tqdm(files_to_delete, desc="  Deletando", leave=False):
                try:
                    p.unlink()
                except OSError as e:
                    print(f"  [WARN] Não foi possível deletar {p}: {e}")

            # Remove pastas vazias
            for folder in group:
                try:
                    if not any(folder.iterdir()):
                        folder.rmdir()
                        print(f"  Pasta vazia removida: {folder.name}")
                except OSError:
                    pass

        total_samples += n
        shard_idx     += 1
        print()

    print(f"Concluído! {shard_idx} shards gerados | {total_samples:,} amostras totais")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consolida sample_*.h5 em shards de 5k amostras com shuffle."
    )
    parser.add_argument(
        "root_dir",
        help="Diretório raiz com sub-pastas contendo sample_*.h5",
    )
    parser.add_argument(
        "--folders_per_shard", type=int, default=5,
        help="Quantas pastas fundir por shard (padrão: 5 → 5k amostras/shard).",
    )
    parser.add_argument(
        "--no_delete", action="store_true",
        help="Não deleta os arquivos originais (útil para testes).",
    )
    parser.add_argument(
        "--compression", default="gzip", choices=["gzip", "lzf", "none"],
        help="Algoritmo de compressão HDF5 (padrão: gzip).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed para o shuffle (padrão: 42).",
    )

    


if __name__ == "__main__":
    
    consolidate_groups(
        root_dir          = "F:/COYO/embeds/train_anyup/",
        folders_per_shard = 4,
        delete_originals  = True,  # True quando confirmar que está ok
        compression       = "gzip",
        seed              = 42,
    )
    
