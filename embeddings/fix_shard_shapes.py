"""
fix_shard_shapes.py
-------------------
Corrige shards com shape errado [N, 1000, 1024, 768]
para o shape correto [N*1000, 1024, 768].
"""

import h5py
import numpy as np
import glob
import os
from pathlib import Path
from tqdm import tqdm


def fix_shards(folder_path: str, dry_run: bool = True) -> None:
    search_pattern = os.path.join(folder_path, "**", "*.h5")
    shard_files = sorted(glob.glob(search_pattern, recursive=True))

    if not shard_files:
        print(f"Nenhum arquivo .h5 encontrado em {folder_path}")
        return

    print(f"{len(shard_files)} shards encontrados.")
    print(f"Modo: {'DRY RUN (não altera nada)' if dry_run else 'ESCRITA REAL'}\n")

    for sf in tqdm(shard_files, desc="Verificando shards"):
        with h5py.File(sf, "r") as f:
            vshape = f["visual_feats"].shape
            tshape = f["text_feats"].shape

        # Shape correto: [N, 1024, 768] — nada a fazer
        if len(vshape) == 3:
            print(f"  [OK]  {Path(sf).name}: {vshape}")
            continue

        # Shape errado: [K, N, 1024, 768]
        if len(vshape) == 4:
            new_vshape = (vshape[0] * vshape[1], vshape[2], vshape[3])
            new_tshape = (tshape[0] * tshape[1], tshape[2], tshape[3])
            print(f"  [FIX] {Path(sf).name}: {vshape} → {new_vshape}")

            if not dry_run:
                # Lê os dados
                with h5py.File(sf, "r") as f:
                    visual = f["visual_feats"][:].reshape(new_vshape)
                    text   = f["text_feats"][:].reshape(new_tshape)
                    has_global = "visual_global" in f
                    if has_global:
                        glob_v = f["visual_global"][:]
                        new_gshape = (glob_v.shape[0] * glob_v.shape[1], glob_v.shape[2])
                        glob_v = glob_v.reshape(new_gshape)

                # Sobrescreve o arquivo
                with h5py.File(sf, "w") as f:
                    f.create_dataset("visual_feats", data=visual, compression="gzip")
                    f.create_dataset("text_feats",   data=text,   compression="gzip")
                    if has_global:
                        f.create_dataset("visual_global", data=glob_v, compression="gzip")
                    f.attrs["n_samples"] = len(visual)

    print("\nConcluído!")


if __name__ == "__main__":
    # Teste primeiro — não altera nada
    #fix_shards(r"F:/COYO/embeds/train_anyup", dry_run=True)
    #fix_shards(r"G:/coyo/embeds/val_anyup",   dry_run=True)
    #fix_shards(r"G:/coyo/embeds/test_anyup",  dry_run=True)

    # Depois confirme mudando para dry_run=False
    fix_shards(r"F:/COYO/embeds/train_anyup", dry_run=False)
    # fix_shards(r"G:/coyo/embeds/val_anyup",   dry_run=False)
    # fix_shards(r"G:/coyo/embeds/test_anyup",  dry_run=False)