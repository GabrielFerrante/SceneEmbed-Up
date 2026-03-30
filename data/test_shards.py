import h5py
import numpy as np
from pathlib import Path
import os
import glob

test_dir = Path("F:/COYO/embeds/train_anyup/")
search_pattern = os.path.join(test_dir, "**", "*.h5")
shard_files = sorted(glob.glob(search_pattern, recursive=True))

if not shard_files:
    print("ERRO: Nenhum arquivo .h5 encontrado no diretório!")
else:
    for sf in shard_files:
        print(f"\n{'='*50}")
        print(f"Arquivo: {sf} )")
        try:
            with h5py.File(sf, "r") as f:
                print(f"Keys: {list(f.keys())}")
                for key in f.keys():
                    arr = f[key][:]
                    print(f"  {key}:")
                    print(f"    shape : {arr.shape}")
                    print(f"    dtype : {arr.dtype}")
                    print(f"    nan   : {np.isnan(arr).any()}")
                    print(f"    inf   : {np.isinf(arr).any()}")
                    print(f"    zeros : {(arr == 0).all(axis=-1).sum()} amostras zeradas")
                    print(f"    min/max: {arr.min():.4f} / {arr.max():.4f}")
        except Exception as e:
            print(f"  ERRO ao abrir: {e}")