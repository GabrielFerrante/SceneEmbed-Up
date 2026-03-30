import glob
import h5py

if __name__ == '__main__':
    files = sorted(glob.glob("G:/coyo/embeds/val_anyup/**/*.h5", recursive=True))
    #files = sorted(glob.glob("F:/COYO/embeds/train_anyup/**/*.h5", recursive=True))
    

    with h5py.File(files[0], "r") as f:
        n_per_shard  = f["visual_feats"].shape[0]
        visual_bytes = f["visual_feats"].nbytes
        text_bytes   = f["text_feats"].nbytes

    bytes_per_shard = visual_bytes + text_bytes
    total_shards    = len(files)
    total_samples   = n_per_shard * total_shards
    total_gb        = bytes_per_shard * total_shards / 1e9

    max_ram_gb      = 15.0
    shards_in_ram   = min(total_shards, int(max_ram_gb * 1e9 // bytes_per_shard))
    ram_used_gb     = bytes_per_shard * shards_in_ram / 1e9

    print(f"Shards totais:        {total_shards}")
    print(f"Amostras por shard:   {n_per_shard:,}")
    print(f"Amostras totais:      {total_samples:,}")
    print(f"Tamanho por shard:    {bytes_per_shard / 1e6:.1f} MB")
    print(f"RAM total necessária: {total_gb:.1f} GB")
    print(f"─────────────────────────────────────────")
    print(f"Shards que cabem em {max_ram_gb:.0f} GB: {shards_in_ram}")
    print(f"RAM que será usada:   {ram_used_gb:.1f} GB")
    print(f"Amostras em memória:  {n_per_shard * shards_in_ram:,} / {total_samples:,} "
          f"({100 * shards_in_ram / total_shards:.1f}% do dataset)")