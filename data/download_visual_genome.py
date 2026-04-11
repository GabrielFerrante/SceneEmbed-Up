"""
download_visual_genome.py
--------------------------
Baixa o Visual Genome (anotacoes + imagens) para avaliacao SGGen Recall@K.

Arquivos baixados:
  1. scene_graphs.json  (~1.1 GB) — scene graphs com objetos e relacoes
  2. image_data.json    (~800 KB) — metadados das imagens
  3. Imagens VG_100K + VG_100K_2 (~15 GB total, opcional)

Uso:
    python data/download_visual_genome.py --data-dir F:/VG
    python data/download_visual_genome.py --data-dir F:/VG --skip-images
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
import zipfile

from tqdm import tqdm


# URLs oficiais do Visual Genome (visualgenome.org)
URLS = {
    "scene_graphs": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/scene_graphs.json.zip",
    "image_data": "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/image_data.json.zip",
    "images_part1": "https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip",
    "images_part2": "https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip",
}


class _DownloadProgress:
    """Hook para urllib.request.urlretrieve com barra tqdm."""

    def __init__(self, desc: str):
        self.bar: tqdm | None = None
        self.desc = desc

    def __call__(self, block_num: int, block_size: int, total_size: int):
        if self.bar is None:
            self.bar = tqdm(
                total=total_size if total_size > 0 else None,
                unit="B",
                unit_scale=True,
                desc=self.desc,
            )
        downloaded = block_num * block_size
        self.bar.update(block_size)
        if total_size > 0 and downloaded >= total_size:
            self.bar.close()


def download_file(url: str, dest: str, desc: str) -> None:
    """Baixa arquivo com barra de progresso. Pula se ja existir."""
    if os.path.exists(dest):
        print(f"  [skip] {desc} ja existe: {dest}")
        return
    print(f"  Baixando {desc}...")
    try:
        urllib.request.urlretrieve(url, dest, reporthook=_DownloadProgress(desc))
    except Exception as e:
        # Remove arquivo parcial
        if os.path.exists(dest):
            os.remove(dest)
        raise RuntimeError(f"Falha ao baixar {url}: {e}") from e


def extract_zip(zip_path: str, dest_dir: str) -> None:
    """Extrai ZIP para dest_dir com progresso."""
    print(f"  Extraindo {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in tqdm(zf.infolist(), desc="Extraindo"):
            zf.extract(member, dest_dir)


def validate_data(data_dir: str, skip_images: bool) -> None:
    """Verifica se os arquivos essenciais existem apos download."""
    sg_path = os.path.join(data_dir, "scene_graphs.json")
    if not os.path.exists(sg_path):
        print(f"  [ERRO] scene_graphs.json nao encontrado em {data_dir}")
        sys.exit(1)

    # Contagem rapida
    print("  Validando scene_graphs.json...")
    with open(sg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  scene_graphs.json: {len(data)} imagens anotadas")

    if not skip_images:
        n_imgs = 0
        for subdir in ["VG_100K", "VG_100K_2"]:
            img_dir = os.path.join(data_dir, subdir)
            if os.path.isdir(img_dir):
                count = len([f for f in os.listdir(img_dir) if f.endswith(".jpg")])
                n_imgs += count
                print(f"  {subdir}: {count} imagens")
        if n_imgs == 0:
            print("  [AVISO] Nenhuma imagem encontrada. Verifique a extracao.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Visual Genome para benchmark SGGen"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Diretorio destino (ex: F:/VG)",
    )
    parser.add_argument(
        "--skip-images",
        action="store_true",
        help="Baixar apenas anotacoes (scene_graphs.json + image_data.json), sem imagens (~15GB)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    os.makedirs(data_dir, exist_ok=True)

    # ── 1. Anotacoes (obrigatorio) ──────────────────────────────────────
    print("\n=== Baixando anotacoes ===")
    for key in ["scene_graphs", "image_data"]:
        zip_name = f"{key}.json.zip"
        zip_path = os.path.join(data_dir, zip_name)
        json_path = os.path.join(data_dir, f"{key}.json")

        if os.path.exists(json_path):
            print(f"  [skip] {key}.json ja existe")
            continue

        download_file(URLS[key], zip_path, key)
        extract_zip(zip_path, data_dir)

        # Limpa ZIP apos extracao
        if os.path.exists(zip_path):
            os.remove(zip_path)

    # ── 2. Imagens (opcional) ───────────────────────────────────────────
    if not args.skip_images:
        print("\n=== Baixando imagens (~15 GB) ===")
        for key in ["images_part1", "images_part2"]:
            zip_name = f"{key}.zip"
            zip_path = os.path.join(data_dir, zip_name)
            download_file(URLS[key], zip_path, key)
            extract_zip(zip_path, data_dir)
            # Nao remove ZIPs de imagens (podem ser uteis para re-extracao)
    else:
        print("\n  --skip-images: pulando download de imagens")

    # ── 3. Validacao ────────────────────────────────────────────────────
    print("\n=== Validacao ===")
    validate_data(data_dir, args.skip_images)
    print("\nDownload finalizado!")


if __name__ == "__main__":
    main()
