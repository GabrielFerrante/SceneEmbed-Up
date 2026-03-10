from __future__ import annotations

import os
from typing import List


def ensure_dir(path: str) -> str:
    """
    Garante que um diretório exista, criando‑o se necessário.

    Returns
    -------
    str
        O próprio caminho recebido, para facilitar encadeamento.
    """
    os.makedirs(path, exist_ok=True)
    return path


def get_next_sample_index(output_dir: str) -> int:
    """
    Descobre o próximo índice de amostra para arquivos `sample_<idx>.h5`.

    Percorre recursivamente `output_dir` em busca de arquivos com o padrão
    `sample_<idx>.h5` e retorna o próximo índice livre.
    """
    existing_indices: List[int] = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if not file.endswith(".h5"):
                continue
            if not file.startswith("sample_"):
                continue
            try:
                num = int(file.split("_")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            existing_indices.append(num)

    return (max(existing_indices) + 1) if existing_indices else 0

