from __future__ import annotations

import os


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



