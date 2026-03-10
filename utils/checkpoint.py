from __future__ import annotations

import os
from typing import Any

import torch
from torch.nn import Module


def save_epoch_checkpoint(
    model: Module,
    epoch: int,
    directory: str = "checkpoints",
    name: str = "aligner_epoch",
) -> str:
    """
    Salva um checkpoint de época padronizado.

    Parameters
    ----------
    model:
        Instância de `nn.Module` cujo `state_dict` será persistido.
    epoch:
        Número da época (1‑based).
    directory:
        Diretório base para os arquivos de checkpoint.
    name:
        Prefixo do nome do arquivo de saída.

    Returns
    -------
    str
        Caminho completo do arquivo salvo.
    """
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}_{epoch}.pth")
    torch.save(model.state_dict(), path)
    return path

