from __future__ import annotations

import datetime
import os
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter


def create_tensorboard_writer(log_root: str = "logs") -> Tuple[SummaryWriter, str]:
    """
    Cria um `SummaryWriter` do TensorBoard com diretório timestampado.

    Parameters
    ----------
    log_root:
        Diretório raiz onde os logs serão salvos.

    Returns
    -------
    writer:
        Instância de `SummaryWriter`.
    log_dir:
        Caminho completo do diretório de logs criado.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join(log_root, timestamp)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    return writer, log_dir

