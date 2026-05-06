from __future__ import annotations

import json
import os
from typing import Dict

from utils.io_utils import ensure_dir


def salvar_recall_results(results: Dict[str, float], path: str = "results/recall_metrics.json") -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados salvos em {path}")
