from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def _to_jsonable(obj: Any) -> Any:
    """
    Converte recursivamente estruturas com torch/numpy para tipos serializáveis em JSON.
    Mantém dict/list/tuple e converte tensores/arrays/escalars para list/float/int.
    """
    # Importes locais para não forçar dependências no import do módulo
    try:
        import torch  # type: ignore
    except Exception:  # pragma: no cover
        torch = None  # type: ignore

    try:
        import numpy as np  # type: ignore
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    if obj is None:
        return None

    # Torch
    if torch is not None and isinstance(obj, getattr(torch, "Tensor")):
        return obj.detach().cpu().tolist()

    # NumPy
    if np is not None:
        if isinstance(obj, getattr(np, "ndarray")):
            return obj.tolist()
        if isinstance(obj, getattr(np, "generic")):
            return obj.item()

    # Tipos nativos
    if isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_to_jsonable(v) for v in obj]

    # Fallback: tenta string (melhor do que quebrar no json.dump)
    return str(obj)


def salvar_grafos_json(
    scene_graph: Dict[str, Any],
    knowledge_graph: Dict[str, Any],
    comparison_metrics: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
    directory: str = "results",
) -> str:
    """
    Persiste scene graph, knowledge graph e métricas associadas em JSON.

    Parameters
    ----------
    scene_graph:
        Dicionário contendo nós e arestas do grafo visual.
    knowledge_graph:
        Dicionário contendo entidades e relações factuais.
    comparison_metrics:
        Métricas opcionais (ex.: semantic_coverage, entity_recall, ...).
    filename:
        Nome do arquivo a ser criado. Se `None`, um nome timestampado é gerado.
    directory:
        Diretório base onde o arquivo será salvo.

    Returns
    -------
    str
        Caminho completo do arquivo JSON salvo.
    """
    os.makedirs(directory, exist_ok=True)

    if filename is None:
        filename = f"graph_{datetime.now().strftime('%H%M%S')}.json"

    data_to_save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "metrics": comparison_metrics or {},
        },
        "scene_graph": scene_graph,
        "knowledge_graph": knowledge_graph,
    }
    data_to_save = _to_jsonable(data_to_save)

    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f" Dados gravados em {path}")
    return path


def salvar_resultados_sg(
    samples: list[Dict[str, Any]],
    aggregate_metrics: Dict[str, float],
    filename: str = "sg_kg_results.json",
    directory: str = "results",
) -> str:
    """
    Persiste todos os resultados de SG/KG em um único JSON.

    Parameters
    ----------
    samples:
        Lista de dicts, cada um com scene_graph, knowledge_graph e metrics.
    aggregate_metrics:
        Métricas agregadas (médias) do eval completo.
    filename:
        Nome do arquivo de saída.
    directory:
        Diretório base.

    Returns
    -------
    str
        Caminho completo do arquivo JSON salvo.
    """
    os.makedirs(directory, exist_ok=True)

    data_to_save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_samples": len(samples),
            "aggregate_metrics": aggregate_metrics,
        },
        "samples": samples,
    }
    data_to_save = _to_jsonable(data_to_save)

    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f" Resultados SG/KG gravados em {path} ({len(samples)} amostras)")
    return path

