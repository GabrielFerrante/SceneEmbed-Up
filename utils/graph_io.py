from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


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

    path = os.path.join(directory, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f" Dados gravados em {path}")
    return path

