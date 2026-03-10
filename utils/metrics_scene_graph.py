from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping


def evaluate_expansion(scene_g: Mapping[str, Any], kg_g: Mapping[str, Any]) -> Dict[str, float]:
    """
    Avalia o ganho semântico entre scene graph e knowledge graph.
    """
    scene_labels = {n["label"].lower().strip() for n in scene_g.get("nodes", [])}
    if not scene_labels:
        return {"expansion_ratio": 0.0}

    kg_expanded_entities = {edge["obj"] for edge in kg_g.get("factual_edges", [])}
    expansion = len(kg_expanded_entities) / len(scene_labels)
    return {"expansion_ratio": float(expansion)}


def compute_mean_hypernym_count(scene_g: Mapping[str, Any], kg_g: Mapping[str, Any]) -> Dict[str, float]:
    """
    Calcula o número médio de hiperônimos (is_a) por objeto da cena.
    """
    scene_labels = {n["label"].lower().strip() for n in scene_g.get("nodes", [])}
    if not scene_labels:
        return {"mean_hypernym_count": 0.0}

    hypernym_counter = {label: 0 for label in scene_labels}
    for edge in kg_g.get("factual_edges", []):
        sub = edge.get("sub", "").lower().strip()
        rel = edge.get("rel", "").lower().strip()
        if rel == "is_a" and sub in hypernym_counter:
            hypernym_counter[sub] += 1

    total_hypernyms = sum(hypernym_counter.values())
    mean_hypernyms = total_hypernyms / len(scene_labels)
    return {"mean_hypernym_count": float(mean_hypernyms)}


def evaluate_compare_graphs(scene_g: Mapping[str, Any], kg_g: Mapping[str, Any]) -> Dict[str, float]:
    """
    Compara Scene Graph com Knowledge Graph em termos estruturais/semânticos.
    """
    scene_labels = {node["label"].lower().strip() for node in scene_g.get("nodes", [])}
    kg_entities = {ent.lower().strip() for ent in kg_g.get("entities", [])}

    if len(scene_labels) == 0:
        semantic_coverage = 0.0
    else:
        semantic_coverage = len(scene_labels.intersection(kg_entities)) / len(scene_labels)

    if len(kg_entities) == 0:
        entity_recall = 0.0
    else:
        entity_recall = len(scene_labels.intersection(kg_entities)) / len(kg_entities)

    sg_relations = {
        (
            scene_g["nodes"][edge["source"]]["label"].lower().strip(),
            edge["relation"].lower().strip(),
            scene_g["nodes"][edge["target"]]["label"].lower().strip(),
        )
        for edge in scene_g.get("edges", [])
        if edge["source"] < len(scene_g.get("nodes", []))
        and edge["target"] < len(scene_g.get("nodes", []))
    }

    kg_relations = {
        (rel[0].lower().strip(), rel[1].lower().strip(), rel[2].lower().strip())
        for rel in kg_g.get("relations", [])
        if len(rel) == 3
    }

    if len(sg_relations) == 0:
        relation_consistency = 0.0
    else:
        relation_consistency = len(sg_relations.intersection(kg_relations)) / len(sg_relations)

    num_nodes = len(scene_g.get("nodes", []))
    num_edges = len(scene_g.get("edges", []))

    if num_nodes <= 1:
        structural_density = 0.0
    else:
        max_possible_edges = num_nodes * (num_nodes - 1)
        structural_density = num_edges / max_possible_edges

    return {
        "semantic_coverage": float(semantic_coverage),
        "entity_recall": float(entity_recall),
        "relation_consistency": float(relation_consistency),
        "structural_density": float(structural_density),
        "num_nodes": float(num_nodes),
        "num_edges": float(num_edges),
    }


def salvar_recall_results(
    recall_results: Mapping[str, float],
    filename: str = "recall_metrics.json",
    directory: str = "results",
) -> str:
    """
    Salva os resultados de Recall@K em um arquivo JSON com metadados.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

    path = os.path.join(directory, filename)
    data_to_save = {
        "timestamp": datetime.now().isoformat(),
        "experiment_info": {
            "model": "LoRA-Aligner-v1",
            "visual_encoder": "DinoV3",
            "text_encoder": "Qwen-7B-Embedder",
        },
        "metrics": dict(recall_results),
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print(f" Métricas de Recall salvas com sucesso em: {path}")
    return path

