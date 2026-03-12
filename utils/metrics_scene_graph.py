from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping


def _iter_sg_triplets(scene_g: Mapping[str, Any]) -> Iterable[tuple[str, str, str]]:
    """
    Extrai tripletas (sub, rel, obj) do Scene Graph, aceitando:
    - Formato novo: edges com {source, target, relation} referenciando índices em nodes
    - Formato legado: edges com {subject, object, relation} diretamente por label
    """
    nodes = scene_g.get("nodes", []) or []
    for edge in scene_g.get("edges", []) or []:
        if not isinstance(edge, Mapping):
            continue

        # Formato novo
        if "source" in edge and "target" in edge:
            try:
                s_idx = int(edge.get("source"))
                t_idx = int(edge.get("target"))
                if s_idx < 0 or t_idx < 0 or s_idx >= len(nodes) or t_idx >= len(nodes):
                    continue
                sub = str(nodes[s_idx].get("label", "")).lower().strip()
                obj = str(nodes[t_idx].get("label", "")).lower().strip()
                rel = str(edge.get("relation", "")).lower().strip()
            except Exception:
                continue
            if sub and rel and obj:
                yield (sub, rel, obj)
            continue

        # Formato legado
        sub = str(edge.get("subject", "")).lower().strip()
        obj = str(edge.get("object", "")).lower().strip()
        rel = str(edge.get("relation", "")).lower().strip()
        if sub and rel and obj:
            yield (sub, rel, obj)


def _iter_kg_triplets(kg_g: Mapping[str, Any]) -> Iterable[tuple[str, str, str]]:
    """
    Extrai tripletas do Knowledge Graph aceitando:
    - Formato atual do projeto: factual_edges com {sub, rel, obj}
    - Formato alternativo: relations como lista de tripletas
    """
    for edge in kg_g.get("factual_edges", []) or []:
        if not isinstance(edge, Mapping):
            continue
        sub = str(edge.get("sub", "")).lower().strip()
        rel = str(edge.get("rel", "")).lower().strip()
        obj = str(edge.get("obj", "")).lower().strip()
        if sub and rel and obj:
            yield (sub, rel, obj)

    for rel in kg_g.get("relations", []) or []:
        if isinstance(rel, (list, tuple)) and len(rel) == 3:
            yield (str(rel[0]).lower().strip(), str(rel[1]).lower().strip(), str(rel[2]).lower().strip())


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

    sg_relations = set(_iter_sg_triplets(scene_g))
    kg_relations = set(_iter_kg_triplets(kg_g))

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

