from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import torch
from torch_geometric.data import Data


def _build_graph(sg: Dict[str, Any], kg: Dict[str, Any]) -> nx.DiGraph:
    """
    Constroi grafo NetworkX combinando Scene Graph + Knowledge Graph.

    Nós do SG ficam azuis, nós/arestas do KG ficam verdes.
    """
    G = nx.DiGraph()

    # ── Scene Graph ─────────────────────────────────────────────────────
    for node in sg.get("nodes", []):
        G.add_node(node["label"], type="scene", color="#3498db")

    for edge in sg.get("edges", []):
        # Formato novo: {source, target, relation, confidence}
        if "source" in edge and "target" in edge:
            try:
                sub_label = sg["nodes"][edge["source"]]["label"]
                obj_label = sg["nodes"][edge["target"]]["label"]
            except Exception:
                continue
            G.add_edge(
                sub_label, obj_label,
                label=edge.get("relation", ""),
                color="#2980b9",
            )
        else:
            # Formato legado: {subject, object, relation}
            G.add_edge(
                edge["subject"], edge["object"],
                label=edge.get("relation", ""),
                color="#2980b9",
            )

    # ── Knowledge Graph ─────────────────────────────────────────────────
    for fact in kg.get("factual_edges", []):
        if not G.has_node(fact["sub"]):
            G.add_node(fact["sub"], type="knowledge", color="#2ecc71")
        if not G.has_node(fact["obj"]):
            G.add_node(fact["obj"], type="knowledge", color="#2ecc71")
        G.add_edge(fact["sub"], fact["obj"], label=fact["rel"], color="#27ae60")

    return G


def _render_graph(
    G: nx.DiGraph,
    title: str,
    output_path: str,
) -> None:
    """Renderiza grafo em PNG via matplotlib."""
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.6, iterations=70)

    node_colors = [G.nodes[n].get("color", "#95a5a6") for n in G.nodes()]
    edge_colors = [G[u][v].get("color", "#bdc3c7") for u, v in G.edges()]

    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color=node_colors,
        node_size=3500,
        font_size=9,
        font_weight="bold",
        edge_color=edge_colors,
        width=1.5,
        arrowsize=18,
        alpha=0.9,
    )

    edge_labels = nx.get_edge_attributes(G, "label")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    plt.title(title)
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Grafo salvo: {output_path}")


def _to_pyg(G: nx.DiGraph) -> Data:
    """Converte DiGraph NetworkX para torch_geometric.data.Data."""
    all_nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    edges = [[node_map[u], node_map[v]] for u, v in G.edges()]
    if edges:
        edge_index = torch.tensor(edges).t().contiguous()
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    return Data(edge_index=edge_index)


def visualizar_sample(
    sg: Dict[str, Any],
    kg: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
    output_path: str = "results/grafo.png",
    sample_label: str = "",
) -> Data:
    """
    Visualiza um par (Scene Graph, Knowledge Graph) e salva como PNG.

    Parameters
    ----------
    sg:
        Scene graph dict com chaves 'nodes' e 'edges'.
    kg:
        Knowledge graph dict com chaves 'factual_edges' e 'entities'.
    metrics:
        Métricas opcionais para exibir no titulo.
    output_path:
        Caminho de saida para o PNG.
    sample_label:
        Rótulo para o titulo (ex: "Sample 0", "batch0_img3").

    Returns
    -------
    torch_geometric.data.Data
    """
    G = _build_graph(sg, kg)

    coverage = 0.0
    if metrics:
        coverage = metrics.get("semantic_coverage", 0.0)

    title = f"SG + KG | Cobertura: {coverage:.2%}"
    if sample_label:
        title += f" | {sample_label}"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _render_graph(G, title, output_path)

    return _to_pyg(G)


def visualizar_e_salvar_grafo(caminho_json: str) -> list[Data]:
    """
    Carrega JSON de resultados e gera visualizações PNG.

    Suporta dois formatos:
    - **Consolidado** (sg_kg_results.json): campo 'samples' com lista de amostras.
      Gera um PNG por amostra.
    - **Individual** (resultado_batch0_img0.json): campos 'scene_graph' e
      'knowledge_graph' no topo. Gera um PNG.

    Parameters
    ----------
    caminho_json:
        Caminho para o JSON gerado pelo pipeline de avaliação.

    Returns
    -------
    list[Data]
        Estruturas PyG para cada grafo visualizado.
    """
    with open(caminho_json, "r", encoding="utf-8") as f:
        data: Any = json.load(f)

    base_name = os.path.splitext(caminho_json)[0]
    pyg_list: list[Data] = []

    # ── Formato consolidado (samples[]) ─────────────────────────────────
    if "samples" in data:
        samples = data["samples"]
        print(f"Formato consolidado: {len(samples)} amostras em {caminho_json}")
        for sample in samples:
            sid = sample.get("sample_id", len(pyg_list))
            sg = sample.get("scene_graph", {})
            kg = sample.get("knowledge_graph", {})
            metrics = sample.get("metrics", {})
            out_path = f"{base_name}_sample{sid}.png"

            pyg = visualizar_sample(sg, kg, metrics, out_path, f"Sample {sid}")
            pyg_list.append(pyg)

    # ── Formato individual (top-level scene_graph) ──────────────────────
    elif "scene_graph" in data:
        sg = data["scene_graph"]
        kg = data.get("knowledge_graph", {})
        metrics = data.get("metadata", {}).get("metrics", {})
        out_path = f"{base_name}.png"

        pyg = visualizar_sample(sg, kg, metrics, out_path, os.path.basename(caminho_json))
        pyg_list.append(pyg)

    else:
        print(f"  [AVISO] Formato nao reconhecido: {caminho_json}")

    return pyg_list
