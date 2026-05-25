"""
scene_graph_viz.py
------------------
Visualização de scene graphs gerados pelo RelTR sobre imagens recuperadas
pelo aligner COYO.

Produz duas saídas por consulta:
  1. Imagem com bounding boxes das entidades (matplotlib)
  2. Grafo dirigido com nós/arestas anotados por confiança (NetworkX + matplotlib)
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # sem display — salva em arquivo
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np
from PIL import Image

from models.SG.reltr_wrapper import SceneGraph, SceneGraphTriple
from models.rerank.semantic_rerank import ReRankResult


def _box_to_pixels(
    box: Tuple[float, float, float, float],
    w: int,
    h: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    return int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)


def draw_boxes(
    image: Image.Image,
    sg: SceneGraph,
    ax: plt.Axes,
    retrieval_score: Optional[float] = None,
    query_text: Optional[str] = None,
) -> None:
    """
    Desenha bounding boxes das entidades sobre a imagem.

    Cada tripla contribui com dois boxes (sujeito e objeto) coloridos
    proporcionalmente à confiança da tripla.

    Parameters
    ----------
    image:
        Imagem PIL original.
    sg:
        SceneGraph com as triplas e dimensões da imagem.
    ax:
        Eixo matplotlib onde desenhar.
    retrieval_score:
        Score de similaridade do retrieval (exibido no título).
    query_text:
        Texto da query de retrieval (exibido no título).
    """
    ax.imshow(np.array(image))
    ax.axis("off")

    title = ""
    if query_text:
        title += f'Query: "{query_text}"\n'
    if retrieval_score is not None:
        title += f"Retrieval score: {retrieval_score:.4f}"
    if title:
        ax.set_title(title, fontsize=9, pad=4)

    cmap = plt.cm.get_cmap("RdYlGn")
    w, h = sg.image_width, sg.image_height

    for triple in sg.triples:
        color = cmap(triple.confidence)

        for entity, box, label in [
            (triple.subject, triple.subject_box, triple.subject),
            (triple.object,  triple.object_box,  triple.object),
        ]:
            x1, y1, x2, y2 = _box_to_pixels(box, w, h)
            rect = mpatches.FancyBboxPatch(
                (x1, y1), x2 - x1, y2 - y1,
                boxstyle="round,pad=2",
                linewidth=1.5,
                edgecolor=color,
                facecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                x1 + 2, y1 - 3,
                label,
                fontsize=6,
                color="white",
                backgroundcolor=(*color[:3], 0.7),
                clip_on=True,
            )


def draw_scene_graph(
    sg: SceneGraph,
    ax: plt.Axes,
    highlight_nodes: Optional[List[str]] = None,
) -> None:
    """
    Desenha o grafo de cena como dígrafo no eixo fornecido.

    Parameters
    ----------
    sg:
        SceneGraph a visualizar.
    ax:
        Eixo matplotlib.
    highlight_nodes:
        Nomes de nós a destacar (ex.: entidades mencionadas na query).
    """
    G = sg.to_networkx()

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No triples detected", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return

    pos = nx.spring_layout(G, seed=42, k=2.0)

    highlight_nodes = set(highlight_nodes or [])
    node_colors = [
        "#FF6B35" if n in highlight_nodes else "#4A90D9"
        for n in G.nodes()
    ]

    edge_labels = {(u, v): d["label"] for u, v, d in G.edges(data=True)}
    edge_widths = [d.get("confidence", 0.5) * 3 for _, _, d in G.edges(data=True)]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=800, alpha=0.9)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=7, font_color="white", font_weight="bold")
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=edge_widths,
        edge_color="#555555",
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax, font_size=6)
    ax.axis("off")


def visualize_retrieval_result(
    image: Image.Image,
    sg: SceneGraph,
    query_text: str,
    retrieval_score: float,
    save_path: str,
    highlight_nodes: Optional[List[str]] = None,
) -> None:
    """
    Gera figura com duas colunas: imagem com boxes | grafo de cena.

    Parameters
    ----------
    image:
        Imagem PIL recuperada.
    sg:
        SceneGraph gerado pelo RelTR.
    query_text:
        Texto da query de retrieval.
    retrieval_score:
        Score de similaridade do aligner.
    save_path:
        Caminho do arquivo de saída (.png).
    highlight_nodes:
        Entidades a destacar no grafo (ex.: palavras-chave da query).
    """
    fig, (ax_img, ax_graph) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    draw_boxes(image, sg, ax_img, retrieval_score=retrieval_score, query_text=query_text)
    draw_scene_graph(sg, ax_graph, highlight_nodes=highlight_nodes)

    ax_graph.set_title(
        f"{len(sg.triples)} triples detected",
        fontsize=9, color="white", pad=4,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved to {save_path}")


def visualize_rerank_result(
    image: Image.Image,
    result: ReRankResult,
    query_text: str,
    save_path: str,
    alpha: float,
) -> None:
    """
    Visualizacao explicativa do re-ranking semantico.

    Layout em 3 colunas:
      - imagem com bboxes (triplas top-K destacadas em verde)
      - scene graph
      - painel textual com scores e triplas que mais alinharam com a query

    Parameters
    ----------
    image:
        Imagem PIL recuperada.
    result:
        ReRankResult com scores antes/depois e triplas explicativas.
    query_text:
        Query do retrieval.
    save_path:
        Caminho do PNG.
    alpha:
        Peso usado na combinacao (mostrado no titulo).
    """
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor("#1a1a2e")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 1.0, 0.9])
    ax_img = fig.add_subplot(gs[0, 0])
    ax_graph = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    draw_boxes(image, result.sg, ax_img, retrieval_score=result.score_final, query_text=query_text)

    # destaca entidades das top-triples
    highlight = set()
    for triple, _ in result.top_triples:
        highlight.add(triple.subject)
        highlight.add(triple.object)
    draw_scene_graph(result.sg, ax_graph, highlight_nodes=list(highlight))
    ax_graph.set_title(f"{len(result.sg.triples)} triples", fontsize=9, color="white", pad=4)

    # painel de explicabilidade
    ax_text.axis("off")
    delta = result.rank_before - result.rank_after
    arrow = "->" if delta == 0 else (f"^{delta}" if delta > 0 else f"v{abs(delta)}")
    lines = [
        f'Query: "{query_text}"',
        "",
        f"Rank: #{result.rank_before}  =>  #{result.rank_after}  ({arrow})",
        "",
        f"score_dense   = {result.score_dense:.4f}",
        f"score_sg      = {result.score_sg:.4f}",
        f"score_final   = {result.score_final:.4f}",
        f"(alpha = {alpha:.2f})",
        "",
        "Top triples aligned with query:",
    ]
    for i, (triple, sim) in enumerate(result.top_triples, 1):
        lines.append(f"  {i}. {triple.subject} -[{triple.predicate}]-> {triple.object}")
        lines.append(f"     sim={sim:.3f}  conf={triple.confidence:.3f}")

    ax_text.text(
        0.02, 0.98, "\n".join(lines),
        transform=ax_text.transAxes,
        fontsize=9, color="white", va="top", ha="left",
        family="monospace",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[viz] Saved to {save_path}")
