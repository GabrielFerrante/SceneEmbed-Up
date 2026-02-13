import json
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import os

def visualizar_e_salvar_grafo(caminho_json):
    # Carregar o arquivo JSON
    with open(caminho_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    sg = data['scene_graph']
    kg = data['knowledge_graph']
    
    #  Construir o Grafo no NetworkX
    G = nx.DiGraph()
    
    # Adicionar Nós/Arestas do Scene Graph (Visual)
    for node in sg['nodes']:
        G.add_node(node['label'], type='scene', color='#3498db')
    for edge in sg['edges']:
        G.add_edge(edge['subject'], edge['object'], label=edge['relation'], color='#2980b9')
        
    # Adicionar Nós/Arestas do Knowledge Graph (Semântico)
    for fact in kg['factual_edges']:
        if not G.has_node(fact['sub']):
            G.add_node(fact['sub'], type='knowledge', color='#2ecc71')
        if not G.has_node(fact['obj']):
            G.add_node(fact['obj'], type='knowledge', color='#2ecc71')
        G.add_edge(fact['sub'], fact['obj'], label=fact['rel'], color='#27ae60')

    # Conversão PyTorch Geometric
    all_nodes = list(G.nodes())
    node_map = {node: i for i, node in enumerate(all_nodes)}
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in G.edges()]).t().contiguous()
    pyg_data = Data(edge_index=edge_index)

    # Renderização
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.6, iterations=70) # Layout mais espaçado
    
    node_colors = [G.nodes[n].get('color', '#95a5a6') for n in G.nodes()]
    edge_colors = [G[u][v].get('color', '#bdc3c7') for u, v in G.edges()]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, 
            node_size=3500, font_size=9, font_weight='bold', 
            edge_color=edge_colors, width=1.5, arrowsize=18, alpha=0.9)
    
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)

    coverage = data['metadata']['metrics'].get('semantic_coverage', 0)
    plt.title(f"Visualização de Grafos | Cobertura: {coverage:.2%}\nArquivo: {os.path.basename(caminho_json)}")

    # 5. Salvar Imagem com o mesmo nome do JSON
    caminho_png = caminho_json.replace('.json', '.png')
    plt.savefig(caminho_png, bbox_inches='tight', dpi=300)
    print(f"Imagem do grafo salva em: {caminho_png}")
    
    plt.close() # Fecha a figura para liberar memória (importante em loops)
    return pyg_data

# Para rodar em lote na sua pasta de resultados:
for arquivo in os.listdir("../results"):
     if arquivo.endswith(".json"):
         visualizar_e_salvar_grafo(os.path.join("results", arquivo))