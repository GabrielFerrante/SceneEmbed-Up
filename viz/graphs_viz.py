import os

from utils.graph_viz import visualizar_e_salvar_grafo

# Para rodar em lote na sua pasta de resultados:
for arquivo in os.listdir("../results"):
    if arquivo.endswith(".json"):
        visualizar_e_salvar_grafo(os.path.join("results", arquivo))