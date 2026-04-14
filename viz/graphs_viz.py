import os

from utils.graph_viz import visualizar_e_salvar_grafo

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

for arquivo in sorted(os.listdir(RESULTS_DIR)):
    if arquivo.endswith(".json"):
        caminho = os.path.join(RESULTS_DIR, arquivo)
        print(f"\nProcessando: {arquivo}")
        visualizar_e_salvar_grafo(caminho)
