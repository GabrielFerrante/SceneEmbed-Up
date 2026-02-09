import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from models.SG.projection import LoRACrossAttentionAligner, calculate_retrieval_score
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.generation import SceneGraphGenerator
from data.data_utils_pytorch import create_all_dataloaders

class SceneGraphEvaluator:
    def __init__(self, generator, device="cuda"):
        self.generator = generator
        self.device = device
        self.dtype = generator.dtype

    @torch.no_grad()
    def evaluate_projection(self, dataloader, k_values=[1, 5, 10]):
        """
        Calcula Recall@K para o alinhamento Imagem-Texto.
        Indica se o Aligner coloca os pares corretos próximos no espaço latente.
        """
        self.generator.aligner.eval()
        img_embeddings = []
        text_embeddings = []

        print("Extraindo Embeddings para Recall@K")
        for images, texts in tqdm(dataloader):
            # Features Visuais
            _, hr_feat = self.generator.encoder.extract_features(images)
            visual_input = hr_feat.view(hr_feat.size(0), hr_feat.size(1), -1).transpose(1, 2).to(self.dtype)
            
            # Embeddings de Texto (Targets)
            formatted_texts = [[t] for t in texts]
            t_queries = self.generator.embedder.embed_components(formatted_texts)
            
            # Projeção (Alinhamento)
            # Usamos o primeiro token de cada query como representação do par
            v_refined = self.generator.aligner(visual_input, t_queries)
            
            img_embeddings.append(F.normalize(v_refined.squeeze(1), dim=-1).cpu())
            text_embeddings.append(F.normalize(t_queries.squeeze(1), dim=-1).cpu())

        # Matriz de Similaridade Global [N_total, N_total]
        img_embeddings = torch.cat(img_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        sim_matrix = torch.matmul(img_embeddings, text_embeddings.T)

        results = {}
        for k in k_values:
            # Verifica se a diagonal (índice i == j) está no top-k de cada linha
            top_k = torch.topk(sim_matrix, k=k, dim=1).indices
            correct = torch.any(top_k == torch.arange(len(sim_matrix)).unsqueeze(1), dim=1)
            results[f"Recall@{k}"] = correct.float().mean().item()
            
        return results

    def evaluate_graph_structure(self, generated_graph, ground_truth_graph):
        """
        Compara o grafo gerado com um grafo de referência (GT).
        Métricas: Precisão de Nós e Arestas.
        """
        gen_nodes = set([n['label'] for n in generated_graph['nodes']])
        gt_nodes = set(ground_truth_graph['nodes'])
        
        # Precisão de Objetos (Nós)
        hit_nodes = gen_nodes.intersection(gt_nodes)
        node_precision = len(hit_nodes) / len(gen_nodes) if gen_nodes else 0
        node_recall = len(hit_nodes) / len(gt_nodes) if gt_nodes else 0
        
        # Precisão de Relações (Arestas)
        # Formato esperado da aresta: (sujeito, relação, objeto)
        gen_edges = set([(e['subject'], e['relation'], e['object']) for e in generated_graph['edges']])
        gt_edges = set(ground_truth_graph['edges'])
        
        hit_edges = gen_edges.intersection(gt_edges)
        edge_precision = len(hit_edges) / len(gen_edges) if gen_edges else 0
        
        return {
            "node_precision": node_precision,
            "node_recall": node_recall,
            "edge_precision": edge_precision,
            "f1_score": 2 * (node_precision * node_recall) / (node_precision + node_recall + 1e-8)
        }
        

# --- Exemplo de uso ---
if __name__ == "__main__":
    import os
    from data.data_utils_pytorch import create_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Instanciar Encoders (Pesados)
    dino = DinoSceneEncoder(device=device)
    qwen = QwenSceneEmbedder(device=device)

    # 2. Setup Aligner (Leve)
    # Certifique-se que o visual_dim condiz com o encoder usado (768 para AnyUp/Dino-B)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096)
    
    weights_path = "checkpoints/aligner_epoch_10.pth" # Ou seu peso final
    if os.path.exists(weights_path):
        aligner.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f" Pesos carregados de {weights_path}")
    else:
        print("Checkpoint não encontrado. Rodando com pesos aleatórios para teste.")

    aligner.to(device).eval()
    
    # 3. Gerador e Avaliador
    generator = SceneGraphGenerator(
        dino_encoder=dino, 
        qwen_embedder=qwen, 
        aligner=aligner, 
        threshold=0.3
    )
    evaluator = SceneGraphEvaluator(generator)
    
    # 4. Carregar Dataloader de Teste
    # Substitua pelo caminho correto do seu conjunto de teste
    test_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=2, num_workers=4, t="test")

    # --- EXECUÇÃO DA AVALIAÇÃO ---

    # A. Métrica de Projeção (Recall Global)
    print("\n--- Avaliando Alinhamento (Recall@K) ---")
    recall_results = evaluator.evaluate_projection(test_dataloader, k_values=[1, 5, 10])
    print(f"Resultados de Busca: {recall_results}")

    # B. Exemplo Qualitativo e Espacial (mIoU)
    # Vamos pegar um batch do test_dataloader para um teste visual
    print("\n--- Teste Qualitativo em Imagem Real ---")
    images, texts = next(iter(test_dataloader))
    img_teste = images[0]  # Pega a primeira imagem do batch
    label_teste = texts[0] # Texto real da imagem
    
    # Simulamos uma lista de candidatos incluindo o real para ver se o modelo acha
    candidatos = [label_teste, "carro", "árvore", "pessoa"] 
    relacoes = ["perto de", "em cima de"]
    
    graph = generator.generate(img_teste, candidatos, relacoes)
    
    print(f"Grafo Gerado para '{label_teste}':")
    print(f"  Nós Detectados: {[n['label'] for n in graph['nodes']]}")
    print(f"  Relações: {graph['edges']}")


    print("\n Avaliação finalizada.")