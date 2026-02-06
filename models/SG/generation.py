

"""

Attention-based Scene Graph Generation

Para montar o grafo apenas com os embeddings que você já extraiu, você deve seguir estes 3 passos:

1. Clusterização Semântica (Criação dos Nós)
Como você tem 200 patches do DINO, alguns deles pertencem ao mesmo objeto.

Ação: Você projeta todos os patches para o espaço de 4096 (usando o SceneGraphAligner).

Ação: Use o embedding do Qwen3 (ex: "cat") como uma query. Calcule a similaridade de cada um dos 200 patches com esse embedding.

Resultado: Os patches com alta similaridade formam o "Nó" do objeto na imagem de forma orgânica (soft-mask).

2. Matriz de Adjacência (Criação das Arestas)
Para saber se o "Nó A" se relaciona com o "Nó B", você olha para a Atenção Cruzada entre eles.

Cálculo: Se os patches que compõem o "Gato" e os patches que compõem a "Mesa" possuem alta atenção mútua nas camadas profundas do DINO, existe uma aresta entre eles.
"""
import torch
import torch.nn.functional as F
from models.SG.projection import LoRACrossAttentionAligner, calculate_retrieval_score

class SceneGraphGenerator:
    def __init__(self, dino_encoder, qwen_embedder, aligner, threshold=0.3):
        self.encoder = dino_encoder
        self.embedder = qwen_embedder
        self.aligner = aligner
        self.threshold = threshold
        self.device = next(aligner.parameters()).device
        self.dtype = qwen_embedder.dtype

    @torch.no_grad()
    def generate(self, image, candidate_nodes: list, candidate_relations: list):
        """
        candidate_nodes: ['gato', 'mesa', 'sofa']
        candidate_relations: ['em cima de', 'perto de']
        """
        self.aligner.eval()
        
        # 1. Extração de Features HR (AnyUp)
        # Saída: [1, 768, 686, 960]
        _, hr_features = self.encoder.extract_features(image)
        B, C, H, W = hr_features.shape
        
        # Prepara entrada para o Aligner: [B, Tokens, 768]
        visual_input = hr_features.view(B, C, -1).transpose(1, 2).to(self.dtype)

        # 2. Utiliza o Aligner para projetar e refinar os nós
        # Passamos a lista de objetos como Query para a Cross-Attention
        # text_queries: [1, N_nós, 4096]
        node_queries = self.embedder.embed_components([candidate_nodes], normalize=False)
        
        # Aqui usamos o forward que definimos no arquivo de projeção
        # node_embeddings_refined: [1, N_nós, 4096]
        node_embeddings_refined = self.aligner(visual_input, node_queries)

        scene_graph = {"nodes": [], "edges": []}

        # 3. Validação e Criação de Nós utilizando calculate_retrieval_score
        # Comparamos o embedding refinado (visual) com o original (textual)
        for i, label in enumerate(candidate_nodes):
            v_aligned = node_embeddings_refined[0, i] # [4096]
            t_original = node_queries[0, i]           # [4096]
            
            # Usando sua função do arquivo de projeção
            score = calculate_retrieval_score(v_aligned, t_original)
            
            if score > self.threshold:
                scene_graph["nodes"].append({
                    "id": i,
                    "label": label,
                    "embedding": v_aligned,
                    "score": score.item()
                })

        # 4. Inferência de Relações (Arestas)
        # Para cada par de nós detectados, testamos os predicados (relações)
        rel_queries = self.embedder.embed_components([candidate_relations], normalize=False)
        # Refinamos as relações contra a imagem também
        rel_embeddings_refined = self.aligner(visual_input, rel_queries)

        for node_a in scene_graph["nodes"]:
            for node_b in scene_graph["nodes"]:
                if node_a["id"] == node_b["id"]: continue
                
                # Criamos um "contexto de relação" (média dos dois nós)
                # ou poderíamos buscar a atenção entre eles.
                pair_context = (node_a["embedding"] + node_b["embedding"]) / 2
                
                for j, rel_label in enumerate(candidate_relations):
                    v_rel_aligned = rel_embeddings_refined[0, j]
                    
                    # Verificamos se a relação visual "bate" com o contexto do par
                    rel_score = calculate_retrieval_score(v_rel_aligned, pair_context)
                    
                    if rel_score > 0.6: # Threshold para arestas
                        scene_graph["edges"].append({
                            "subject": node_a["label"],
                            "relation": rel_label,
                            "object": node_b["label"],
                            "confidence": rel_score.item()
                        })

        return scene_graph
    
device = "cuda"
aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096)
aligner.load_state_dict(torch.load("lora_cross_aligner_weights.pth"), strict=False)
aligner.to(device).eval()