

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
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
import json
import os
from datetime import datetime

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
    

class KnowledgeGraphGenerator:
    def __init__(self, qwen_model, qwen_tokenizer):
        """
        Utiliza a LLM (Qwen) para extrair fatos sobre os objetos detectados.
        """
        self.model = qwen_model
        self.tokenizer = qwen_tokenizer
        self.device = next(qwen_model.parameters()).device

    def expand_node(self, node_label: str, max_facts: int = 3):
        """
        Consulta o conhecimento do Qwen sobre um objeto específico.
        """
        prompt = f"Descreva 3 fatos curtos e fundamentais sobre o objeto '{node_label}' no formato: Sujeito | Relação | Objeto. Exemplo: Gato | é um | animal."
        
        # Simulação de inferência (ajuste conforme a chamada do seu Qwen)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Aqui você faria o parsing da string para extrair as tripletas
        return response
    
    @torch.no_grad()
    def generate_from_scene(self, scene_graph: dict):
        """
        Transforma os nós de um Scene Graph em um Knowledge Graph expandido.
        """
        knowledge_graph = {
            "entities": [],
            "factual_edges": []
        }
        
        # Pega as labels únicas detectadas na cena
        detected_labels = list(set([node['label'] for node in scene_graph['nodes']]))
        
        for label in detected_labels:
            # Adiciona a entidade base
            knowledge_graph["entities"].append(label)
            
            # Prompt estruturado para evitar alucinações de formato
            prompt = f"<|im_start|>system\nVocê é um extrator de fatos ontológicos.<|im_end|>\n" \
                     f"<|im_start|>user\nListe 2 fatos sobre '{label}' no formato: Substantivo | Relação | Objeto.<|im_end|>\n" \
                     f"<|im_start|>assistant\n"
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(**inputs, max_new_tokens=40, temperature=0.1)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parsing robusto
            for line in response.split('\n'):
                if "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) == 3:
                        knowledge_graph["factual_edges"].append({
                            "sub": parts[0], "rel": parts[1], "obj": parts[2]
                        })
                
        return knowledge_graph

def salvar_grafos_json(scene_graph, knowledge_graph, comparison_metrics=None, filename=None):
    directory = "results"
    if not os.path.exists(directory): os.makedirs(directory)

    if filename is None:
        filename = f"graph_{datetime.now().strftime('%H%M%S')}.json"
    
    # Estrutura completa para persistência
    data_to_save = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "metrics": comparison_metrics or {} # Inclui a 'semantic_coverage' aqui
        },
        "scene_graph": scene_graph,
        "knowledge_graph": knowledge_graph
    }

    with open(os.path.join(directory, filename), 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    
    print(f" Dados gravados em {filename}")
