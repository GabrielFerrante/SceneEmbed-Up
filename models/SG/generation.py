

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
        
    def _compute_directional_context(self, query_node, context_node):
        """
        Substitui a média vetorial por Cross-Attention entre os nós.
        Faz com que o Subject (query) extraia contexto do Object (key/value).
        """
        # Dimensões: [1, 1, 4096]
        q = query_node.unsqueeze(0).unsqueeze(0)
        k = v = context_node.unsqueeze(0).unsqueeze(0)
        
        d_k = q.size(-1)
        # Scaled Dot-Product Attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=self.dtype))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # O contexto resultante é direcional: como o Sujeito se relaciona com o Objeto
        directional_context = torch.matmul(attn_weights, v)
        return directional_context.squeeze()

    @torch.no_grad()
    def generate(self, image, candidate_nodes: list, candidate_relations: list):
        self.aligner.eval()
        
        # 1. Extração de Features Visuais
        _, hr_features = self.encoder.extract_features(image)
        B, C, H, W = hr_features.shape
        visual_input = hr_features.view(B, C, -1).transpose(1, 2).to(self.dtype)

        # 2. Refinamento de Nós com Pesos de Atenção
        node_queries = self.embedder.embed_components([candidate_nodes], normalize=False)
        
        # ALTERAÇÃO 1: Recebendo embeddings E pesos do Aligner
        node_embeddings_refined, node_attn_weights = self.aligner(visual_input, node_queries)

        scene_graph = {"nodes": [], "edges": []}

        # 3. Criação de Nós
        for i, label in enumerate(candidate_nodes):
            v_aligned = node_embeddings_refined[0, i]
            t_original = node_queries[0, i]
            
            score = calculate_retrieval_score(v_aligned, t_original)
            
            if score > self.threshold:
                scene_graph["nodes"].append({
                    "id": i,
                    "label": label,
                    "embedding": v_aligned,
                    "attn_weights": node_attn_weights[0, i], # Guardando os pesos para análise
                    "score": score.item()
                })

        # 4. Inferência de Relações Direcionais
        rel_queries = self.embedder.embed_components([candidate_relations], normalize=False)
        rel_embeddings_refined, _ = self.aligner(visual_input, rel_queries)

        for node_a in scene_graph["nodes"]:
            for node_b in scene_graph["nodes"]:
                if node_a["id"] == node_b["id"]: continue
                
                # ALTERAÇÃO 2: Contexto direcional via Cross-Attention em vez de média
                # Isso garante que (A, rel, B) != (B, rel, A)
                pair_context = self._compute_directional_context(node_a["embedding"], node_b["embedding"])
                
                for j, rel_label in enumerate(candidate_relations):
                    v_rel_aligned = rel_embeddings_refined[0, j]
                    
                    # Score de validação da aresta
                    rel_score = calculate_retrieval_score(v_rel_aligned, pair_context)
                    
                    if rel_score > 0.55: # Threshold levemente ajustado para atenção
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
        prompt = f"Describe three short, fundamental facts about the object '{node_label}' in the format: Subject | Relation | Object. Example: Cat | is a | animal."
        
        # Simulação de inferência (ajuste conforme a chamada do seu Qwen)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs, max_new_tokens=50)
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

        
        # Aqui você faria o parsing da string para extrair as tripletas
        return response
    
    @torch.no_grad()
    def generate_from_scene(self, scene_graph: dict):

        knowledge_graph = {
            "entities": set(),
            "factual_edges": []
        }

        detected_labels = list(set(
            node['label'].lower().strip()
            for node in scene_graph['nodes']
        ))

        for label in detected_labels:

            knowledge_graph["entities"].add(label)

            prompt = (
                f"<|im_start|>system\n"
                f"List exactly 2 universal, taxonomy-level facts about '{label}'. "
                f"Avoid opinions, abilities, or cultural associations."
                f"<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Use only class or category relations. "
                f"Format strictly as: Subject | is_a | Object. "
                f"Always use exactly the word '{label}' as the Subject."
                f"<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt"
            ).to(self.device)

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=40,
                temperature=0.1
            )

            response = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True
            )

            for line in response.split('\n'):
                if "|" not in line:
                    continue

                parts = [p.strip().lower() for p in line.split("|")]

                if len(parts) != 3:
                    continue

                _, rel, obj = parts

                rel = "is_a"  # força consistência

                knowledge_graph["factual_edges"].append({
                    "sub": label,
                    "rel": rel,
                    "obj": obj
                })

                knowledge_graph["entities"].add(obj)

        knowledge_graph["entities"] = list(knowledge_graph["entities"])

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
