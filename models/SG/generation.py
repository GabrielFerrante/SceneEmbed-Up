import torch
import torch.nn.functional as F
from models.aligners.lora_cross_attention import calculate_retrieval_score
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
        
        # 1. Extração: retorna [1, 768, 224, 224]
        _, hr_feat = self.encoder.extract_features(image.unsqueeze(0).to("cuda")) 
                    
        # 2. Pooling
        # Reduzimos aqui para 32x32
        hr_feat_small = torch.nn.functional.adaptive_avg_pool2d(hr_feat, (32, 32))
                    
        # 3. Squeeze e Transpose: [1, 768, 32, 32] -> [768, 1024] -> [1024, 768]
        # Aqui removemos o batch do loop para achatar
        c, h, w = hr_feat_small.shape[1:]
        flat_feat = hr_feat_small.squeeze(0).reshape(c, -1).transpose(0, 1)
        
        visual_input = flat_feat.unsqueeze(0)

        # 2. Refinamento de Nós com Pesos de Atenção
        node_queries = self.embedder.embed_components([candidate_nodes], normalize=False)
        
        # ALTERAÇÃO 1: Recebendo embeddings E pesos do Aligner
        node_embeddings_refined, node_attn_weights, _ = self.aligner(visual_input, node_queries)

        # Formato padronizado do Scene Graph:
        # - nodes: lista com ids contíguos (0..N-1)
        # - edges: {source, target, relation, confidence}
        scene_graph = {"nodes": [], "edges": []}

        # 3. Criação de Nós (filtrados por score)
        kept_original_ids = []
        for i, label in enumerate(candidate_nodes):
            v_aligned = node_embeddings_refined[0, i]
            t_original = node_queries[0, i]
            
            score = calculate_retrieval_score(v_aligned, t_original)
            
            if score > self.threshold:
                kept_original_ids.append(i)
                scene_graph["nodes"].append({
                    # `id` contíguo (índice no array final de nós)
                    "id": len(scene_graph["nodes"]),
                    "label": label,
                    "embedding": v_aligned,
                    "attn_weights": node_attn_weights[0, i], # Guardando os pesos para análise
                    "score": score.item()
                })

        # 4. Inferência de Relações Direcionais
        rel_queries = self.embedder.embed_components([candidate_relations], normalize=False)
        rel_embeddings_refined, _, _ = self.aligner(visual_input, rel_queries)

        for node_a in scene_graph["nodes"]:
            for node_b in scene_graph["nodes"]:
                if node_a["id"] == node_b["id"]:
                    continue
                
                # ALTERAÇÃO 2: Contexto direcional via Cross-Attention em vez de média
                # Isso garante que (A, rel, B) != (B, rel, A)
                pair_context = self._compute_directional_context(node_a["embedding"], node_b["embedding"])
                
                for j, rel_label in enumerate(candidate_relations):
                    v_rel_aligned = rel_embeddings_refined[0, j]
                    
                    # Score de validação da aresta
                    rel_score = calculate_retrieval_score(v_rel_aligned, pair_context)
                    
                    if rel_score > 0.55: # Threshold levemente ajustado para atenção
                        scene_graph["edges"].append({
                            "source": int(node_a["id"]),
                            "relation": rel_label,
                            "target": int(node_b["id"]),
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
