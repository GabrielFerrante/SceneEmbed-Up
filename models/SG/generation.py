import torch
import torch.nn.functional as F
from models.aligners.lora_cross_attention import calculate_retrieval_score
import json
import os
from datetime import datetime


def _label_connected_components(mask: torch.Tensor) -> torch.Tensor:
    # mask: [H, W] bool -> [H, W] int (0 = background); 4-connectivity
    from scipy.ndimage import label
    arr = mask.detach().cpu().numpy().astype("uint8")
    labels, _ = label(arr)
    return torch.from_numpy(labels).to(mask.device)


class SceneGraphGenerator:
    def __init__(self, dino_encoder, qwen_embedder, aligner, threshold=0.3, edge_threshold=0.95, sg_classifier_head=None, min_patches: int = 2):
        self.encoder = dino_encoder
        self.embedder = qwen_embedder
        self.aligner = aligner
        self.threshold = threshold
        self.edge_threshold = edge_threshold
        self.device = next(aligner.parameters()).device
        self.dtype = qwen_embedder.dtype
        self.sg_classifier_head = sg_classifier_head
        self.min_patches = min_patches
        
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

        # Cast para o dtype do aligner (bfloat16)
        visual_input = visual_input.to(self.dtype)
        node_queries = node_queries.to(self.dtype)

        # ALTERAÇÃO 1: Recebendo embeddings E pesos do Aligner
        node_embeddings_refined, node_attn_weights, _ = self.aligner(visual_input, node_queries)

        # Formato padronizado do Scene Graph:
        # - nodes: lista com ids contíguos (0..N-1)
        # - edges: {source, target, relation, confidence}
        scene_graph = {"nodes": [], "edges": []}

        # 3. Criação de Nós (filtrados por score)
        kept_original_ids = []
        if self.sg_classifier_head is not None:
            # MIL patch-level: patch_logits [1, 1024, vocab_size]
            patch_logits = self.sg_classifier_head(visual_input)
            probs = torch.sigmoid(patch_logits)[0]                     # [1024, vocab_size]
            H = W = 32                                                  # grid consistente com o pooling
            probs_grid = probs.reshape(H, W, -1)                        # [32, 32, vocab_size]
            visual_grid = visual_input[0].reshape(H, W, -1)             # [32, 32, 768]

            for class_idx, label in enumerate(candidate_nodes):
                if class_idx >= probs_grid.shape[-1]:
                    break  # robustez: candidate_nodes pode exceder vocab_size
                mask = probs_grid[:, :, class_idx] > self.threshold     # [32, 32] bool
                if not mask.any():
                    continue
                components = _label_connected_components(mask)          # [32, 32] int
                n_comps = int(components.max().item())
                for comp_id in range(1, n_comps + 1):
                    comp_mask = (components == comp_id)
                    if int(comp_mask.sum().item()) < self.min_patches:
                        continue
                    # Instance feature: mean-pool dos patches no componente
                    inst_feat_768 = visual_grid[comp_mask].mean(dim=0)             # [768]
                    # Projeta p/ espaco textual via visual_proj do aligner (congelado no eval)
                    inst_emb_4096 = self.aligner.visual_proj(inst_feat_768)        # [4096]
                    inst_score = probs_grid[:, :, class_idx][comp_mask].mean().item()
                    scene_graph["nodes"].append({
                        "id": len(scene_graph["nodes"]),
                        "label": label,
                        "embedding": inst_emb_4096,
                        "score": float(inst_score),
                        "patch_mask": comp_mask.detach().cpu(),
                    })
        else:
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
        rel_queries = rel_queries.to(self.dtype)
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
                    
                    if rel_score > self.edge_threshold:
                        scene_graph["edges"].append({
                            "source": int(node_a["id"]),
                            "relation": rel_label,
                            "target": int(node_b["id"]),
                            "confidence": rel_score.item()
                        })

        return scene_graph
    

class KnowledgeGraphGenerator:
    """
    Gera Knowledge Graph a partir de um Scene Graph.

    Suporta dois modos:
    - **Cache** (taxonomy_cache): lookup instantâneo em dict pré-gerado
      (ver data/extract_candidates.py). Não requer modelo causal.
    - **LLM** (qwen_model + qwen_tokenizer): geração online via modelo causal.

    Relacoes suportadas (VALID_RELATIONS):
    - is_a: taxonomia ("dog | is_a | animal")
    - part_of: meronomia ("wheel | part_of | car")
    - made_of: material ("table | made_of | wood")
    - used_for: funcao ("knife | used_for | cutting")
    - has_property: atributo ("fire | has_property | hot")

    Parameters
    ----------
    qwen_model:
        Modelo causal para geração (opcional se taxonomy_cache fornecido).
    qwen_tokenizer:
        Tokenizer do modelo causal (opcional se taxonomy_cache fornecido).
    taxonomy_cache:
        Dict label → lista de {sub, rel, obj} pré-extraído.
    """

    # Relacoes permitidas no knowledge graph
    VALID_RELATIONS: list[str] = [
        "is_a",         # taxonomia: "dog | is_a | animal"
        "part_of",      # meronomia: "wheel | part_of | car"
        "made_of",      # material: "table | made_of | wood"
        "used_for",     # funcao: "knife | used_for | cutting"
        "has_property", # atributo: "fire | has_property | hot"
    ]

    # Mapa para fuzzy match: variantes comuns -> relacao canonica
    _RELATION_ALIASES: dict[str, str] = {
        "is a": "is_a",
        "isa": "is_a",
        "is_a": "is_a",
        "part of": "part_of",
        "partof": "part_of",
        "part_of": "part_of",
        "made of": "made_of",
        "madeof": "made_of",
        "made_of": "made_of",
        "made from": "made_of",
        "used for": "used_for",
        "usedfor": "used_for",
        "used_for": "used_for",
        "used to": "used_for",
        "has property": "has_property",
        "hasproperty": "has_property",
        "has_property": "has_property",
        "has attribute": "has_property",
        "has_attribute": "has_property",
    }

    def __init__(self, qwen_model=None, qwen_tokenizer=None, taxonomy_cache: dict | None = None):
        self.model = qwen_model
        self.tokenizer = qwen_tokenizer
        self.taxonomy_cache = taxonomy_cache or {}
        if qwen_model is not None:
            self.device = next(qwen_model.parameters()).device

    def _normalize_relation(self, raw_rel: str) -> str | None:
        """
        Normaliza uma relacao extraida do LLM para a forma canonica.

        Faz fuzzy match via alias table. Retorna None se a relacao
        nao puder ser mapeada para nenhuma relacao valida.

        Parameters
        ----------
        raw_rel : str
            Relacao bruta extraida do output do LLM (ex: "is a", "part_of").

        Returns
        -------
        str | None
            Relacao canonica (ex: "is_a") ou None se invalida.
        """
        cleaned = raw_rel.strip().lower()
        # Lookup direto no alias table
        if cleaned in self._RELATION_ALIASES:
            return self._RELATION_ALIASES[cleaned]
        # Tenta remover espacos/underscores extras
        no_spaces = cleaned.replace(" ", "").replace("_", "")
        for alias, canonical in self._RELATION_ALIASES.items():
            if alias.replace(" ", "").replace("_", "") == no_spaces:
                return canonical
        return None

    @torch.no_grad()
    def generate_from_scene(self, scene_graph: dict) -> dict:
        """
        Gera knowledge graph a partir de um scene graph detectado.

        Parameters
        ----------
        scene_graph : dict
            Dict com chave 'nodes' (lista de dicts com 'label').

        Returns
        -------
        dict
            Knowledge graph com chaves:
            - 'entities': list[str] — entidades unicas
            - 'factual_edges': list[dict] — cada dict tem {sub: str, rel: str, obj: str}
              onde rel pertence a VALID_RELATIONS
        """
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

            # Tenta cache primeiro, senão usa LLM
            if label in self.taxonomy_cache:
                for edge in self.taxonomy_cache[label]:
                    knowledge_graph["factual_edges"].append(edge)
                    knowledge_graph["entities"].add(edge["obj"])
            elif self.model is not None:
                self._generate_facts_llm(label, knowledge_graph)
            # Se não tem cache nem modelo, pula (sem fatos para este label)

        knowledge_graph["entities"] = list(knowledge_graph["entities"])
        return knowledge_graph

    @torch.no_grad()
    def _generate_facts_llm(self, label: str, knowledge_graph: dict) -> None:
        """
        Gera fatos multi-relacao via modelo causal (fallback quando não há cache).

        Pede ao LLM fatos usando as relacoes definidas em VALID_RELATIONS.
        Faz parsing e validacao de cada relacao extraida.

        Parameters
        ----------
        label : str
            Label da entidade detectada no scene graph.
        knowledge_graph : dict
            Knowledge graph sendo construido (modificado in-place).
            Chaves: 'entities' (set[str]), 'factual_edges' (list[dict]).
        """
        relations_str = ", ".join(self.VALID_RELATIONS)
        examples = (
            "dog | is_a | animal\n"
            "wheel | part_of | car\n"
            "table | made_of | wood\n"
            "knife | used_for | cutting\n"
            "fire | has_property | hot"
        )

        prompt = (
            f"<|im_start|>system\n"
            f"List exactly 5 universal, factual statements about '{label}'. "
            f"Avoid opinions, abilities, or cultural associations. "
            f"Use ONLY these relations: {relations_str}.\n"
            f"Examples:\n{examples}"
            f"<|im_end|>\n"
            f"<|im_start|>user\n"
            f"Format strictly as: Subject | relation | Object\n"
            f"Always use exactly the word '{label}' as the Subject. "
            f"Use one line per fact. Pick the most informative relation for each fact."
            f"<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
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

            _sub, raw_rel, obj = parts

            # Valida e normaliza a relacao via fuzzy match
            rel = self._normalize_relation(raw_rel)
            if rel is None:
                continue  # Relacao desconhecida, descarta o fato

            knowledge_graph["factual_edges"].append({
                "sub": label,
                "rel": rel,
                "obj": obj
            })

            knowledge_graph["entities"].add(obj)
