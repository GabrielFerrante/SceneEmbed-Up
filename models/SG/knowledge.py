"""
knowledge.py
------------
Geracao de Knowledge Graph a partir de Scene Graphs detectados.

Suporta dois modos:
  - Cache (taxonomy_cache): lookup instantaneo em dict pre-gerado.
  - LLM (qwen_model + qwen_tokenizer): geracao online via modelo causal.
"""

from __future__ import annotations

import torch


class KnowledgeGraphGenerator:
    """
    Gera Knowledge Graph a partir de um Scene Graph.

    Suporta dois modos:
    - **Cache** (taxonomy_cache): lookup instantaneo em dict pre-gerado
      (ver data/extract_candidates.py). Nao requer modelo causal.
    - **LLM** (qwen_model + qwen_tokenizer): geracao online via modelo causal.

    Relacoes suportadas (VALID_RELATIONS):
    - is_a: taxonomia ("dog | is_a | animal")
    - part_of: meronomia ("wheel | part_of | car")
    - made_of: material ("table | made_of | wood")
    - used_for: funcao ("knife | used_for | cutting")
    - has_property: atributo ("fire | has_property | hot")

    Parameters
    ----------
    qwen_model:
        Modelo causal para geracao (opcional se taxonomy_cache fornecido).
    qwen_tokenizer:
        Tokenizer do modelo causal (opcional se taxonomy_cache fornecido).
    taxonomy_cache:
        Dict label -> lista de {sub, rel, obj} pre-extraido.
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
        if cleaned in self._RELATION_ALIASES:
            return self._RELATION_ALIASES[cleaned]
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

            if label in self.taxonomy_cache:
                for edge in self.taxonomy_cache[label]:
                    knowledge_graph["factual_edges"].append(edge)
                    knowledge_graph["entities"].add(edge["obj"])
            elif self.model is not None:
                self._generate_facts_llm(label, knowledge_graph)

        knowledge_graph["entities"] = list(knowledge_graph["entities"])
        return knowledge_graph

    @torch.no_grad()
    def _generate_facts_llm(self, label: str, knowledge_graph: dict) -> None:
        """
        Gera fatos multi-relacao via modelo causal (fallback quando nao ha cache).

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

            rel = self._normalize_relation(raw_rel)
            if rel is None:
                continue

            knowledge_graph["factual_edges"].append({
                "sub": label,
                "rel": rel,
                "obj": obj
            })

            knowledge_graph["entities"].add(obj)
