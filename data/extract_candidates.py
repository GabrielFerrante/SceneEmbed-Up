"""
extract_candidates.py
---------------------
Gera o taxonomy_cache.json consumido pelo KnowledgeGraphGenerator (modo cache).

Detecção de objetos na imagem agora é responsabilidade da SGClassifierHead
(treinada em train_sg_head.py sobre VG-150). Esse script cuida apenas da
outra peça do pipeline de KG: dada uma lista de labels (o vocab VG-150
persistido em checkpoints/sg_head_vocab.json), chama Qwen3-0.6B para
produzir fatos multi-relacionais do tipo:
  dog | is_a | animal
  wheel | part_of | car

Uso:
    python data/extract_candidates.py
    python data/extract_candidates.py --vocab checkpoints/sg_head_vocab.json \\
        --output checkpoints/taxonomy_cache.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Adiciona raiz do projeto ao path para importar models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


LLM_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_VOCAB_PATH = os.path.join("checkpoints", "sg_head_vocab.json")
DEFAULT_OUTPUT_PATH = os.path.join("checkpoints", "taxonomy_cache.json")

# Relações suportadas no Knowledge Graph.
# Deve corresponder a KnowledgeGraphGenerator.VALID_RELATIONS em models/SG/generation.py.
KG_RELATIONS = [
    "is_a",
    "part_of",
    "made_of",
    "used_for",
    "has_property",
]


@torch.no_grad()
def extract_taxonomy(
    labels: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    device: str,
) -> dict[str, list[dict[str, str]]]:
    """Gera fatos KG multi-relacionais para cada label via Qwen3-0.6B."""
    allowed_rels = set(KG_RELATIONS)
    taxonomy: dict[str, list[dict[str, str]]] = {}

    system_prompt = (
        "You output ONLY structured facts in the exact format: "
        "Subject | relation | Object. No explanations. "
        f"Use ONLY these relations: {', '.join(KG_RELATIONS)}."
    )

    for label in tqdm(labels, desc="Extraindo KG facts"):
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Give up to 5 diverse facts about '{label}'.\n"
                    f"Use exactly '{label}' as the Subject.\n"
                    f"Use different relations when possible. No repetitions.\n"
                    f"Examples (for unrelated subjects):\n"
                    f"dog | is_a | animal\n"
                    f"wheel | part_of | car\n"
                    f"table | made_of | wood\n"
                    f"knife | used_for | cutting\n"
                    f"fire | has_property | hot\n"
                    f"Now do it for '{label}':"
                ),
            },
        ]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        edges: list[dict[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for line in response.split("\n"):
            if "|" not in line:
                continue
            parts = [p.strip().lower() for p in line.split("|")]
            if len(parts) != 3:
                continue
            _, rel, obj = parts

            rel = rel.replace(" ", "_")

            if rel not in allowed_rels:
                continue
            if not obj or len(obj.split()) > 3:
                continue

            key = (rel, obj)
            if key in seen:
                continue
            seen.add(key)

            edges.append({"sub": label, "rel": rel, "obj": obj})

        taxonomy[label] = edges

    return taxonomy


def load_vocab(vocab_path: str) -> list[str]:
    """Lê vocab salvo por train_sg_head.py: {'obj_list': [...]}."""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(
            f"Vocab nao encontrado em {vocab_path}. "
            "Rode primeiro: python train_sg_head.py --vg-dir <dir>"
        )
    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    obj_list = data.get("obj_list")
    if not isinstance(obj_list, list) or not obj_list:
        raise ValueError(f"{vocab_path} nao contem chave 'obj_list' valida")
    return obj_list


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera taxonomy_cache.json para KnowledgeGraphGenerator"
    )
    parser.add_argument("--vocab", type=str, default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--model", type=str, default=LLM_MODEL_ID)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}")
    if device == "cuda":
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        free = (total * 1e9 - torch.cuda.memory_allocated()) / 1e9
        print(f"VRAM: {free:.1f}/{total:.1f} GB livre")

    labels = load_vocab(args.vocab)
    print(f"Labels loaded: {len(labels)} ({args.vocab})")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    llm = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(device).eval()

    try:
        taxonomy = extract_taxonomy(labels, llm, tokenizer, device)
    except RuntimeError as e:
        if "CUDA" in str(e).upper() and torch.cuda.is_available():
            print(torch.cuda.memory_summary())
        raise

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)
    print(f"Taxonomy salva em: {args.output}  ({len(taxonomy)} labels)")


if __name__ == "__main__":
    main()
