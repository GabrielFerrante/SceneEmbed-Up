"""
extract_candidates.py
---------------------
Extrai candidatos (objetos concretos) das legendas do dataset de teste
e taxonomias (is_a) para cada candidato, usando um modelo causal leve
(Qwen3-0.6B).

Gera dois JSONs:
  1. candidates_test.json  — índice → lista de candidatos
  2. taxonomy_cache.json   — label → lista de {sub, rel, obj}

Consumidos pelo eval_with_h5.py sem precisar de modelo causal na avaliação.

Uso:
    python data/extract_candidates.py
"""

from __future__ import annotations

import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils_pytorch import CoyoExtractedDataset, get_transforms
from torch.utils.data import random_split


# ── Configuração ─────────────────────────────────────────────────────────
COYO_ROOT = "F:/COYO/coyo/extracted"
CANDIDATES_OUTPUT = "F:/COYO/candidates_test.json"
TAXONOMY_OUTPUT = "F:/COYO/taxonomy_cache.json"
MODEL_ID = "Qwen/Qwen3-0.6B"
BATCH_SIZE = 32          # legendas por batch (CPU-bound no tokenizer)
MAX_NEW_TOKENS = 30
MAX_SAMPLES = 50         # limite de amostras a processar
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32


def build_test_indices(root_dir: str) -> list[int]:
    """
    Reproduz o split do create_all_dataloaders (seed=42) e retorna
    os índices globais do subset de teste.
    """
    transform = get_transforms(256)
    full_ds = CoyoExtractedDataset(root_dir=root_dir, transform=transform)
    total = len(full_ds)

    train_size = 500_000
    val_size = 10_000
    test_size = 10_000
    unused = total - (train_size + val_size + test_size)

    gen = torch.Generator().manual_seed(42)
    _, _, test_subset, _ = random_split(
        full_ds, [train_size, val_size, test_size, unused], generator=gen
    )
    return full_ds, test_subset


@torch.no_grad()
def extract_candidates_batch(
    captions: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> list[list[str]]:
    """
    Extrai candidatos de um batch de legendas usando chat template do Qwen3
    com `enable_thinking=False` para evitar vazamento de raciocínio CoT.
    """
    messages_batch = [
        [
            {
                "role": "system",
                "content": (
                    "You extract concrete physical objects from image captions. "
                    "Output ONLY a comma-separated list of single-word nouns. "
                    "No explanations, no reasoning, no extra text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Caption: {cap}\n"
                    "Example input: A black cat sitting on a wooden table\n"
                    "Example output: cat, table"
                ),
            },
        ]
        for cap in captions
    ]

    prompts = [
        tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # desativa CoT do Qwen3
        )
        for msgs in messages_batch
    ]

    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    input_len = inputs.input_ids.shape[1]
    results = []
    for seq in outputs:
        text = tokenizer.decode(seq[input_len:], skip_special_tokens=True)
        # Pega apenas a primeira linha (descarta possível continuação)
        text = text.strip().split("\n")[0]
        candidatos = [
            p.strip().lower()
            for p in text.split(",")
            if 1 < len(p.strip()) <= 30 and " " not in p.strip()
        ]
        results.append(candidatos)

    return results


@torch.no_grad()
def extract_taxonomy(
    labels: list[str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> dict[str, list[dict[str, str]]]:
    """
    Gera fatos taxonômicos (is_a) para cada label usando chat template do
    Qwen3 com `enable_thinking=False`.
    """
    taxonomy: dict[str, list[dict[str, str]]] = {}

    for label in tqdm(labels, desc="Extraindo taxonomias"):
        messages = [
            {
                "role": "system",
                "content": (
                    "You output ONLY structured taxonomy facts in the exact "
                    "format: Subject | is_a | Object. No explanations."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Give exactly 2 taxonomy facts about '{label}'.\n"
                    f"Use exactly '{label}' as the Subject.\n"
                    f"Example for 'cat':\n"
                    f"cat | is_a | mammal\n"
                    f"cat | is_a | animal\n"
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

        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True,
        )

        edges = []
        for line in response.split("\n"):
            if "|" not in line:
                continue
            parts = [p.strip().lower() for p in line.split("|")]
            if len(parts) != 3:
                continue
            sub, rel, obj = parts
            # Sanitização: descarta obj com mais de 3 palavras ou vazio
            if not obj or len(obj.split()) > 3:
                continue
            edges.append({"sub": label, "rel": "is_a", "obj": obj})

        taxonomy[label] = edges

    return taxonomy


def main() -> None:
    print(f"Carregando modelo causal {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE
    ).to(DEVICE).eval()

    print(f"Modelo carregado em {DEVICE} ({DTYPE})")
    if DEVICE == "cuda":
        vram = torch.cuda.memory_allocated() / 1e9
        print(f"VRAM utilizada: {vram:.2f} GB")

    # ── Etapa 1: Candidatos ──────────────────────────────────────────────
    print("Construindo split de teste (seed=42)...")
    full_ds, test_subset = build_test_indices(COYO_ROOT)

    candidates_map: dict[int, list[str]] = {}
    failed = 0

    indices = list(range(min(len(test_subset), MAX_SAMPLES)))
    print(f"Processando {len(indices)} amostras (MAX_SAMPLES={MAX_SAMPLES})")

    for start in tqdm(range(0, len(indices), BATCH_SIZE), desc="Extraindo candidatos"):
        end = min(start + BATCH_SIZE, len(indices))
        batch_indices = indices[start:end]

        captions = []
        valid_indices = []
        for idx in batch_indices:
            sample = test_subset[idx]
            if sample is None:
                failed += 1
                continue
            _, caption = sample
            if not caption:
                failed += 1
                continue
            captions.append(caption)
            valid_indices.append(idx)

        if not captions:
            continue

        batch_candidates = extract_candidates_batch(captions, model, tokenizer)

        for idx, cands in zip(valid_indices, batch_candidates):
            all_cands = list(set(cands + ["person", "vehicle", "object"]))
            candidates_map[idx] = all_cands

    os.makedirs(os.path.dirname(CANDIDATES_OUTPUT) or ".", exist_ok=True)
    with open(CANDIDATES_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(candidates_map, f, ensure_ascii=False, indent=2)

    print(f"\nCandidatos extraídos: {len(candidates_map)} amostras ({failed} falhas)")
    print(f"  Salvo em: {CANDIDATES_OUTPUT}")

    # ── Etapa 2: Taxonomias ──────────────────────────────────────────────
    # Coleta todos os labels únicos de todos os candidatos
    all_labels = set()
    for cands in candidates_map.values():
        all_labels.update(cands)
    all_labels = sorted(all_labels)

    print(f"\nExtraindo taxonomias para {len(all_labels)} labels únicos...")
    taxonomy = extract_taxonomy(all_labels, model, tokenizer)

    with open(TAXONOMY_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(taxonomy, f, ensure_ascii=False, indent=2)

    print(f"  Salvo em: {TAXONOMY_OUTPUT}")
    print("\nPré-extração finalizada!")


if __name__ == "__main__":
    main()
