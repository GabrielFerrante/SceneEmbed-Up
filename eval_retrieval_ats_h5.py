"""
eval_retrieval_ats_h5.py
-------------------------
Avaliacao de retrieval (Recall@K bidirecional, I2T e T2I) do
LoRACrossAttentionAligner treinado com ATS (AdaptiveTokenSampler), sobre
shards HDF5 anyup (1024 patches, 32x32).

Diferente de eval_retrieval_with_h5.py, o checkpoint aqui contem sampler +
aligner (nn.ModuleDict salvo por train_aligner_ats_h5.py). Para cada batch:
  1. ATS seleciona K=196 patches relevantes dos 1024 patches AnyUp
     (text-guided + content-based scoring)
  2. Aligner roda cross-attention sobre os K patches selecionados
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset_withSSD
from eval_retrieval_with_h5 import (
    _build_sim_matrix,
    calcular_recall_bidirecional,
    diagnosticar_embeddings,
)
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from train_aligner_ats_h5 import AdaptiveTokenSampler
from utils.metrics import salvar_recall_results


# ---------------------------------------------------------------------------
# Avaliador ATS + Aligner
# ---------------------------------------------------------------------------

class ATSRetrievalEvaluator:
    """Avaliador de retrieval (Recall@K) com seleção adaptativa de tokens (ATS)."""

    def __init__(
        self,
        sampler: AdaptiveTokenSampler,
        aligner: LoRACrossAttentionAligner,
        device: str = "cuda",
    ) -> None:
        self.sampler = sampler
        self.aligner = aligner
        self.device = device
        self.dtype = torch.bfloat16

    @torch.no_grad()
    def evaluate_projection(
        self,
        dataloader: DataLoader,
        k_values: List[int] = [1, 5, 10],
        chunk_size: int = 512,
        run_diagnostics: bool = True,
    ) -> Dict[str, float]:
        """
        Avalia o par (sampler, aligner) com Recall@K bidirecional (I2T e T2I).

        Shapes internos
        ---------------
        visual_input:  `[B, 1024, 768]`  — patches AnyUp (32x32)
        text_queries:  `[B, 1, 4096]`
        selected:      `[B, 196, 768]`   — saída do AdaptiveTokenSampler
        attn_output:   `[B, 1, 4096]`    — saída do aligner (dim do texto)
        v_global:      `[B, 4096]`       — embedding visual refinado, pós-squeeze
        V:             `[N, 4096]`       — todos os visuais, CPU
        T:             `[N, 4096]`       — todos os textos, CPU
        sim_matrix:    `[N, N]`          — similaridades simétricas
        """
        self.sampler.eval()
        self.aligner.eval()

        if run_diagnostics:
            diagnosticar_embeddings(dataloader, n_batches=5)

        all_v_global: List[torch.Tensor] = []
        all_t_norm: List[torch.Tensor] = []

        print("1. Extracting visual (ATS-selected) and text representations...")
        for visual_input, text_queries in tqdm(dataloader, desc="Embedding extraction"):
            visual_input = visual_input.to(self.device).to(self.dtype)  # [B, 1024, 768]
            text_queries = text_queries.to(self.device).to(self.dtype)  # [B, 1, 4096]

            selected, _, _ = self.sampler(visual_input, text_queries)    # [B, 196, 768]
            attn_output, _, _ = self.aligner(selected, text_queries)
            v_global = attn_output.squeeze(1)  # [B, 4096]

            # Normalização L2 antes de armazenar — garante produto escalar = cos sim
            v_norm = F.normalize(v_global, p=2, dim=-1).bfloat16().cpu()
            t_norm = F.normalize(text_queries.squeeze(1), p=2, dim=-1).bfloat16().cpu()

            all_v_global.append(v_norm)
            all_t_norm.append(t_norm)

        V = torch.cat(all_v_global, dim=0)  # [N, 4096] — CPU
        T = torch.cat(all_t_norm,   dim=0)  # [N, 4096] — CPU

        print(f"V shape: {V.shape}, avg norm: {V.norm(dim=-1).mean():.4f}")
        print(f"T shape: {T.shape}, avg norm: {T.norm(dim=-1).mean():.4f}")
        print(f"Similaridade V[0] vs T[0] (par correto): {(V[0] * T[0]).sum():.4f}")
        print(f"Similaridade V[0] vs T[1] (par errado):  {(V[0] * T[1]).sum():.4f}")

        assert V.shape[1] == T.shape[1], (
            f"Dimensões incompatíveis: V={V.shape}, T={T.shape}."
        )

        N = V.shape[0]
        print(f"2. Building symmetric similarity matrix [{N}×{N}]...")
        sim_matrix = _build_sim_matrix(V, T, self.device, chunk_size)

        print("3. Calculando Recall@K bidirecional...")
        return calcular_recall_bidirecional(sim_matrix, k_values)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    K_PATCHES = 196

    sampler = AdaptiveTokenSampler(visual_dim=768, text_dim=4096, k=K_PATCHES)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)
    combined = nn.ModuleDict({"sampler": sampler, "aligner": aligner})

    weights_path = "checkpoints/best_ats_aligner.pth"
    if os.path.exists(weights_path):
        combined.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"Pesos carregados de {weights_path}")
    else:
        print("Checkpoint not found. Running with random weights.")

    combined = combined.to(device).to(torch.bfloat16).eval()

    test_h5_ds = ShardedH5Dataset_withSSD("G:/coyo/embeds/test_anyup")  # 1024 patches (AnyUp)
    test_h5_loader = DataLoader(
        test_h5_ds, batch_size=128, shuffle=False,
        pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True,
    )

    batch_h5 = next(iter(test_h5_loader))
    print(f"  h5 loader — visual: {batch_h5[0].shape}, texto: {batch_h5[1].shape}")

    evaluator = ATSRetrievalEvaluator(sampler=sampler, aligner=aligner, device=device)

    print("\n--- Avaliando Alinhamento ATS (Recall@K Bidirecional) ---")
    recall_results: Dict[str, float] = evaluator.evaluate_projection(
        test_h5_loader,
        k_values=[1, 5, 10],
        chunk_size=512,
        run_diagnostics=True,
    )

    print("\nResultados:")
    for k in [1, 5, 10]:
        i2t = recall_results.get(f"I2T_Recall@{k}")
        t2i = recall_results.get(f"T2I_Recall@{k}")
        mean = recall_results.get(f"Mean_Recall@{k}")
        if i2t is not None:
            print(f"  @{k:2d}  I2T={i2t:.4f}  T2I={t2i:.4f}  Mean={mean:.4f}")

    salvar_recall_results(recall_results, path="results/recall_metrics_ats.json")
