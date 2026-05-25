"""
eval_with_h5.py
---------------
Avaliacao de retrieval (Recall@K bidirecional) do LoRACrossAttentionAligner
sobre shards HDF5 pre-computados.
"""

from __future__ import annotations

import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.data_utils_pytorch import ShardedH5Dataset_withSSD
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from torch.utils.data import DataLoader
from utils.metrics import salvar_recall_results


# ---------------------------------------------------------------------------
# Diagnóstico de embeddings H5
# ---------------------------------------------------------------------------

def diagnosticar_embeddings(
    dataloader: DataLoader,
    n_batches: int = 10,
) -> None:
    """
    Verifica qualidade básica dos embeddings carregados do H5.

    Imprime norma média, min/max e dtype para os primeiros `n_batches`
    batches. Útil para detectar embeddings zerados, NaN ou mal normalizados
    antes de iniciar uma avaliação longa.

    Parameters
    ----------
    dataloader:
        DataLoader do ShardedH5Dataset.
    n_batches:
        Número de batches a inspecionar.
    """
    print("\n--- H5 Embeddings Diagnostic ---")
    for i, (visual_input, text_queries) in enumerate(dataloader):
        if i >= n_batches:
            break

        v_norm = visual_input.float().norm(dim=-1)   # [B, N_patches]
        t_norm = text_queries.float().norm(dim=-1)   # [B, 1] ou [B, N]

        print(
            f"  Batch {i:02d} | "
            f"visual shape={tuple(visual_input.shape)} dtype={visual_input.dtype} "
            f"norm_mean={v_norm.mean():.3f} | "
            f"text shape={tuple(text_queries.shape)} dtype={text_queries.dtype} "
            f"norm_mean={t_norm.mean():.3f}"
        )

        if torch.isnan(visual_input).any() or torch.isnan(text_queries).any():
            print("  [WARN] NaN detected in embeddings!")

    print("--- End of Diagnostic ---\n")


# ---------------------------------------------------------------------------
# Construção da matriz de similaridade simétrica
# ---------------------------------------------------------------------------

def _build_sim_matrix(
    V: torch.Tensor,
    T: torch.Tensor,
    device: str,
    chunk_size: int = 512,
) -> torch.Tensor:
    """
    Calcula matriz de similaridade V @ T^T (linhas=imagens, colunas=textos).

    Usa produto escalar direto após normalização L2. A matriz NÃO é
    simetrizada, permitindo que I2T e T2I sejam avaliados independentemente.

    Parameters
    ----------
    V:
        Tensor `[N, D]` de embeddings visuais normalizados (CPU).
    T:
        Tensor `[N, D]` de embeddings textuais normalizados (CPU).
    device:
        Dispositivo de cálculo ('cuda' ou 'cpu').
    chunk_size:
        Tamanho do chunk para processar em blocos e controlar uso de VRAM.

    Returns
    -------
    torch.Tensor
        Matriz `[N, N]` onde `[i, j] = sim(visual_i, texto_j)`, CPU, float32.
    """
    N = V.shape[0]
    sim_matrix = torch.zeros(N, N, dtype=torch.float32)

    T_gpu = T.to(device)  # [N, D] — fixo na GPU durante o loop

    for start in tqdm(range(0, N, chunk_size), desc="Similarity chunks", leave=False):
        end = min(start + chunk_size, N)
        v_chunk = V[start:end].to(device)           # [C, D]
        sim_chunk = torch.matmul(v_chunk, T_gpu.T)  # [C, N]
        sim_matrix[start:end] = sim_chunk.float().cpu()

        del v_chunk, sim_chunk
        torch.cuda.empty_cache()

    del T_gpu
    torch.cuda.empty_cache()

    return sim_matrix


# ---------------------------------------------------------------------------
# Cálculo de Recall@K bidirecional
# ---------------------------------------------------------------------------

def calcular_recall_bidirecional(
    sim_matrix: torch.Tensor,
    k_values: List[int],
) -> Dict[str, float]:
    """
    Calcula Recall@K nas direções Image→Text e Text→Image.

    Segue o protocolo padrão de avaliação de retrieval multimodal: 
    para cada amostra, verifica se o par correto está nos top-K candidatos recuperados.

    Parameters
    ----------
    sim_matrix:
        Tensor `[N, N]` onde sim_matrix[i][j] = similaridade(visual_i, texto_j).
    k_values:
        Lista de valores K a avaliar (ex.: [1, 5, 10]).

    Returns
    -------
    dict
        Métricas no formato::

            {
                "I2T_Recall@1":  0.xx,  # imagem → texto correto no top-1
                "T2I_Recall@1":  0.xx,  # texto  → imagem correta no top-1
                "Mean_Recall@1": 0.xx,  # média das duas direções
                ...
            }
    """
    N = sim_matrix.size(0)
    labels = torch.arange(N)
    results: Dict[str, float] = {}

    for k in k_values:
        if k > N:
            continue

        # Image → Text: linha i = scores da imagem i contra todos os textos
        _, top_i2t = torch.topk(sim_matrix, k=k, dim=1)       # [N, k]
        correct_i2t = torch.any(top_i2t == labels.unsqueeze(1), dim=1)
        results[f"I2T_Recall@{k}"] = correct_i2t.float().mean().item()

        # Text → Image: linha j da transposta = scores do texto j contra todas as imagens
        _, top_t2i = torch.topk(sim_matrix.T, k=k, dim=1)     # [N, k]
        correct_t2i = torch.any(top_t2i == labels.unsqueeze(1), dim=1)
        results[f"T2I_Recall@{k}"] = correct_t2i.float().mean().item()

        # Média (padrão reportado em papers)
        results[f"Mean_Recall@{k}"] = (
            results[f"I2T_Recall@{k}"] + results[f"T2I_Recall@{k}"]
        ) / 2.0

    return results


# ---------------------------------------------------------------------------
# Avaliador principal
# ---------------------------------------------------------------------------

class SceneGraphEvaluator:
    """Avaliador de retrieval (Recall@K) sobre embeddings refinados pelo aligner."""

    def __init__(self, aligner: LoRACrossAttentionAligner, device: str = "cuda") -> None:
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
        Avalia o aligner com Recall@K bidirecional (I2T e T2I).

        Extrai embeddings visuais refinados pelo aligner e embeddings textuais
        normalizados, constrói a matriz de similaridade simétrica e calcula
        Recall@K nas duas direções.

        Parameters
        ----------
        dataloader:
            DataLoader de `ShardedH5Dataset` com pares (visual_feats, text_feats).
        k_values:
            Lista de K a avaliar.
        chunk_size:
            Tamanho do chunk para a matriz de similaridade (controla VRAM).
        run_diagnostics:
            Se True, roda diagnóstico de qualidade dos embeddings antes de avaliar.

        Returns
        -------
        dict
            Métricas bidirecionais no formato descrito em `calcular_recall_bidirecional`.

        Shapes internos
        ---------------
        visual_input:  `[B, N_patches, 768]`
        text_queries:  `[B, 1, 4096]`
        attn_output:   `[B, 1, 4096]`  — saída do aligner (dim do texto)
        v_global:      `[B, 4096]`     — embedding visual refinado, pós-squeeze
        V:             `[N, 4096]`     — todos os visuais, CPU
        T:             `[N, 4096]`     — todos os textos, CPU
        sim_matrix:    `[N, N]`        — similaridades simétricas
        """
        self.aligner.eval()

        if run_diagnostics:
            diagnosticar_embeddings(dataloader, n_batches=5)

        all_v_global: List[torch.Tensor] = []
        all_t_norm: List[torch.Tensor] = []

        print("1. Extracting visual and text representations...")
        for visual_input, text_queries in tqdm(dataloader, desc="Embedding extraction"):
            visual_input = visual_input.to(self.device).to(self.dtype)  # [B, N_patches, 768]
            text_queries = text_queries.to(self.device).to(self.dtype)  # [B, 1, 4096]

            attn_output, _, _ = self.aligner(visual_input, text_queries)
            v_global = attn_output.squeeze(1)  # [B, 4096]

            # Normalização L2 antes de armazenar — garante produto escalar = cos sim
            v_norm = F.normalize(v_global, p=2, dim=-1).bfloat16().cpu()
            t_norm = F.normalize(text_queries.squeeze(1), p=2, dim=-1).bfloat16().cpu()

            all_v_global.append(v_norm)
            all_t_norm.append(t_norm)

        V = torch.cat(all_v_global, dim=0)  # [N, 4096] — CPU
        T = torch.cat(all_t_norm,   dim=0)  # [N, 4096] — CPU
        


        # Diagnóstico — rode isso antes de calcular a matriz
        print(f"V shape: {V.shape}, avg norm: {V.norm(dim=-1).mean():.4f}")
        print(f"T shape: {T.shape}, avg norm: {T.norm(dim=-1).mean():.4f}")
        print(f"Similaridade V[0] vs T[0] (par correto): {(V[0] * T[0]).sum():.4f}")
        print(f"Similaridade V[0] vs T[1] (par errado):  {(V[0] * T[1]).sum():.4f}")
        print(f"V == T: {torch.allclose(V, T, atol=1e-3)}")

        assert V.shape[1] == T.shape[1], (
            f"Dimensões incompatíveis: V={V.shape}, T={T.shape}. "
            "Verifique a dim de saída do visual_proj no aligner."
        )

        N = V.shape[0]
        print(f"2. Building symmetric similarity matrix [{N}×{N}]...")

        # Constrói matriz simétrica e calcula Recall bidirecional
        sim_matrix = _build_sim_matrix(V, T, self.device, chunk_size)

        print("3. Calculando Recall@K bidirecional...")
        results = calcular_recall_bidirecional(sim_matrix, k_values)

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096, rank=64)

    #weights_path = "checkpoints/best_aligner_no_up.pth"
    weights_path = "checkpoints/best_aligner.pth"
    if os.path.exists(weights_path):
        aligner.load_state_dict(
            torch.load(weights_path, map_location=device),
            strict=False,
        )
        print(f"Pesos carregados de {weights_path}")
    else:
        print("Checkpoint not found. Running with random weights.")

    aligner.to(device).to(torch.bfloat16).eval()

    test_h5_ds = ShardedH5Dataset_withSSD("E:/COYO/embeds/test_noup") #WITHOUT ANYUP
    #test_h5_ds = ShardedH5Dataset_withSSD("G:/coyo/embeds/test_anyup") #WITH ANYUP
    test_h5_loader = DataLoader(
        test_h5_ds, batch_size=128, shuffle=False,
        pin_memory=True, num_workers=4, prefetch_factor=4, persistent_workers=True,
    )

    batch_h5 = next(iter(test_h5_loader))
    print(f"  h5 loader — visual: {batch_h5[0].shape}, texto: {batch_h5[1].shape}")

    evaluator = SceneGraphEvaluator(aligner=aligner, device=device)

    print("\n--- Avaliando Alinhamento (Recall@K Bidirecional) ---")
    recall_results: Dict[str, float] = evaluator.evaluate_projection(
        test_h5_loader,
        k_values=[1, 5, 10],
        chunk_size=512,
        run_diagnostics=True,
    )

    print("\nResultados:")
    for k in [1, 5, 10]:
        i2t = recall_results.get(f"I2T_Recall@{k}", None)
        t2i = recall_results.get(f"T2I_Recall@{k}", None)
        mean = recall_results.get(f"Mean_Recall@{k}", None)
        if i2t is not None:
            print(f"  @{k:2d}  I2T={i2t:.4f}  T2I={t2i:.4f}  Mean={mean:.4f}")

    salvar_recall_results(recall_results)