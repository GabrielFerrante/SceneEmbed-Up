import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.aligners.lora_cross_attention import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from data.data_utils_pytorch import create_all_dataloaders
import os
from utils.metrics import salvar_recall_results


class RetrievalEvaluator:
    """
    Avaliador de Recall@K usando similaridade patch-texto com max pooling
    sobre o LoRACrossAttentionAligner. Nao depende do SceneGraphGenerator.
    """

    def __init__(self, aligner, dino_encoder, qwen_embedder, device: str = "cuda"):
        self.aligner = aligner
        self.encoder = dino_encoder
        self.embedder = qwen_embedder
        self.device = device
        self.dtype = qwen_embedder.dtype

    @torch.no_grad()
    def evaluate_projection(self, dataloader, k_values=[1, 5, 10]):
        """
        Calcula Recall@K usando Similaridade Patch-Texto com Max Pooling.
        """
        self.aligner.eval()

        all_image_patches = []  # Lista de tensores [N_patches, Dim]
        all_text_queries = []   # Lista de tensores [Dim]

        print("1. Extraindo Patches e Embeddings de Texto...")
        for images, texts in tqdm(dataloader):
            images = images.to(self.device)

            # 1.1 - Extrair patches (DinoV3)
            # hr_feat: [B, C, H, W] -> visual_input: [B, N_patches, 768]
            _, hr_feat = self.encoder.extract_features(images)
            B, C, H, W = hr_feat.shape
            visual_input = hr_feat.view(B, C, -1).transpose(1, 2).to(self.dtype)

            # Embeddings de Texto: [B, 4096]
            formatted_texts = [[t] for t in texts]
            t_queries = self.embedder.embed_components(formatted_texts)
            t_queries = F.normalize(t_queries.squeeze(1), dim=-1)

            # 1.2 - Para cada imagem no batch, guardamos seus patches crus
            for i in range(B):
                patches_i = visual_input[i]  # [N_patches, 768]
                all_image_patches.append(patches_i.cpu())
                all_text_queries.append(t_queries[i].cpu())

        num_samples = len(all_text_queries)
        text_embeddings = torch.stack(all_text_queries)  # [N_total, Dim]

        # 2. Rankear Imagens via Max Pooling de Similaridade
        print(f"2. Calculando Matriz de Similaridade [Max Pooling] para {num_samples} amostras...")
        sim_matrix = torch.zeros((num_samples, num_samples))

        for i in range(num_samples):
            patches_i = all_image_patches[i].to(self.device)  # [P, D]

            # Expande patches para todos os textos: [N, P, D]
            patches_batch = patches_i.unsqueeze(0).expand(num_samples, -1, -1)
            # Textos em batch: [N, 1, D]
            text_batch = text_embeddings.unsqueeze(1)

            # Um unico forward
            p_refined, _ = self.aligner(
                patches_batch,  # [N, P, D]
                text_batch      # [N, 1, D]
            )

            p_refined = F.normalize(p_refined, dim=-1)  # [N, P, D]

            # Similaridade patch-text: [N, P]
            sim = torch.matmul(
                p_refined,
                text_embeddings.unsqueeze(-1)  # [N, D, 1]
            ).squeeze(-1)

            # max over patches: [N]
            scores = sim.max(dim=1).values
            sim_matrix[i] = scores

        # 3. Calcular Recall@K
        results = {}
        labels = torch.arange(num_samples)
        for k in k_values:
            if k > num_samples:
                continue
            # Top-k ao longo da dimensao de imagens
            _, top_k = torch.topk(sim_matrix, k=k, dim=0)  # [k, N_total]
            correct = torch.any(top_k == labels.unsqueeze(0), dim=0)
            results[f"Recall@{k}"] = correct.float().mean().item()

        return results


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dino = DinoSceneEncoder(device=device)
    qwen = QwenSceneEmbedder(device=device)

    aligner = LoRACrossAttentionAligner(
        visual_dim=768,
        text_dim=4096
    )

    weights_path = "checkpoints/aligner_epoch_10.pth"
    if os.path.exists(weights_path):
        aligner.load_state_dict(
            torch.load(weights_path, map_location=device),
            strict=False
        )
        print(f"Weights loaded from {weights_path}")
    else:
        print("Checkpoint not found. Running with random weights.")

    aligner.to(device).eval()

    evaluator = RetrievalEvaluator(
        aligner=aligner,
        dino_encoder=dino,
        qwen_embedder=qwen,
        device=device,
    )

    test_dataloader = create_all_dataloaders(
        "F:/COYO/coyo/extracted",
        batch_size=2,
        num_workers=4,
        t="test"
    )

    print("\n--- Evaluating Alignment (Recall@K) ---")
    recall_results = evaluator.evaluate_projection(
        test_dataloader,
        k_values=[1, 5, 10]
    )

    for k, v in recall_results.items():
        print(f"{k}: {v:.4f}")

    salvar_recall_results(recall_results)
