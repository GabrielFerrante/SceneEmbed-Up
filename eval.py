import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from models.SG.projection import LoRACrossAttentionAligner, calculate_retrieval_score
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.generation import SceneGraphGenerator, KnowledgeGraphGenerator, salvar_grafos_json
from data.data_utils_pytorch import create_all_dataloaders
import os
import json
from datetime import datetime

class SceneGraphEvaluator:
    def __init__(self, generator, device="cuda"):
        self.generator = generator
        self.device = device
        self.dtype = generator.dtype

    @torch.no_grad()
    def evaluate_projection(self, dataloader, k_values=[1, 5, 10]):
        """
        Calcula Recall@K para o alinhamento Imagem-Texto.
        Indica se o Aligner coloca os pares corretos próximos no espaço latente.
        """
        self.generator.aligner.eval()
        img_embeddings = []
        text_embeddings = []

        print("Extraindo Embeddings para Recall@K")
        for images, texts in tqdm(dataloader):
            # Features Visuais
            _, hr_feat = self.generator.encoder.extract_features(images)
            visual_input = hr_feat.view(hr_feat.size(0), hr_feat.size(1), -1).transpose(1, 2).to(self.dtype)
            
            # Embeddings de Texto (Targets)
            formatted_texts = [[t] for t in texts]
            t_queries = self.generator.embedder.embed_components(formatted_texts)
            
            # Projeção (Alinhamento)
            # Usamos o primeiro token de cada query como representação do par
            v_refined = self.generator.aligner(visual_input, t_queries)
            
            img_embeddings.append(F.normalize(v_refined.squeeze(1), dim=-1).cpu())
            text_embeddings.append(F.normalize(t_queries.squeeze(1), dim=-1).cpu())

        # Matriz de Similaridade Global [N_total, N_total]
        img_embeddings = torch.cat(img_embeddings, dim=0)
        text_embeddings = torch.cat(text_embeddings, dim=0)
        sim_matrix = torch.matmul(img_embeddings, text_embeddings.T)

        results = {}
        for k in k_values:
            # Verifica se a diagonal (índice i == j) está no top-k de cada linha
            top_k = torch.topk(sim_matrix, k=k, dim=1).indices
            correct = torch.any(top_k == torch.arange(len(sim_matrix)).unsqueeze(1), dim=1)
            results[f"Recall@{k}"] = correct.float().mean().item()
            
        return results
    
    def salvar_recall_results(recall_results, filename="recall_metrics.json", directory="results"):
        """
        Salva os resultados de Recall@K em um arquivo JSON.
        """
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        path = os.path.join(directory, filename)
        
        # Prepara a estrutura com metadados
        data_to_save = {
            "timestamp": datetime.now().isoformat(),
            "experiment_info": {
                "model": "LoRA-Aligner-v1",
                "visual_encoder": "DinoV3",
                "text_encoder": "Qwen-7B-Embedder"
            },
            "metrics": recall_results  # Aqui entra o dicionário {Recall@1: x, Recall@5: y, ...}
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        
        print(f" Métricas de Recall salvas com sucesso em: {path}")


    def evaluate_compare_graphs(self, scene_g, kg_g ):
        
        """
        Verifica se as labels físicas existem na base de conhecimento.
        """
        scene_labels = {n['label'] for n in scene_g['nodes']}
        knowledge_labels = set(kg_g['entities'])
        
        # Quão bem o conhecimento cobre a cena?
        coverage = len(scene_labels & knowledge_labels) / len(scene_labels) if scene_labels else 0
        return {"semantic_coverage": coverage}
    
    

def extrair_candidatos_llm(label_real, embedder):
    """
    Usa o Qwen para extrair apenas palavras-chave (objetos) da frase.
    """
    prompt = f"Extraia apenas os substantivos/objetos da frase: '{label_real}'. Responda apenas as palavras separadas por vírgula."
    
    # Chama o método de geração do seu embedder/model
    inputs = embedder.tokenizer(prompt, return_tensors="pt").to(embedder.device)
    outputs = embedder.model.generate(**inputs, max_new_tokens=20)
    palavras = embedder.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Limpa a resposta: "homem, bicicleta, estrada" -> ["homem", "bicicleta", "estrada"]
    candidatos = [p.strip().lower() for p in palavras.split(',') if len(p.strip()) > 1]
    return candidatos

if __name__ == "__main__":
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    #  Instanciar Encoders (Pesados)
    dino = DinoSceneEncoder(device=device)
    qwen = QwenSceneEmbedder(device=device)

    #  Setup Aligner (Leve)
    # Certifique-se que o visual_dim condiz com o encoder usado (768 para AnyUp/Dino-B)
    aligner = LoRACrossAttentionAligner(visual_dim=768, text_dim=4096)
    
    weights_path = "checkpoints/aligner_epoch_10.pth" # Ou seu peso final
    if os.path.exists(weights_path):
        aligner.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        print(f" Pesos carregados de {weights_path}")
    else:
        print("Checkpoint não encontrado. Rodando com pesos aleatórios para teste.")

    aligner.to(device).eval()
    
    #  Gerador e Avaliador
    generator_sg = SceneGraphGenerator(
        dino_encoder=dino, 
        qwen_embedder=qwen, 
        aligner=aligner, 
        threshold=0.3
    )
    evaluator = SceneGraphEvaluator(generator_sg)
    
    #  Carregar Dataloader de Teste
    # Substitua pelo caminho correto do seu conjunto de teste
    test_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=2, num_workers=4, t="test")

    # --- EXECUÇÃO DA AVALIAÇÃO ---

    # A. Métrica de Projeção (Recall Global)
    print("\n--- Avaliando Alinhamento (Recall@K) ---")
    recall_results = evaluator.evaluate_projection(test_dataloader, k_values=[1, 5, 10])
    for k, v in recall_results.items():
        print(f"{k}: {v:.4f}")
    print(f"Resultados de Busca: {recall_results}")
    # SALVA NO DISCO
    evaluator.salvar_recall_results(recall_results)
    
    print("\n Iniciando Processamento do Dataset de Teste...")

    # Gerador de Grafo de Conhecimento (usa o Qwen para fatos)
    generator_kg = KnowledgeGraphGenerator(
        qwen_model=qwen.model, 
        qwen_tokenizer=qwen.tokenizer
    )

    # Listas para métricas globais
    all_coverages = []

    try:
        # Iterar pelo dataloader de teste
        # Assumindo batch_size=1 para teste qualitativo ou ajustando o loop interno
        for batch_idx, (images, texts) in enumerate(tqdm(test_dataloader, desc="Processando Teste")):
            
            # Processar cada imagem do batch
            for i, img_tensor in enumerate(images):
                # Converter tensor para PIL (necessário para o generator.generate)
                # Dependendo do seu transform, pode precisar de denormalização
                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                
                # Label real do dataset (ground truth)
                label_real = texts[i]
                
                candidatos_dinamicos = extrair_candidatos_llm(label_real, qwen) 
    
                # Adicione categorias genéricas fixas para dar "escolha" ao Aligner
                lista_candidatos = list(set(candidatos_dinamicos + ["person", "vehicle", "object"]))
                
                # Gerar SG com a lista refinada
                
                sg_result = generator_sg.generate(img_pil, lista_candidatos, ["near", "mounted on"])
                
                
                # 2. Gerar KG (Semântica)
                kg_result = generator_kg.generate_from_scene(sg_result)
                
                # 3. Comparar Grafos
                comparativo = evaluator.evaluate_compare_graphs(sg_result, kg_result)
                all_coverages.append(comparativo['semantic_coverage'])
                
                # 4. Gravar JSON
                # Nomeamos o arquivo com o índice do batch e da imagem
                nome_arquivo = f"resultado_batch{batch_idx}_img{i}.json"
                salvar_grafos_json(sg_result, kg_result, comparativo, filename=nome_arquivo)

        # Métricas Finais do Dataset
        media_cobertura = sum(all_coverages) / len(all_coverages) if all_coverages else 0
        print(f"\n Teste Finalizado!")
        print(f" Cobertura Semântica Média no Dataset: {media_cobertura:.2%}")
        print(f" Todos os grafos foram salvos na pasta 'results/'")

    except Exception as e:
        print(f" Erro durante o processamento do dataset: {e}")