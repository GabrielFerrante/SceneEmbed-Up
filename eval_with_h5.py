import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from models.SG.projection import LoRACrossAttentionAligner
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from models.SG.generation import SceneGraphGenerator, KnowledgeGraphGenerator, salvar_grafos_json
from data.data_utils_pytorch import ShardedH5Dataset
from data.data_utils_pytorch import create_all_dataloaders
from torch.utils.data import DataLoader
import os
import json
from datetime import datetime
from collections import defaultdict

class SceneGraphEvaluator:
    def __init__(self, generator, device="cuda"):
        self.generator = generator
        self.device = device
        self.dtype = generator.dtype

    @torch.no_grad()
    def evaluate_projection(self, dataloader, k_values=[1, 5, 10]):
        """
        Calcula Recall@K usando Similaridade Patch-Texto com Max Pooling.
        """
        self.generator.aligner.eval()
        
        all_image_patches = [] # Lista de tensores [N_patches, Dim]
        all_text_queries = []  # Lista de tensores [Dim]

        print("1. Extraindo Patches e Embeddings de Texto...")
        for visual_input, text_queries in tqdm(dataloader):
            visual_input = visual_input.to(device)
            text_queries = text_queries.to(device)
            
            for i in range(B):
                # Armazena patches crus (sem passar no aligner ainda!)
                patches_i = visual_input[i]  # [N_patches, 768] 
                all_image_patches.append(patches_i.cpu())
                
                # Armazena embedding de texto normalizado
                all_text_queries.append(text_queries[i].cpu())


        num_samples = len(all_text_queries)
        text_embeddings = torch.stack(all_text_queries) # [N_total, Dim]
        
        # 2. Rankear Imagens via Max Pooling de Similaridade
        print(f"2. Calculando Matriz de Similaridade [Max Pooling] para {num_samples} amostras...")
        sim_matrix = torch.zeros((num_samples, num_samples))

        for i in range(num_samples):

            patches_i = all_image_patches[i].to(self.device)  # [P, D]

            # Expandimos patches para todos os textos
            patches_batch = patches_i.unsqueeze(0).expand(
                num_samples, -1, -1
            )  # [N, P, D]

            # Textos já estão em batch
            text_batch = text_embeddings.unsqueeze(1)  # [N, 1, D]

            #  Um único forward
            p_refined, _ = self.generator.aligner(
                patches_batch,   # [N, P, D]
                text_batch       # [N, 1, D]
            )

            p_refined = F.normalize(p_refined, dim=-1)  # [N, P, D]

            # Similaridade patch-text
            sim = torch.matmul(
                p_refined,
                text_embeddings.unsqueeze(-1)  # [N, D, 1]
            ).squeeze(-1)  # [N, P]

            # max over patches
            scores = sim.max(dim=1).values  # [N]

            sim_matrix[i] = scores

        # 3. Calcular Recall@K
        results = {}
        labels = torch.arange(num_samples)
        for k in k_values:
            if k > num_samples: continue
            
            # Top-k índices ao longo da dimensão das imagens (rankeando quais imagens batem com o texto)
            # Ou vice-versa. Aqui: para cada texto (coluna), quais as top-k imagens (linhas)?
            _, top_k = torch.topk(sim_matrix, k=k, dim=0) # [k, N_total]
            
            # Verifica se o índice correto está no top-k
            correct = torch.any(top_k == labels.unsqueeze(0), dim=0)
            results[f"Recall@{k}"] = correct.float().mean().item()
            
        return results
    
    def evaluate_expansion(scene_g, kg_g):
        """
        avalia o ganho semantico
        """

        scene_labels = {
                n['label'].lower().strip()
                for n in scene_g['nodes']
        }
        if not scene_labels:
            return {"expansion_ratio": 0.0}
        
        kg_expanded_entities = {
                edge['obj']
                for edge in kg_g['factual_edges']
        }

        expansion = len(kg_expanded_entities) / len(scene_labels)
        
        

        return {"expansion_ratio": expansion}
    
    def compute_mean_hypernym_count(scene_g, kg_g):
        """
        Calcula o número médio de hiperônimos (is_a)
        por objeto da cena.
        """

        # Labels da cena normalizadas
        scene_labels = {
            n['label'].lower().strip()
            for n in scene_g.get('nodes', [])
        }

        if not scene_labels:
            return {"mean_hypernym_count": 0.0}

        # Contador de hiperônimos por entidade
        hypernym_counter = defaultdict(int)

        for edge in kg_g.get('factual_edges', []):
            sub = edge.get('sub', '').lower().strip()
            rel = edge.get('rel', '').lower().strip()

            if rel == "is_a" and sub in scene_labels:
                hypernym_counter[sub] += 1

        # Soma total de hiperônimos
        total_hypernyms = sum(hypernym_counter.values())

        mean_hypernyms = total_hypernyms / len(scene_labels)

        return {
            "mean_hypernym_count": mean_hypernyms
        }
        
    def evaluate_compare_graphs(self, scene_g, kg_g):
        """
        Compara Scene Graph com Knowledge Graph.
        
        Retorna métricas de:
        - semantic_coverage
        - entity_recall
        - relation_consistency
        - structural_density
        """

        # ==========================
        # 1. Normalização
        # ==========================
        scene_labels = {
            node["label"].lower().strip()
            for node in scene_g.get("nodes", [])
        }

        kg_entities = {
            ent.lower().strip()
            for ent in kg_g.get("entities", [])
        }

        # ==========================
        # 2. Cobertura Semântica
        # ==========================
        if len(scene_labels) == 0:
            semantic_coverage = 0.0
        else:
            semantic_coverage = len(
                scene_labels.intersection(kg_entities)
            ) / len(scene_labels)

        # ==========================
        # 3. Entity Recall (KG-side)
        # ==========================
        if len(kg_entities) == 0:
            entity_recall = 0.0
        else:
            entity_recall = len(
                scene_labels.intersection(kg_entities)
            ) / len(kg_entities)

        # ==========================
        # 4. Relation Consistency
        # ==========================
        sg_relations = {
            (
                scene_g["nodes"][edge["source"]]["label"].lower().strip(),
                edge["relation"].lower().strip(),
                scene_g["nodes"][edge["target"]]["label"].lower().strip()
            )
            for edge in scene_g.get("edges", [])
            if edge["source"] < len(scene_g["nodes"])
            and edge["target"] < len(scene_g["nodes"])
        }

        kg_relations = {
            (
                rel[0].lower().strip(),
                rel[1].lower().strip(),
                rel[2].lower().strip()
            )
            for rel in kg_g.get("relations", [])
            if len(rel) == 3
        }

        if len(sg_relations) == 0:
            relation_consistency = 0.0
        else:
            relation_consistency = len(
                sg_relations.intersection(kg_relations)
            ) / len(sg_relations)

        # ==========================
        # 5. Structural Density
        # ==========================
        num_nodes = len(scene_g.get("nodes", []))
        num_edges = len(scene_g.get("edges", []))

        if num_nodes <= 1:
            structural_density = 0.0
        else:
            max_possible_edges = num_nodes * (num_nodes - 1)
            structural_density = num_edges / max_possible_edges

        # ==========================
        # 6. Resultado Final
        # ==========================
        return {
            "semantic_coverage": semantic_coverage,
            "entity_recall": entity_recall,
            "relation_consistency": relation_consistency,
            "structural_density": structural_density,
            "num_nodes": num_nodes,
            "num_edges": num_edges
        }
    
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
    
    

def extrair_candidatos_llm(label_real, embedder):
    """
    Usa o Qwen para extrair apenas palavras-chave (objetos) da frase.
    """
    prompt = f"Return only single-word concrete physical objects of {label_real}. No verbs. No adjectives. No determiners. Format: word1, word2, word3"
    
    # Chama o método de geração do seu embedder/model
    inputs = embedder.tokenizer(prompt, return_tensors="pt").to(embedder.device)
    outputs = embedder.model.generate(**inputs, max_new_tokens=20)
    input_len = inputs.input_ids.shape[1]
    palavras = embedder.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    # Limpa a resposta: "homem, bicicleta, estrada" -> ["homem", "bicicleta", "estrada"]
    candidatos = [p.strip().lower() for p in palavras.split(',') if len(p.strip()) > 1]
    return candidatos

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
        print(f"Pesos carregados de {weights_path}")
    else:
        print("Checkpoint não encontrado. Rodando com pesos aleatórios.")

    aligner.to(device).eval()

 
    generator_sg = SceneGraphGenerator(
        dino_encoder=dino,
        qwen_embedder=qwen,
        aligner=aligner,
        threshold=0.3
    )

    generator_kg = KnowledgeGraphGenerator(
        qwen_model=qwen.model,
        qwen_tokenizer=qwen.tokenizer
    )

    evaluator = SceneGraphEvaluator(generator_sg)

    test_ds = ShardedH5Dataset("F:/COYO/embeds/test_anyup/")
    test_dataloader = DataLoader(test_ds, batch_size=16, shuffle=True, pin_memory=True)

  
    print("\n--- Avaliando Alinhamento (Recall@K) ---")
    recall_results = evaluator.evaluate_projection(
        test_dataloader,
        k_values=[1, 5, 10]
    )

    for k, v in recall_results.items():
        print(f"{k}: {v:.4f}")

    evaluator.salvar_recall_results(recall_results)


    print("\nIniciando Processamento do Dataset de Teste...")

    # Métricas globais
    all_coverages = []
    all_expansions = []
    all_hypernym_means = []
    all_node_counts = []
    all_edge_counts = []

    try:
        for batch_idx, (images, texts) in enumerate(
            tqdm(test_dataloader, desc="Processando Teste")
        ):
            for i, img_tensor in enumerate(images):

                img_pil = transforms.ToPILImage()(img_tensor.cpu())
                label_real = texts[i]

             
                candidatos_dinamicos = extrair_candidatos_llm(
                    label_real, qwen
                )

                lista_candidatos = list(
                    set(candidatos_dinamicos + ["person", "vehicle", "object"])
                )

              
                sg_result = generator_sg.generate(
                    img_pil,
                    lista_candidatos,
                    ["near", "mounted on"]
                )

                
                kg_result = generator_kg.generate_from_scene(
                    sg_result
                )

               

                # Cobertura
                comparativo = evaluator.evaluate_compare_graphs(
                    sg_result, kg_result
                )
                semantic_coverage = comparativo["semantic_coverage"]
                all_coverages.append(semantic_coverage)

                # Expansão
                expansion_result = evaluator.evaluate_expansion(
                    sg_result, kg_result
                )
                all_expansions.append(
                    expansion_result["expansion_ratio"]
                )

                # Hypernym mean
                mean_hypernym = evaluator.compute_mean_hypernym_count(
                    sg_result
                )
                all_hypernym_means.append(mean_hypernym)

                # Estatísticas estruturais
                all_node_counts.append(len(sg_result["nodes"]))
                all_edge_counts.append(len(sg_result["edges"]))

                
                nome_arquivo = f"resultado_batch{batch_idx}_img{i}.json"
                salvar_grafos_json(
                    sg_result,
                    kg_result,
                    comparativo,
                    filename=nome_arquivo
                )

        
        def safe_mean(values):
            return sum(values) / len(values) if values else 0

        print("\nTeste Finalizado!")
        print("=" * 50)
        print(f"Recall Results: {recall_results}")
        print(f"Cobertura Semântica Média: {safe_mean(all_coverages):.4f}")
        print(f"Expansion Ratio Médio: {safe_mean(all_expansions):.4f}")
        print(f"Mean Hypernym Count: {safe_mean(all_hypernym_means):.4f}")
        print(f"Nós Médios por SG: {safe_mean(all_node_counts):.2f}")
        print(f"Relações Médias por SG: {safe_mean(all_edge_counts):.2f}")
        print("=" * 50)

    except Exception as e:
        print(f"Erro durante o processamento: {e}")