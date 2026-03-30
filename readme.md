# SceneEmbed-Up

**Alinhamento Multimodal de Alta Resolução para Geração de Scene Graphs**

Pipeline de pesquisa que combina **DINOv3** (encoder visual com upsampling HR) + **Qwen3-Embedding-8B** (encoder textual) + **LoRA Cross-Attention Aligner** para aprender um espaço de embeddings compartilhado e gerar Scene Graphs semânticos enriquecidos com Knowledge Graphs.

---

## Visão Geral da Arquitetura

```
Imagem
  └─► DinoSceneEncoder (ViT-B/16 + AnyUp)
        ├─ CLS token    [1, 768]            ← representação global
        └─ HR patches   [B, 768, H, W]
              │
              │  pool 32×32 + flatten
              ▼
          [B, 1024, 768]
              │
              ▼
  LoRACrossAttentionAligner
    visual_proj (frozen) + LoRA (rank=16)
    CrossAttention: Q=text, K=V=visual
              │
              ▼
      attn_output [B, N_queries, 4096]
              │
    ┌─────────┴──────────┐
    ▼                    ▼
Scene Graph         Knowledge Graph
(Nós + Arestas      (Tripletas is_a
 direcionais)        via Qwen3 LLM)
```

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
![alt text](image-4.png)

---

## Funcionalidades

- **Encoder visual de alta resolução:** DINOv3 ViT-B/16 com upsampling por [AnyUp](https://github.com/wimmerth/anyup) ou FeatUp (JBU Stack), preservando detalhes espaciais finos.
- **Encoder textual de alta capacidade:** Qwen3-Embedding-8B com mean pooling e instrução de tarefa contextual.
- **Aligner com LoRA:** Projeção visual congelada + adaptadores LoRA treináveis (~200K params), cross-attention assimétrico texto→visual.
- **Scene Graph generativo:** Detecção de objetos por thresholding de similaridade + inferência de relações direcionais via cross-attention entre nós.
- **Knowledge Graph expansivo:** Extração de fatos taxonômicos (`is_a`) via prompting estruturado do Qwen3.
- **Pipeline de dados COYO-700M:** Suporte a ~15M imagens com dataloaders eficientes via shards HDF5 com buffer rotativo em RAM.
- **Métricas completas:** Recall@K bidirecional (I2T + T2I), semantic coverage, entity recall, expansion ratio, mean hypernym count.

---

## Estrutura do Projeto

```
SceneEmbed-Up/
├── models/
│   ├── encoders/
│   │   ├── dinov3_extrator.py       # DinoSceneEncoder
│   │   └── qwen3_extrator.py        # QwenSceneEmbedder
│   ├── aligners/
│   │   └── lora_cross_attention.py  # LoRACrossAttentionAligner
│   ├── ups/
│   │   └── hr_conversions.py        # AnyUpModel, JBUStack, LoftUpModel
│   └── SG/
│       └── generation.py            # SceneGraphGenerator, KnowledgeGraphGenerator
├── data/
│   ├── data_utils_pytorch.py        # Datasets e DataLoaders
│   ├── extract_files_coyo.py        # Extração dos .tar do img2dataset
│   └── get_metadados_coyo.py        # Download de metadados COYO
├── embeddings/
│   ├── generate_shards.py           # Exportação de embeddings em shards H5
│   ├── consolidate_shards.py        # Consolidação com shuffle
│   └── fix_shard_shapes.py          # Correção de shapes
├── utils/
│   ├── early_stopping.py            # EarlyStopping
│   ├── logging_utils.py             # TensorBoard writer
│   ├── checkpoint.py                # Salvar checkpoints por época
│   ├── graph_io.py                  # Persistência de grafos em JSON
│   ├── graph_viz.py                 # Visualização com networkx
│   └── metrics_scene_graph.py       # Métricas de avaliação
├── train_with_h5.py                 # Treino rápido com embeddings pré-computados
├── train_with_Images.py             # Treino fim-a-fim com imagens
├── eval_with_h5.py                  # Avaliação com Recall@K bidirecional
├── eval_with_images.py              # Avaliação com geração de SG/KG
├── memory.mdc                       # Contexto rápido para agentes/LLMs
└── Docs/
    └── ARCHITECTURE.md              # Documentação detalhada da arquitetura
```

---

## Instalação

```bash
# Clone o repositório
git clone <repo-url>
cd SceneEmbed-Up

# Instale dependências principais
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers huggingface_hub
pip install h5py torch_geometric networkx matplotlib tqdm Pillow numpy

# AnyUp (requer natten para atenção local)
pip install natten
# AnyUp é carregado via torch.hub automaticamente na primeira execução

# Token HuggingFace para DINOv3 (modelo restrito)
# Crie tokenDINOV3.json: {"token": "hf_..."}
```

---

## Uso Rápido

### 1. Download e preparação dos dados

```bash
# Baixar metadados COYO
python data/get_metadados_coyo.py

# Download das imagens via img2dataset
img2dataset --url_list F:/COYO/coyo_meta/data \
  --input_format "parquet" --url_col "url" --caption_col "text" \
  --output_format webdataset --output_folder F:/COYO/coyo \
  --processes_count 16 --thread_count 64 --image_size 256 \
  --number_sample 15000000

# Extrair arquivos .tar
python data/extract_files_coyo.py
```

### 2. Gerar embeddings pré-computados

```bash
# Gera shards H5 diretamente (recomendado)
python embeddings/generate_shards.py

# Consolidar com shuffle para treino
python embeddings/consolidate_shards.py
```

### 3. Treinar o Aligner

```bash
# Modo rápido (usa embeddings pré-computados) — recomendado
python train_with_h5.py

# Modo fim-a-fim (requer +24 GB VRAM)
python train_with_Images.py
```

### 4. Avaliar

```bash
# Recall@K bidirecional + métricas de SG/KG
python eval_with_h5.py

# Apenas com imagens reais
python eval_with_images.py
```

### 5. Visualizar grafos gerados

```bash
python viz/graphs_viz.py
# Gera PNG para cada JSON em results/
```

---

## Formato dos Embeddings (HDF5)

Cada shard `.h5` contém:

| Dataset         | Shape             | Dtype   | Descrição                    |
|-----------------|-------------------|---------|------------------------------|
| `visual_feats`  | `[N, 1024, 768]`  | float16 | Patches 32×32 do DINOv3      |
| `text_feats`    | `[N, 1, 4096]`    | float16 | Embedding Qwen3 da legenda   |
| `visual_global` | `[N, 768]`        | float16 | Token CLS do DINOv3          |

Armazenamento: 7Gb por shard de 5k amostras (compressão gzip).

---

## Função de Loss

```
loss = contrastive_loss + 0.5 × loss_vg + 0.05 × entropy_reg
```

| Componente          | Descrição                                                        |
|---------------------|------------------------------------------------------------------|
| `contrastive_loss`  | InfoNCE simétrico entre visual refinado e texto (τ=0.07)        |
| `loss_vg`           | InfoNCE sobre média dos patches antes do cross-attention        |
| `entropy_reg`       | `(mean_entropy - 1.5)²` → foca atenção em ~4-5 patches/head    |

---

## Métricas

### Retrieval (Alinhamento)

| Métrica             | Descrição                                          |
|---------------------|----------------------------------------------------|
| `I2T_Recall@K`      | Top-K imagens → encontra o texto correto          |
| `T2I_Recall@K`      | Top-K textos → encontra a imagem correta          |
| `Mean_Recall@K`     | Média bidirecional (padrão CLIP/BLIP)             |

### Scene/Knowledge Graphs

| Métrica                | Descrição                                         |
|------------------------|---------------------------------------------------|
| `semantic_coverage`    | Cobertura das entidades visuais pelo KG           |
| `entity_recall`        | Recall das entidades KG no SG                     |
| `expansion_ratio`      | Entidades KG / nós SG                            |
| `mean_hypernym_count`  | Média de relações `is_a` por objeto              |
| `structural_density`   | Densidade de arestas no SG                        |

---
                    

## Logs e Checkpoints

```
logs/
  └── YYYYMMDD-HHMMSS/   ← TensorBoard (loss, acc, entropy por step e época)

checkpoints/
  ├── best_aligner.pth   ← Melhor modelo (EarlyStopping)
  └── aligner_epoch_N.pth

results/
  ├── resultado_batch0_img0.json
  ├── resultado_batch0_img0.png  ← grafo visualizado
  └── recall_metrics.json
```

Visualizar logs:
```bash
tensorboard --logdir logs/
```

---

## Documentação

- [`Docs/ARCHITECTURE.md`](Docs/ARCHITECTURE.md) — Arquitetura completa e detalhada
- [`memory.mdc`](memory.mdc) — Referência rápida para agentes/LLMs

---

## Referências

- [DINOv2](https://github.com/facebookresearch/dinov2) — Meta AI
- [Qwen3](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — Alibaba Cloud
- [AnyUp](https://github.com/wimmerth/anyup) — Wimmer et al.
- [FeatUp](https://github.com/mhamilton723/FeatUp) — Hamilton et al.
- [COYO-700M](https://github.com/kakaobrain/coyo-dataset) — Kakao Brain
- [img2dataset](https://github.com/rom1504/img2dataset) — Beaumont

---

## Convenções do Código

- **Shapes documentados** em todas as funções com `[B, C, H, W]`
- **`@torch.no_grad()`** em métodos de inferência
- **`bfloat16`** para operações GPU e armazenamento H5
- **Context managers** obrigatórios para H5PY
- **`tqdm`** em todos os loops de processamento de dados
- Erros de GPU → `torch.cuda.memory_summary()` antes de raise
