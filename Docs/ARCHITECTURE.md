# Arquitetura do SceneEmbed-Up

> Documentação completa e detalhada da arquitetura do projeto.  
> Última atualização: gerada automaticamente a partir do estado atual do código.

---

## Visão Geral

O **SceneEmbed-Up** é um pipeline multimodal de pesquisa com dois objetivos centrais:

1. **Alinhamento visual-textual** — aprender um espaço de embeddings compartilhado entre patches de imagem (DINOv3) e texto (Qwen3), de modo que objetos visuais e suas descrições linguísticas sejam vizinhos nesse espaço.
2. **Geração de Grafos de Cena** — usar esse alinhamento para detectar objetos em imagens, inferir relações espaciais/semânticas entre eles (Scene Graph) e expandi-los com conhecimento factual do LLM (Knowledge Graph).

---

## Diagrama de Alto Nível

```
Imagem (PIL/Tensor)
        │
        ▼
┌──────────────────────┐
│   DinoSceneEncoder   │  (facebook/dinov3-vitb16-pretrain-lvd1689m)
│  ┌────────────────┐  │
│  │  ViT-B/16 DINO │  │
│  └────────────────┘  │
│  CLS token [1, 768]  │  ← representação global da cena
│  Patches [B,768,H,W] │  ← mapa espacial de features
│  + Upsampler (AnyUp) │  → [B, 768, H_orig, W_orig] ou fixo (224,224)
└──────────────────────┘
        │
        │ adaptive_avg_pool2d → [B, 768, 32, 32]
        │ reshape             → [B, 1024, 768]  (patches achatados)
        │
        ▼
┌──────────────────────────────────────────────┐
│         LoRACrossAttentionAligner            │
│  ┌───────────────────────────────────────┐   │
│  │  visual_proj (congelado): 768 → 4096  │   │
│  │  + LoRA delta: A[768,16] @ B[16,4096] │   │
│  └───────────────────────────────────────┘   │
│              v_features [B, 1024, 4096]       │
│                     │                        │
│              CrossAttention                  │ ← text_queries [B, N, 4096]
│              (Q=text, K=V=visual)            │
│                     │                        │
│              attn_output [B, N, 4096]        │
└──────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐    ┌──────────────────────────┐
│    Scene Graph Generator    │    │  Knowledge Graph Generator│
│  Threshold + Retrieval Score│    │  (Qwen3 geração de texto) │
│  Nós + Arestas Direcionais  │    │  Tripletas is_a           │
└─────────────────────────────┘    └──────────────────────────┘
```

---

## Módulos Detalhados

### 1. Encoders (`models/encoders/`)

#### 1.1 DinoSceneEncoder

**Arquivo:** `models/encoders/dinov3_extrator.py`

| Parâmetro       | Valor padrão                                    |
|-----------------|-------------------------------------------------|
| Modelo          | `facebook/dinov3-vitb16-pretrain-lvd1689m`      |
| Token auth      | `tokenDINOV3.json`                              |
| Device          | CUDA se disponível                              |
| Upsampler       | `"anyup"` (padrão) ou `"featup"`               |

**Método `extract_features(img_tensor)`:**

```
Entrada : img_tensor (PIL Image ou Tensor)
Saída   :
  cls_token   : [B, 768]           — token CLS (representação global)
  hr_features : [B, 768, H, W]     — mapa HR de features espaciais
```

**Fluxo interno:**
1. `AutoImageProcessor` normaliza e prepara o tensor.
2. `AutoModel` (ViT-B/16) produz `last_hidden_state [B, N_total, 768]`.
3. CLS token extraído de `[:, 0, :]`.
4. Patches espaciais: `[:, 1:h*w+1, :]` → reshape `[B, 768, h, w]`.
5. Upsampler sobe resolução para HR:
   - **AnyUp** (`wimmerth/anyup`): resolução original da imagem.
   - **FeatUp** (`JBUStack`): fixo `[B, 384, 224, 224]` + `nn.Conv2d(768→384)`.

**Upsamplers disponíveis (`models/ups/hr_conversions.py`):**

| Classe          | Descrição                                          | Output dim |
|-----------------|----------------------------------------------------|------------|
| `AnyUpModel`    | torch.hub multi-backbone, usa natten               | 768        |
| `JBUStack`      | Joint Bilateral Upsample com kernels aprendidos    | feat_dim   |
| `LoftUpModel`   | torch.hub CLIP-based (alternativo)                 | varies     |

O `JBULearnedRange` implementa um kernel dinâmico por pixel usando `AdaptiveConv`:
- Kernel spatial Gaussiano aprendível (`sigma_spatial`).
- Kernel range projetado via atenção softmax.
- Upsampling 4× em cascata (4 estágios de 2×).

---

#### 1.2 QwenSceneEmbedder

**Arquivo:** `models/encoders/qwen3_extrator.py`

| Parâmetro | Valor                     |
|-----------|---------------------------|
| Modelo    | `Qwen/Qwen3-Embedding-8B` |
| Dtype     | `bfloat16`                |
| Device    | CUDA                      |

**Método `embed_components(batch_texts, max_length=512, normalize=True)`:**

```
Entrada : batch_texts : List[List[str]] ou List[str]
          ex: [["cat", "on top of"], ["dog", "near"]]

Saída   : Tensor [B, N_textos, 4096]  (normalizado L2 se normalize=True)
```

**Detalhes:**
- Cada texto recebe o prefixo `Instruct: {task}\nQuery:{text}`.
- Pooling por **média ponderada pela attention mask** (mean pooling).
- Reshape crucial: `[B*N, 4096]` → `[B, N, 4096]`.

---

### 2. Aligner (`models/aligners/lora_cross_attention.py`)

**Classe:** `LoRACrossAttentionAligner`

```
Entrada:
  hr_patches    : [B, N_patches, 768]   — patches visuais achatados
  text_queries  : [B, N_queries, 4096]  — embeddings textuais Qwen

Saída:
  attn_output   : [B, N_queries, 4096]  — texto refinado pelo contexto visual
  attn_weights  : [B, heads, N_queries, N_patches]
  v_features    : [B, N_patches, 4096]  — patches projetados no espaço do texto
```

**Componentes:**

```
visual_proj: Linear(768, 4096)    ← CONGELADO (base de projeção)
lora_A_v:   Parameter [768, 16]   ← treinável (inicialização Kaiming)
lora_B_v:   Parameter [16, 4096]  ← treinável (inicialização zero)
scaling:    32.0 / rank           ← alpha=32, rank=16

v_features = visual_proj(patches) + scaling * (patches @ lora_A @ lora_B)

CrossAttention:
  Q = text_queries   [B, N_q, 4096]
  K = V = v_features [B, N_p, 4096]
  num_heads = 8, batch_first = True
```

O design congela a projeção base para estabilidade e treina apenas os parâmetros LoRA (~200K parâmetros treináveis vs ~3M totais).

**Função auxiliar `calculate_retrieval_score`:**
- Input: `visual_aligned [N, D]` + `text_embedding [D]`
- Output: cosine similarity média (multi-termo) ou escalar (vetor único)

---

### 3. Geração de Grafos (`models/SG/generation.py`)

#### 3.1 SceneGraphGenerator

**Fluxo `generate(image, candidate_nodes, candidate_relations)`:**

```
1. extract_features(image) → lr_feat [1, 768, H, W]
2. adaptive_avg_pool2d → [1, 768, 32, 32]
3. reshape → visual_input [1, 1024, 768]

4. embed_components(candidate_nodes) → node_queries [1, N_nodes, 4096]
5. aligner(visual_input, node_queries) → node_embeddings_refined, attn_weights

6. Para cada nó candidato:
   score = cosine_similarity(node_embedding_refined, node_query_original)
   se score > threshold (0.3): adicionar ao scene_graph["nodes"]

7. embed_components(candidate_relations) → rel_queries [1, N_rels, 4096]
8. aligner(visual_input, rel_queries) → rel_embeddings_refined

9. Para cada par (node_a, node_b):
   pair_context = _compute_directional_context(a.embedding, b.embedding)
   Para cada relação:
     rel_score = cosine_similarity(rel_embedding, pair_context)
     se rel_score > 0.55: adicionar aresta

Saída:
  {
    "nodes": [{"id", "label", "embedding", "attn_weights", "score"}],
    "edges": [{"source", "relation", "target", "confidence"}]
  }
```

**`_compute_directional_context(query_node, context_node)`:**  
Cross-attention simples (Q=sujeito, K=V=objeto) que captura a relação direcional A→B.  
Isso garante que `(A rel B) ≠ (B rel A)`.

---

#### 3.2 KnowledgeGraphGenerator

**Fluxo `generate_from_scene(scene_graph)`:**

Para cada label detectado no scene graph, gera um prompt estruturado ao Qwen:
```
<|im_start|>system: List 2 universal taxonomy-level facts about '{label}'...
<|im_start|>user: Format strictly as: Subject | is_a | Object...
```

Parser extrai tripletas `sub | is_a | obj` do output textual.

```
Saída:
  {
    "entities": ["cat", "animal", "mammal", ...],
    "factual_edges": [{"sub": "cat", "rel": "is_a", "obj": "animal"}, ...]
  }
```

---

### 4. Pipeline de Dados (`data/`)

#### 4.1 Aquisição

```
HuggingFace COYO-700M
        │
        │ img2dataset (parquet → .tar webdataset)
        ▼
F:/COYO/coyo/*.tar
        │
        │ extract_files_coyo.py
        ▼
F:/COYO/coyo/extracted/
  └── 00000/
      ├── 000000001.jpg
      ├── 000000001.txt   ← caption
      └── ...
```

**Comando img2dataset:**
```bash
img2dataset --url_list F:/COYO/coyo_meta/data \
  --input_format "parquet" --url_col "url" --caption_col "text" \
  --output_format webdataset --output_folder F:/COYO/coyo \
  --processes_count 16 --thread_count 64 --image_size 256 \
  --number_sample 15000000
```

#### 4.2 Datasets PyTorch

**`CoyoExtractedDataset`:**
- Varre recursivamente `root_dir` para `.jpg/.jpeg/.png`.
- Lê caption do `.txt` correspondente.
- `__getitem__` retorna `(image_tensor [3,H,W], caption str)`.
- `None` em caso de imagem corrompida (filtrado pelo `CoyoCollate`).

**Split padrão (seed=42):**
```
Total ~7.8M imagens
  train : 500,000
  val   : 10,000
  test  : 10,000
  unused: ~7.3M
```

**`ShardedH5Dataset_withSSD`:**
- Para val/test em SSD externo.
- Abre cada arquivo H5 individualmente por amostra (IO aleatório).
- Compatível com `num_workers > 0`.

**`ShardedH5Dataset_withHD`:**
- Para treino em HD mecânico.
- Buffer rotativo: carrega `shards_in_memory` shards na RAM.
- Prefetch em thread de background.
- `rotate_buffer()` chamado ao final de cada época.

---

### 5. Pipeline de Embeddings (`embeddings/`)

#### 5.1 Geração

```
DataLoader (imagens + captions)
        │
        ├── DinoSceneEncoder.extract_features()
        │   ├── CLS token [B, 768]
        │   └── HR patches → pool 32×32 → [B, 1024, 768]
        │
        └── QwenSceneEmbedder.embed_components()
            └── [B, 1, 4096]
                    │
                    ▼
            ShardWriter → shard_XXXXXX.h5
              visual_feats  [N, 1024, 768] float16
              text_feats    [N, 1,    4096] float16
              visual_global [N, 768]        float16
```

**`ShardWriter` (em `generate_shards.py`):**
- Datasets HDF5 resizáveis (`maxshape=(None, ...)`, `chunks=(64, ...)`)
- Retomada automática: detecta último shard incompleto
- Validação de integridade após cada shard fechado

#### 5.2 Consolidação

```
shard_000000.h5, shard_000001.h5, ...
        │
        │ consolidate_shards.py
        │ (shuffle interno, groups de N pastas)
        ▼
consolidated_0000.h5, consolidated_0001.h5, ...
```

---

### 6. Treinamento

#### 6.1 `train_with_h5.py` (modo principal)

**Hiperparâmetros:**
| Parâmetro        | Valor  |
|------------------|--------|
| epochs           | 100    |
| batch_size       | 64     |
| lr               | 1e-4   |
| weight_decay     | 0.01   |
| temperature      | 0.07   |
| TARGET_ENTROPY   | 1.5    |
| LAMBDA_ENTROPY   | 0.05   |
| grad_clip        | 1.0    |

**Função de Loss:**
```python
# 1. Contrastivo principal (visual refinado ↔ texto)
v_norm  = normalize(attn_output.squeeze(1))  # [B, 4096]
t_norm  = normalize(text_queries.squeeze(1)) # [B, 4096]
logits  = v_norm @ t_norm.T / 0.07           # [B, B]
contrastive_loss = (CE(logits, labels) + CE(logits.T, labels)) / 2

# 2. Supervisão direta dos patches (sem passar pelo cross-attn)
v_global = v_features.mean(dim=1)            # [B, 4096]
loss_vg  = (CE(v_global @ t.T/0.07, labels) + simetrico) / 2

# 3. Regularização de entropia da atenção
entropy     = -sum(w * log(w + 1e-8), dim=-1)  # [B, heads, queries]
entropy_reg = (entropy.mean() - 1.5) ** 2

loss = contrastive_loss + 0.5 * loss_vg + 0.05 * entropy_reg
```

**Schedulers e callbacks:**
- `EarlyStopping(patience=10, min_delta=0.001)` — salva `best_aligner.pth`
- `save_epoch_checkpoint(aligner, epoch)` — salva `aligner_epoch_N.pth`
- TensorBoard: loss total, componentes, accuracy e entropia por step + por época

#### 6.2 `train_with_Images.py` (modo fim-a-fim)

Mesmo loop mas:
- Extrai features em tempo real via `DinoSceneEncoder` e `QwenSceneEmbedder`
- `batch_size=4` (limitado por VRAM com todos os modelos carregados)
- `torch.amp.autocast(dtype=bfloat16)` para eficiência

---

### 7. Avaliação

#### 7.1 Recall@K Bidirecional (`eval_with_h5.py`)

```
1. Iterar dataloader H5:
   - aligner(visual, text) → attn_output
   - v_norm = normalize(attn_output.squeeze(1))  [B, 4096]
   - t_norm = normalize(text.squeeze(1))          [B, 4096]

2. Concatenar todos: V [N, 4096], T [N, 4096]

3. Construir sim_matrix [N, N] em chunks (controle VRAM):
   sim_matrix[i] = T[i:i+C] @ V.T
   sim_matrix = (sim_matrix + sim_matrix.T) / 2  ← simetrização

4. I2T Recall@K: para cada imagem i, top-K textos → acerto se j==i
5. T2I Recall@K: para cada texto i, top-K imagens → acerto se j==i
6. Mean Recall@K = (I2T + T2I) / 2
```

#### 7.2 Métricas de Grafos (`utils/metrics_scene_graph.py`)

| Métrica                | Descrição                                                      |
|------------------------|----------------------------------------------------------------|
| `semantic_coverage`    | `|SG_labels ∩ KG_entities| / |SG_labels|`                     |
| `entity_recall`        | `|SG_labels ∩ KG_entities| / |KG_entities|`                   |
| `relation_consistency` | `|SG_triplets ∩ KG_triplets| / |SG_triplets|`                 |
| `structural_density`   | `|edges| / (|nodes| * (|nodes| - 1))`                         |
| `expansion_ratio`      | `|KG_expanded_entities| / |SG_labels|`                        |
| `mean_hypernym_count`  | Média de relações `is_a` por objeto do scene graph            |

---

### 8. Visualização e Persistência

**`utils/graph_io.py`** — `salvar_grafos_json`:
- Persiste `{metadata, scene_graph, knowledge_graph}` em JSON.
- `_to_jsonable` converte tensores/numpy recursivamente para tipos serializáveis.

**`utils/graph_viz.py`** — `visualizar_e_salvar_grafo`:
- Lê JSON de resultado.
- Constrói grafo `networkx.DiGraph` com nós coloridos por tipo (azul=visual, verde=factual).
- Salva PNG com layout spring (`k=0.6, iterations=70`) e etiquetas de relações.
- Retorna `torch_geometric.data.Data` com `edge_index`.

---

## Fluxo Completo de Uso

```
1. Baixar dados:
   python data/get_metadados_coyo.py
   # executar img2dataset (ver data/command_download.md)
   python data/extract_files_coyo.py

2. Gerar embeddings pré-computados:
   python embeddings/generate_shards.py
   # ou: python embeddings/generate_small_files.py → python embeddings/small_to_shards.py
   python embeddings/consolidate_shards.py

3. Corrigir shapes se necessário:
   python embeddings/fix_shard_shapes.py

4. Treinar aligner:
   python train_with_h5.py          # rápido, pré-computado
   # ou
   python train_with_Images.py      # fim-a-fim (lento, mais VRAM)

5. Avaliar:
   python eval_with_h5.py           # Recall@K + SG/KG metrics
   # ou
   python eval_with_images.py

6. Visualizar grafos:
   python viz/graphs_viz.py
```

---

## Considerações de VRAM

| Modo              | Modelos ativos        | VRAM estimada |
|-------------------|-----------------------|---------------|
| train_with_h5     | Aligner apenas        | ~2-4 GB       |
| train_with_Images | DINO + Qwen + Aligner | ~24+ GB       |
| eval_with_h5      | Aligner apenas        | ~2-4 GB       |
| eval_with_images  | DINO + Qwen + Aligner | ~24+ GB       |

DINO ViT-B ~330M params, Qwen3-8B ~8B params (bfloat16 → ~16 GB).

---

## Decisões de Design Relevantes

1. **LoRA sobre visual_proj congelado**: Permite adaptar a projeção visual sem esquecimento catastrófico. A projeção base fornece um bom ponto de partida (espaço 768→4096) enquanto o LoRA aprende o alinhamento específico da tarefa.

2. **Cross-attention texto-como-query**: O texto direciona a atenção sobre os patches visuais (não o contrário), alinhado com o objetivo de "encontrar evidências visuais para conceitos textuais".

3. **Entropia da atenção regularizada**: Previne atenção colapsar em um único patch (entropia → 0) ou ser uniforme (entropia → log(N)). Target=1.5 nats ≈ 4-5 patches ativos por head.

4. **Supervisão dupla (contrastivo + vg)**: O `loss_vg` supervisiona os patches *antes* do cross-attention, forçando a projeção LoRA a ser semanticamente informativa independente da query textual.

5. **Buffer rotativo com prefetch**: Solução pragmática para treinar com HDs mecânicos lentos sem bottleneck no I/O — mantém 1-9 shards na RAM e pré-carrega o próximo em background.
