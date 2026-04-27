# SceneEmbed-Up

**Alinhamento Multimodal de Alta Resolução para Geração de Scene Graphs**

Pipeline de pesquisa que combina **DINOv3** (encoder visual + upsampling HR) + **Qwen3-Embedding-8B** (encoder textual) + **LoRA Cross-Attention Aligner** para aprender um espaço de embeddings compartilhado e gerar Scene Graphs semânticos enriquecidos com Knowledge Graphs.

---

## Visão Geral da Arquitetura

```
RAW IMAGE (PIL)
         │
         ├───────────────────────────────────────────────┐
         ▼                                               ▼
 DinoSceneEncoder                                 MdetrDetector
 cls_token     [1, 768]                   MDETR pré-treinado no VG
 hr_features   [1, 768, H_hr, W_hr]      text query: "person . sky . ..."
         │                               pred_logits [1, Q, seq_len]
         │  adaptive_avg_pool2d(32×32)   pred_boxes  [1, Q, 4]
         ▼                                               │
   [B, 768, 32, 32]                                      ▼
         │                                       list[Detection]
         │  reshape + permute                    .bbox_xyxy [4] pixels
         ▼                                       .bbox_grid [4] (0..32)
   [B, 32, 32, 768]                              .label, .score
         │                                               │
         │        QwenSceneEmbedder                      │
         │        text_queries [B, 1, 4096]              │
         │                │                              │
         ▼                ▼                              │
  LoRACrossAttentionAligner                              │
   v_features   [B, 1024, 4096]  ──────────────────────┐│
   attn_output  [B, 1, 4096]                           ││
   attn_weights [B, 8, 1, 1024]                        ▼▼
   (usado em retrieval)                 visual_grid [1, 32, 32, 768]
                                                       │
                        ┌──────────────────────────────┤
                        ▼                              ▼
              AttributeClassifier             RelationPredictor
              ROI mean-pool por bbox          P = N(N-1) pares
              [M, 768] → [M, 4096]           sub/obj/union → [P, 768]
              AttributeHead [M, 200]         project → [P, 4096]
              sigmoid + top-5 attrs          RelationHead → [P, 50]
                        │                              │
                        └────────────┬─────────────────┘
                                     ▼
                                Scene Graph
                           nodes: objetos + atributos
                           edges: (sub, pred, obj, conf)
                                     │
                                     ▼
                         KnowledgeGraphGenerator
                         is_a, part_of, made_of,
                         used_for, has_property
```

---

## Modelos

### DinoSceneEncoder `models/encoders/dinov3_extrator.py`

| Parâmetro | Valor |
|---|---|
| `model_name` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `upsampler` | `"anyup"` ou `"featup"` |

**`extract_features(img_tensor)` → `(cls_token, hr_features)`**

| Saída | Shape | Descrição |
|---|---|---|
| `cls_token` | `[1, 768]` | Descritor global da imagem |
| `hr_features` | `[1, 768, H_hr, W_hr]` (anyup) | Features espaciais HR |

Pipeline interno: DINOv3 → `[B, N_total, 768]` → extrai patches espaciais → reshape `[B, 768, h, w]` → upsample → pool `[B, 768, 32, 32]`.

---

### QwenSceneEmbedder `models/encoders/qwen3_extrator.py`

| Parâmetro | Valor |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-8B` |
| `dtype` | `bfloat16` |

**`embed_components(batch_texts)` → `[B, N_texts, 4096]`**

Adiciona prefix de instrução a cada texto, tokeniza, forward, mean pooling sobre tokens, L2 normalize.

---

### LoRACrossAttentionAligner `models/aligners/lora_cross_attention.py`

| Parâmetro | Valor |
|---|---|
| `visual_dim` | 768 |
| `text_dim` | 4096 |
| `rank` | 64 |
| `num_heads` | 8 |

**Forward:**

```
Entradas:
  hr_patches    [B, 1024, 768]   (grid 32×32 achatado)
  text_queries  [B, 1, 4096]

Projeção visual com LoRA:
  base_v = visual_proj(hr_patches)              [B, 1024, 4096]
  lora_v = (hr_patches @ lora_A) @ lora_B      [B, 1024, 4096]
  v_features = base_v + scaling * lora_v        [B, 1024, 4096]

Cross-Attention (Q=text, K=V=visual):
  attn_output, attn_weights = cross_attn(
      query = text_queries,    [B, 1, 4096]
      key   = v_features,      [B, 1024, 4096]
      value = v_features       [B, 1024, 4096]
  )

Saídas:
  attn_output   [B, 1, 4096]
  attn_weights  [B, 8, 1, 1024]
  v_features    [B, 1024, 4096]
```

**Loss de treino:**
```
loss = contrastive_loss + 0.05 * entropy_reg
       InfoNCE simétrico (τ=0.05)   (mean_entropy − 1.5)²
```

`loss_vg` é apenas monitoramento no TensorBoard — não participa do backward.

---

### MdetrDetector `models/detectors/mdetr_detector.py`

Detector open-vocabulary pré-treinado no Visual Genome. Não requer fine-tuning.

| Parâmetro | Valor |
|---|---|
| checkpoint | `ashkamath/mdetr-resnet-50` (HuggingFace Hub) |
| `score_threshold` | 0.5 |
| text query | `"person . sky . building . ..."` (150 classes VG-150) |

**`detect(image)` → `list[Detection]`**

```python
@dataclass
class Detection:
    bbox_xyxy:  Tensor   # [4] coordenadas em pixels
    bbox_grid:  Tensor   # [4] coordenadas em grid (0..32)
    label:      str
    label_idx:  int
    score:      float
```

Pipeline: `pred_logits [Q, seq_len]` → sigmoid → max sobre tokens do texto → mapeamento token→classe via `offset_mapping` → filtro por score → cxcywh (norm) → xyxy (pixels) → escala grid.

---

### AttributeHead `models/SG/attribute_head.py`

Classificação multi-label de atributos sobre features alinhadas.

| Parâmetro | Valor |
|---|---|
| `feat_dim` | 4096 |
| `vocab_size` | 200 |
| `hidden` | 1024 |

```
[M, 4096] → LayerNorm → Linear(4096→1024) → GELU → Dropout(0.1) → Linear(1024→200) → [M, 200]
```

---

### AttributeClassifier `models/SG/attribute_classifier.py`

Orquestra predição de atributos sobre detecções usando o aligner congelado.

```
Entradas: detections + visual_grid [1, 32, 32, 768]

  ROI mean-pool por bbox_grid  →  [M, 768]
  aligner.visual_proj          →  [M, 4096]
  AttributeHead                →  [M, 200] logits
  sigmoid(threshold=0.3) + top-5

Saída: Detection.attributes = [{"name": str, "score": float}, ...]
```

---

### RelationHead `models/SG/relation_head.py`

Classificação de predicado para pares de objetos (direcional).

| Parâmetro | Valor |
|---|---|
| `text_dim` | 4096 |
| `vocab_size` | 50 |
| `proj_dim` | 512 |
| `hidden` | 1024 |
| `use_ctx` | True |

```
Entradas: sub_feat [P, 4096], obj_feat [P, 4096], union_feat [P, 4096]

  s = Linear(4096→512)(sub_feat)
  o = Linear(4096→512)(obj_feat)
  u = Linear(4096→512)(union_feat)

  concat [s, o, s−o, u]  →  [P, 2048]
  LayerNorm → Linear(2048→1024) → GELU → Dropout → Linear(1024→50)
  →  [P, 50] logits
```

A diferença `s − o` garante assimetria: (A→B) ≠ (B→A).

---

### RelationPredictor `models/SG/relation_predictor.py`

Prediz relações entre todos os pares de objetos detectados.

```
N objetos → P = N(N-1) pares ordenados

  ROI mean-pool: sub, obj, union  →  [P, 768] cada
  aligner.visual_proj             →  [P, 4096] cada
  RelationHead                    →  [P, 50] logits
  softmax + top-1 por par

Saída: [{"source": str, "target": str, "relation": str, "confidence": float}]
```

---

### KnowledgeGraphGenerator `models/SG/knowledge.py`

Expande o scene graph com fatos taxonômicos via cache ou Qwen3 LLM.

**Relações:** `is_a`, `part_of`, `made_of`, `used_for`, `has_property`

```python
{
  "entities":      [str, ...],
  "factual_edges": [{"sub": str, "rel": str, "obj": str}, ...]
}
```

---

### Módulos de Upsampling `models/ups/hr_conversions.py`

| Módulo | Método |
|---|---|
| `AnyUpModel` | PyTorch Hub `wimmerth/anyup` |
| `JBUStack` | 4 estágios JBU progressivos (×2 cada = ×16 total) |

---

## Estrutura do Projeto

```
SceneEmbed-Up/
├── models/
│   ├── encoders/
│   │   ├── dinov3_extrator.py        # DinoSceneEncoder
│   │   └── qwen3_extrator.py         # QwenSceneEmbedder
│   ├── aligners/
│   │   └── lora_cross_attention.py   # LoRACrossAttentionAligner
│   ├── ups/
│   │   └── hr_conversions.py         # AnyUpModel, JBUStack
│   ├── detectors/
│   │   └── mdetr_detector.py         # MdetrDetector (pré-treinado VG)
│   └── SG/
│       ├── attribute_head.py         # AttributeHead (200 classes)
│       ├── attribute_classifier.py   # AttributeClassifier
│       ├── relation_head.py          # RelationHead (50 predicados)
│       ├── relation_predictor.py     # RelationPredictor
│       └── knowledge.py             # KnowledgeGraphGenerator
├── data/
│   ├── data_utils_pytorch.py         # Datasets/DataLoaders COYO + shards H5
│   └── vg_dataset.py                 # Visual Genome VG-150 (vocab, splits, datasets)
├── embeddings/
│   ├── generate_shards.py            # Exportação de embeddings COYO → H5
│   └── generate_vg_attr_shards.py    # Exportação de features VG por objeto → H5
├── train_aligner_with_h5.py          # Treino do Aligner (shards H5)
├── train_aligner_with_Images.py      # Treino do Aligner (imagens direto)
├── train_attribute_head.py           # Treino da AttributeHead (shards H5 VG)
├── train_relation_head.py            # Treino da RelationHead (GT bboxes VG)
├── eval_retrieval_with_h5.py         # Recall@K bidirecional (shards H5)
├── eval_retrieval_with_images.py     # Recall@K bidirecional (imagens)
└── eval_sg_vg.py                     # SGGen Recall@K / mR@K no VG
```

---

## Treinamento

### Ordem recomendada

**1. Aligner** — treinado em COYO-700M com InfoNCE + entropy_reg:

```bash
python train_aligner_with_h5.py          # recomendado (shards pré-computados)
python train_aligner_with_Images.py      # requer VRAM > 16 GB
```

**2. Extração de features VG** — DINO congelado, ROI-pool por objeto:

```bash
python embeddings/generate_vg_attr_shards.py   # obj_feats [N,768] + attr_labels [N,200]
```

**3. Heads VG-150** — Aligner congelado:

```bash
python train_attribute_head.py           # AttributeHead, loss: BCE multi-label
python train_relation_head.py            # RelationHead, loss: CE multi-class
```

> O detector **MdetrDetector** não requer treino — usa checkpoint `ashkamath/mdetr-resnet-50` pré-treinado no Visual Genome diretamente do HuggingFace Hub.

---

## Dados

| Dataset | Uso |
|---|---|
| COYO-700M (~15M imgs) | Treino do Aligner |
| Visual Genome (VG-150) | Extração de features, AttributeHead, RelationHead |

**Shards H5 — Aligner (COYO):**

| Campo | Shape | Dtype |
|---|---|---|
| `visual_feats` | `[N, 1024, 768]` | float16 |
| `text_feats` | `[N, 1, 4096]` | float16 |
| `visual_global` | `[N, 768]` | float16 |

~7 GB por shard de 5k amostras (gzip).

**Shards H5 — AttributeHead (VG):**

| Campo | Shape | Dtype |
|---|---|---|
| `obj_feats` | `[N, 768]` | float16 |
| `attr_labels` | `[N, 200]` | float16 |

~400–600 MB total (todos os objetos VG-150 anotados com atributos).

---

## Métricas

| Tipo | Métrica | Descrição |
|---|---|---|
| Retrieval | `I2T/T2I Recall@K` | Top-K bidirecional |
| SGG | `R@20/50/100` | Recall sobre tripletas (sub, pred, obj) |
| SGG | `mR@20/50/100` | Mean Recall por classe de predicado |

---

## Tech Stack

- Python 3.10+, PyTorch (CUDA 12.1), bfloat16
- HuggingFace Transformers
- H5PY (shards de embeddings)
- NetworkX (grafos), TensorBoard (logging)

---

## Referências

- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — Alibaba Cloud
- [MDETR](https://github.com/ashkamath/mdetr) — Kamath et al., ICCV 2021
- [AnyUp](https://github.com/wimmerth/anyup) — Wimmer et al.
- [FeatUp](https://github.com/mhamilton723/FeatUp) — Hamilton et al.
- [COYO-700M](https://github.com/kakaobrain/coyo-dataset) — Kakao Brain
- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) — Krishna et al.
- Xu et al., *Scene Graph Generation by Iterative Message Passing*, CVPR 2017
- Tang et al., *Unbiased Scene Graph Generation from Biased Training*, CVPR 2020
