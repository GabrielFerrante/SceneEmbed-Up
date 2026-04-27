# SceneEmbed-Up

**Alinhamento Multimodal de Alta Resolução para Geração de Scene Graphs**

Pipeline de pesquisa que combina **DINOv3** (encoder visual + upsampling HR) + **Qwen3-Embedding-8B** (encoder textual) + **LoRA Cross-Attention Aligner** para aprender um espaço de embeddings compartilhado e gerar Scene Graphs semânticos enriquecidos com Knowledge Graphs.

---

## Visão Geral da Arquitetura

```
RAW IMAGE [3, 640, 640]
         │
         ├───────────────────────────────────────────────┐
         ▼                                               ▼
 DinoSceneEncoder                                  DetrDetector
 cls_token     [1, 768]                         DETR-R50, 150 classes
 hr_features   [1, 768, 686, 960]              logits [1, 300, 151]
         │                                     pred_boxes [1, 300, 4]
         │  adaptive_avg_pool2d(32×32)                   │
         ▼                                               ▼
   [B, 768, 32, 32]                             list[Detection]
         │                                      .bbox_xyxy [4] pixels
         │  reshape                              .bbox_grid [4] (0..32)
         ▼                                      .label, .score
   [B, 1024, 768]                                       │
         │                                              │
         │        QwenSceneEmbedder                     │
         │        text_queries [B, 1, 4096]             │
         │                │                             │
         ▼                ▼                             │
  LoRACrossAttentionAligner                             │
   v_features   [B, 1024, 4096]  ─────────────────────┐│
   attn_output  [B, 1, 4096]                          ││
   attn_weights [B, 8, 1, 1024]                       ▼▼
   (usado em retrieval)                  visual_grid [1, 32, 32, 768]
                                                      │
                         ┌────────────────────────────┤
                         ▼                            ▼
               AttributeClassifier            RelationPredictor
               ROI mean-pool per bbox         P = N(N-1) pares
               [M, 768] → [M, 4096]          sub/obj/union → [P, 768]
               AttributeHead [M, 200]        project → [P, 4096]
               sigmoid + top-5 attrs         RelationHead → [P, 50]
                         │                            │
                         └────────────┬───────────────┘
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
| `hr_features` | `[1, 768, 686, 960]` (anyup) / `[1, 384, 224, 224]` (featup) | Features espaciais HR |

Pipeline interno: DINOv3 → `[B, N_total, 768]` → extrai patches espaciais → reshape `[B, 768, h, w]` → upsample.

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
| `alpha` | 32 (`scaling = alpha/rank`) |

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

### DetrDetector `models/detectors/detr_detector.py`

DETR-R50 fine-tunado para 150 classes do Visual Genome.

| Parâmetro | Valor |
|---|---|
| queries | 300 (padrão DETR) |
| input size | 640×640 |
| `score_threshold` | 0.5 |

**`detect(image)` → `list[Detection]`**

```python
@dataclass
class Detection:
    bbox_xyxy:  Tensor   # [4] coordenadas em pixels (0..640)
    bbox_grid:  Tensor   # [4] coordenadas em grid (0..32)
    label:      str
    label_idx:  int
    score:      float
```

Pipeline: `logits [1, 300, 151]` → softmax → filtro por score → cxcywh (norm) → xyxy (pixels) → escala grid (`× 32/640`).

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
# Saída
{
  "entities":      [str, ...],
  "factual_edges": [{"sub": str, "rel": str, "obj": str}, ...]
}
```

---

### Módulos de Upsampling `models/ups/hr_conversions.py`

| Módulo | Saída | Método |
|---|---|---|
| `AnyUpModel` | `[B, 768, 224, 224]` | PyTorch Hub `wimmerth/anyup` |
| `JBUStack` | `[B, 384, 224, 224]` | 4 estágios JBU progressivos (×2 cada = ×16 total) |

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
│   │   └── detr_detector.py          # DetrDetector
│   └── SG/
│       ├── attribute_head.py         # AttributeHead (200 classes)
│       ├── attribute_classifier.py   # AttributeClassifier
│       ├── relation_head.py          # RelationHead (50 predicados)
│       ├── relation_predictor.py     # RelationPredictor
│       └── knowledge.py             # KnowledgeGraphGenerator
├── data/
│   ├── data_utils_pytorch.py         # Datasets/DataLoaders COYO + shards H5
│   └── vg_detection_dataset.py       # Visual Genome VG-150
├── embeddings/
│   └── generate_shards.py            # Exportação de embeddings H5
├── train_aligner_with_h5.py          # Treino do Aligner (shards H5)
├── train_aligner_with_Images.py      # Treino do Aligner (imagens direto)
├── train_detr_vg150.py               # Fine-tuning do DETR-R50
├── train_attribute_head.py           # Treino da AttributeHead (GT bboxes)
├── train_relation_head.py            # Treino da RelationHead (GT bboxes)
├── eval_retrieval_with_h5.py         # Recall@K bidirecional (shards H5)
├── eval_retrieval_with_images.py     # Recall@K bidirecional (imagens)
└── eval_sg_vg.py                     # SGGen Recall@K / mR@K no VG
```

---

## Treinamento

Dois estágios independentes:

**1. Aligner** — treinado em COYO-700M com InfoNCE + entropy_reg:

```bash
python train_aligner_with_h5.py          # recomendado (shards pré-computados)
python train_aligner_with_Images.py      # requer VRAM > 16 GB
```

**2. Heads VG-150** — Aligner + DINO congelados (setup PredCls-like, bboxes GT):

```bash
python train_detr_vg150.py               # fine-tuning do detector
python train_attribute_head.py           # AttributeHead, loss: BCE multi-label
python train_relation_head.py            # RelationHead, loss: CE multi-class
```

---

## Dados

| Dataset | Uso |
|---|---|
| COYO-700M (~15M imgs) | Treino do Aligner |
| Visual Genome (VG-150) | Treino DETR, AttributeHead, RelationHead |

**Shards H5 (Aligner):**

| Campo | Shape | Dtype |
|---|---|---|
| `visual_feats` | `[N, 1024, 768]` | bfloat16 |
| `text_feats` | `[N, 1, 4096]` | bfloat16 |
| `visual_global` | `[N, 768]` | bfloat16 |

~7 GB por shard de 5k amostras (gzip).

---

## Métricas

| Tipo | Métrica | Descrição |
|---|---|---|
| Retrieval | `I2T/T2I Recall@K` | Top-K bidireccional |
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
- [AnyUp](https://github.com/wimmerth/anyup) — Wimmer et al.
- [FeatUp](https://github.com/mhamilton723/FeatUp) — Hamilton et al.
- [COYO-700M](https://github.com/kakaobrain/coyo-dataset) — Kakao Brain
- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) — Krishna et al.
- Xu et al., *Scene Graph Generation by Iterative Message Passing*, CVPR 2017
- Tang et al., *Unbiased Scene Graph Generation from Biased Training*, CVPR 2020
