# SceneEmbed-Up

**Alinhamento Multimodal de Alta Resolução com Visualização via Scene Graphs**

Pipeline de pesquisa que combina **DINOv3** (encoder visual + upsampling HR) + **Qwen3-Embedding-8B** (encoder textual) + **LoRA Cross-Attention Aligner** para aprender um espaço de embeddings compartilhado sobre dados COYO. Scene Graphs são gerados pelo modelo pré-treinado **RelTR** para visualização e explicabilidade dos resultados de retrieval.

---

## Visão Geral da Arquitetura

```
QUERY TEXTUAL                    IMAGENS COYO (indexadas)
      │                                    │
      ▼                                    ▼
QwenSceneEmbedder              DinoSceneEncoder
[1, 1, 4096]                   hr_features [B, 768, 32, 32]
      │                                    │
      │                         reshape + permute
      │                         [B, 1024, 768]
      │                                    │
      └──────────┬────────────────────────┘
                 ▼
     LoRACrossAttentionAligner
     Q=text, K=V=visual (LoRA rank=64)
     attn_output [B, 1, 4096]
                 │
          L2 normalize
                 │
         Similaridade Cosseno
                 │
         Top-K Imagens Recuperadas
                 │
                 ▼
            RelTR (pré-treinado VG-150)
            end-to-end, sem fine-tuning
                 │
         Scene Graph (triplas + bboxes)
                 │
                 ▼
         Visualização (imagem + dígrafo)
         results/sg_viz/
```

---

## Pipeline de Execução

### Fase 0 — Dados COYO

```bash
python data/get_metadados_coyo.py
python data/get_small_sample_coyo.py
python data/extract_files_coyo.py
python embeddings/generate_shards.py          # → shards H5 (visual+text embeddings)
```

### Fase 1 — Aligner (COYO)

```bash
# Opção A: shards pré-computados (recomendado, VRAM ~8 GB)
python train_aligner_with_h5.py

# Opção B: imagens direto (requer VRAM > 16 GB)
python train_aligner_with_Images.py

# → checkpoints/best_aligner.pth

# Avaliação Recall@K
python eval_retrieval_with_h5.py
```

### Fase 2 — Setup RelTR

```bash
git clone https://github.com/yrcong/RelTR.git reltr_repo
mkdir -p checkpoints/reltr
# Baixar pesos VG-150:
# Salvar em: checkpoints/reltr/reltr_vg.pth
```

### Fase 3 — Retrieval + Scene Graph

```bash
python eval_retrieval_sg.py --query "a dog on a skateboard" --top_k 5
python eval_retrieval_sg.py --query "person riding bike" --top_k 3 --threshold 0.2

# Opções completas:
#   --image_dir     diretório com imagens COYO (default: data/coyo_sample)
#   --top_k         número de imagens a recuperar (default: 5)
#   --threshold     score mínimo RelTR para incluir tripla (default: 0.3)
#   --num_queries   número de triplas RelTR por imagem (default: 20)
#   --aligner_ckpt  checkpoint do aligner (default: checkpoints/best_aligner.pth)
#   --reltr_ckpt    checkpoint RelTR (default: checkpoints/reltr/reltr_vg.pth)
#   --output_dir    diretório de saída (default: results/sg_viz)
```

### Dependências entre fases

```
COYO data → generate_shards → train_aligner → eval_retrieval_sg
                                                      │
                               RelTR pré-treinado ───┘
```

---

## Checkpoints

| Arquivo | Produzido por | Usado em |
|---|---|---|
| `checkpoints/best_aligner.pth` | `train_aligner_with_h5.py` | `eval_retrieval_*.py`, `eval_retrieval_sg.py` |
| `checkpoints/reltr/reltr_vg.pth` | download (pré-treinado) | `eval_retrieval_sg.py` |

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
| `hr_features` | `[1, 768, H_hr, W_hr]` | Features espaciais HR |

---

### QwenSceneEmbedder `models/encoders/qwen3_extrator.py`

| Parâmetro | Valor |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-8B` |
| `dtype` | `bfloat16` |

**`embed_components(batch_texts)` → `[B, N_texts, 4096]`**

---

### LoRACrossAttentionAligner `models/aligners/lora_cross_attention.py`

| Parâmetro | Valor |
|---|---|
| `visual_dim` | 768 |
| `text_dim` | 4096 |
| `rank` | 64 |
| `num_heads` | 8 |

```
hr_patches    [B, 1024, 768]
text_queries  [B, 1, 4096]

Projeção visual com LoRA:
  base_v = visual_proj(hr_patches)              [B, 1024, 4096]
  lora_v = (hr_patches @ lora_A) @ lora_B      [B, 1024, 4096]
  v_features = base_v + scaling * lora_v        [B, 1024, 4096]

Cross-Attention (Q=text, K=V=visual):
  attn_output [B, 1, 4096]

Loss: contrastive_loss (InfoNCE, τ=0.05) + 0.05 * entropy_reg
```

---

### RelTRWrapper `models/SG/reltr_wrapper.py`

Wrapper sobre o [RelTR](https://github.com/yrcong/RelTR) pré-treinado em VG-150. Recebe uma imagem PIL e retorna um `SceneGraph` com triplas `(sujeito, predicado, objeto)` e bounding boxes normalizados.

| Parâmetro | Valor padrão |
|---|---|
| `num_queries` | 20 triplas por imagem |
| `threshold` | 0.3 (score mínimo) |
| Vocabulário | 150 entidades + 50 predicados VG-150 |

```python
from models.SG.reltr_wrapper import RelTRWrapper

reltr = RelTRWrapper(checkpoint="checkpoints/reltr/reltr_vg.pth")
sg = reltr.predict(image)          # image: PIL.Image
print(sg.triples[0])               # SceneGraphTriple(subject, predicate, object, ...)
G = sg.to_networkx()               # nx.DiGraph
```

---

### Módulos de Upsampling `models/ups/hr_conversions.py`

| Módulo | Método |
|---|---|
| `AnyUpModel` | PyTorch Hub `wimmerth/anyup` |
| `JBUStack` | 4 estágios JBU progressivos (×2 cada = ×16 total) |

---

## Visualização

`eval_retrieval_sg.py` gera uma figura por imagem recuperada com duas colunas:

- **Esquerda**: imagem original com bounding boxes das entidades detectadas, coloridos por confiança (vermelho→verde)
- **Direita**: dígrafo do scene graph com nós rotulados; entidades presentes na query são destacadas em laranja

Saídas salvas em `results/sg_viz/<query>/rank01_<img>.png`.

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
│   └── SG/
│       └── reltr_wrapper.py          # RelTRWrapper (pré-treinado, sem fine-tuning)
├── data/
│   ├── data_utils_pytorch.py         # Datasets/DataLoaders COYO + shards H5
│   ├── get_metadados_coyo.py
│   ├── get_small_sample_coyo.py
│   └── extract_files_coyo.py
├── embeddings/
│   └── generate_shards.py            # Embeddings COYO → H5
├── viz/
│   └── scene_graph_viz.py            # Visualização retrieval + scene graph
├── utils/
│   ├── checkpoint.py
│   ├── early_stopping.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   └── metrics.py                    # salvar_recall_results
├── train_aligner_with_h5.py          # Fase 1 — Aligner (shards H5)
├── train_aligner_with_Images.py      # Fase 1 — Aligner (imagens direto)
├── eval_retrieval_with_h5.py         # Avaliação Recall@K (shards H5)
├── eval_retrieval_with_images.py     # Avaliação Recall@K (imagens)
└── eval_retrieval_sg.py              # Retrieval + Scene Graph + Visualização
```

---

## Dados

| Dataset | Uso |
|---|---|
| COYO-700M (~15M imgs) | Treino do Aligner |
| Visual Genome (VG-150) | Pesos pré-treinados do RelTR (download direto) |

**Shards H5 — Aligner (COYO):**

| Campo | Shape | Dtype |
|---|---|---|
| `visual_feats` | `[N, 1024, 768]` | float16 |
| `text_feats` | `[N, 1, 4096]` | float16 |
| `visual_global` | `[N, 768]` | float16 |

~7 GB por shard de 5k amostras (gzip).

---

## Métricas

| Tipo | Métrica | Descrição |
|---|---|---|
| Retrieval | `I2T/T2I Recall@K` | Top-K bidirecional image↔text |
| Retrieval | `Mean Recall@K` | Média I2T + T2I |

---

## Tech Stack

- Python 3.10+, PyTorch (CUDA 12.1), bfloat16
- HuggingFace Transformers
- H5PY (shards de embeddings)
- NetworkX + Matplotlib (visualização de grafos)
- TensorBoard (logging de treino)
- RelTR (repositório externo em `reltr_repo/`)

---

## Referências

- [DINOv3](https://github.com/facebookresearch/dinov3) — Meta AI
- [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B) — Alibaba Cloud
- [RelTR](https://github.com/yrcong/RelTR) — Cong et al., TPAMI 2023
- [AnyUp](https://github.com/wimmerth/anyup) — Wimmer et al.
- [FeatUp](https://github.com/mhamilton723/FeatUp) — Hamilton et al.
- [COYO-700M](https://github.com/kakaobrain/coyo-dataset) — Kakao Brain
- [Visual Genome](https://homes.cs.washington.edu/~ranjay/visualgenome/) — Krishna et al.
