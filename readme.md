# SceneEmbed-Up

**Alinhamento Multimodal com Visualização e Re-Ranking via Scene Graphs**

Pipeline de pesquisa que combina **DINOv3** (encoder visual, com ou sem upsampling) + **Qwen3-Embedding-8B** (encoder textual) + **LoRA Cross-Attention Aligner** para retrieval texto↔imagem em COYO. Scene Graphs gerados pelo modelo pré-treinado **RelTR** servem tanto para **visualização explicativa** quanto para **re-ranking semântico** dos resultados.

Suporta duas variantes do encoder visual:
- **Com AnyUp** — patches HR upsampled `[1024, 768]` (mais ricos espacialmente)
- **Sem upsampler** — patches LR nativos `[196, 768]` (4× mais leve, melhor para retrieval global)

---

## Visão Geral da Arquitetura

```
QUERY TEXTUAL                  IMAGENS COYO (indexadas)
      │                                  │
      ▼                                  ▼
QwenSceneEmbedder            DinoSceneEncoder
[1, 1, 4096]                 hr_features [B, 768, H, W]
                                     │
                              reshape + permute
                              [B, P, 768]
                                     │
                                     ▼
                  LoRACrossAttentionAligner
                  Q=text, K=V=visual (LoRA rank=64)
                  attn_output [B, 1, 4096]
                                     │
                              L2 normalize + cosine
                                     │
                              Top-K candidatos
                                     │
                                     ▼
                            RelTR (VG-150)
                            scene graphs por imagem
                            ┌────────┴────────┐
                            ▼                 ▼
                  SemanticReRanker      Visualização
                  α·dense + (1-α)·sg    img + scene graph
                  reordena top-K        results/sg_viz_*/
```

`P = 196` (sem upsampler) ou `P = 1024` (com AnyUp).

---

## Pipeline de Execução

### Fase 0 — Dados COYO

```bash
python data/get_metadados_coyo.py
python data/get_small_sample_coyo.py
python data/extract_files_coyo.py

# Shards H5 (escolher UMA das duas variantes):
python embeddings/generate_shards.py          # com AnyUp     → visual_feats [N,1024,768]
python embeddings/generate_shards_no_up.py    # sem upsampler → visual_feats [N, 196,768]
                                              #                 + image_paths salvos
```

### Fase 1 — Aligner (COYO)

```bash
# Variante A — com AnyUp (clássico)
python train_aligner_with_h5.py        # shards H5 pré-computados, VRAM ~8 GB
python train_aligner_with_Images.py    # alternativa: imagens direto, VRAM > 16 GB

# Variante B — sem upsampler (recomendado para retrieval)
python train_aligner_no_up.py          # shards H5 [N, 196, 768]

# Variante C — com AnyUp + seleção adaptativa de tokens (ATS, aprendido)
python train_aligner_ats_h5.py         # shards H5 [N,1024,768] → seleciona K=196 patches

# Variante D — com AnyUp + seleção via PCA (critério de variância, clássico)
python train_aligner_pca_h5.py         # shards H5 [N,1024,768] → seleciona K=196 patches

# → checkpoints/best_aligner.pth          (com AnyUp)
# → checkpoints/best_aligner_no_up.pth    (sem upsampler)
# → checkpoints/best_ats_aligner.pth      (com AnyUp + ATS)
# → checkpoints/best_pca_aligner.pth      (com AnyUp + PCA)

# Avaliação Recall@K
python eval_retrieval_with_h5.py       # ajustar paths + ckpt no script
```

### Fase 2 — Setup RelTR

```bash
git clone https://github.com/yrcong/RelTR.git reltr_repo
mkdir -p checkpoints/reltr
# Baixar pesos VG-150 e salvar em: checkpoints/reltr/reltr_vg.pth
```

### Fase 3 — Retrieval + Scene Graph + (opcional) Re-Ranking

**Variante A — Aligner treinado COM AnyUp:**
```bash
python eval_retrieval_sg.py --query "a dog on a skateboard" --top_k 5

# Com re-ranking semântico via scene graphs
python eval_retrieval_sg.py --query "person riding bike" --top_k 10 \
    --rerank --rerank_alpha 0.5 --rerank_pool topn_mean --rerank_topn 3
```

**Variante B — Aligner treinado SEM upsampler:**
```bash
python eval_retrieval_sg_no_up.py --query "two men standing together" --top_k 5

# Com re-ranking
python eval_retrieval_sg_no_up.py --query "a cat on a chair" --top_k 10 --rerank
```

**Opções principais:**

| Flag | Default | Descrição |
|---|---|---|
| `--query` | (obrigatório) | Texto da consulta |
| `--image_dir` | `F:/COYO/coyo/extracted/00000` | Diretório com imagens COYO a indexar |
| `--top_k` | 5 | Quantos candidatos retornar (também é o N do rerank) |
| `--threshold` | 0.3 | Score mínimo RelTR para incluir tripla no grafo |
| `--num_queries` | 20 | Máximo de triplas por imagem |
| `--aligner_ckpt` | `checkpoints/best_aligner.pth` (`_no_up`) | Checkpoint do aligner |
| `--reltr_ckpt` | `checkpoints/reltr/reltr_vg.pth` | Checkpoint RelTR |
| `--output_dir` | `results/sg_viz_anyup` ou `results/sg_viz_no_up` | Pasta de saída separada por variante |
| `--rerank` | (flag) | Ativa re-ranking semântico |
| `--rerank_alpha` | 0.5 | `final = α·dense + (1-α)·sg` |
| `--rerank_pool` | `topn_mean` | `max` ou `topn_mean` para agregar sims query↔tripla |
| `--rerank_topn` | 3 | N para `topn_mean` |
| `--explain_top_k` | 3 | Triplas explicativas exibidas por imagem |

**Saídas geradas em `<output_dir>/<query>/`:**
- `rank{NN}_<img>.png` — visualizações pré-rerank (sempre salvas)
- `rerank{NN}_<img>.png` — visualizações pós-rerank (se `--rerank`)

### Dependências entre fases

```
COYO → generate_shards{_no_up}? → train_aligner{_no_up}? → eval_retrieval_sg{_no_up}?
                                                                    │
                                                  RelTR pré-treinado ┘
```

---

## Checkpoints

| Arquivo | Produzido por | Usado em |
|---|---|---|
| `checkpoints/best_aligner.pth` | `train_aligner_with_h5.py` | `eval_retrieval_with_h5.py`, `eval_retrieval_sg.py` |
| `checkpoints/best_aligner_no_up.pth` | `train_aligner_no_up.py` | `eval_retrieval_sg_no_up.py` |
| `checkpoints/best_ats_aligner.pth` | `train_aligner_ats_h5.py` | avaliação da variante ATS (sampler + aligner) |
| `checkpoints/best_pca_aligner.pth` | `train_aligner_pca_h5.py` | avaliação da variante PCA (aligner) |
| `checkpoints/reltr/reltr_vg.pth` | download (pré-treinado) | ambos os `eval_retrieval_sg*.py` |

---

## Modelos

### DinoSceneEncoder `models/encoders/dinov3_extrator.py`

| Parâmetro | Valor |
|---|---|
| `model_name` | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| `upsampler` | `"anyup"`, `"featup"` ou `"none"` |

**Métodos:**

| Método | Saída | Quando usar |
|---|---|---|
| `extract_features(img)` | `(cls [1,768], hr_features [1,768,H,W])` | Pipeline com AnyUp/FeatUp |
| `extract_patches_seq(img)` | `(cls [1,768], patches [1,N,768])` | Pipeline sem upsampler (N=196 para 224²/patch 16) |

---

### QwenSceneEmbedder `models/encoders/qwen3_extrator.py`

| Parâmetro | Valor |
|---|---|
| `model_id` | `Qwen/Qwen3-Embedding-8B` |
| `dtype` | `bfloat16` |

`embed_components(batch_texts, normalize=False) → [B, N_texts, 4096]`

---

### LoRACrossAttentionAligner `models/aligners/lora_cross_attention.py`

| Parâmetro | Valor |
|---|---|
| `visual_dim` | 768 |
| `text_dim` | 4096 |
| `rank` | 64 |
| `num_heads` | 8 |

```
hr_patches    [B, P, 768]      (P = 196 ou 1024)
text_queries  [B, 1, 4096]

Projeção visual com LoRA:
  base_v = visual_proj(hr_patches)              [B, P, 4096]
  lora_v = (hr_patches @ lora_A) @ lora_B       [B, P, 4096]
  v_features = base_v + scaling * lora_v        [B, P, 4096]

Cross-Attention (Q=text, K=V=visual):
  attn_output [B, 1, 4096]

Loss: contrastive_loss (InfoNCE, τ=0.05) + 0.05 * entropy_reg
```

---

### RelTRWrapper `models/SG/reltr_wrapper.py`

Wrapper sobre o [RelTR](https://github.com/yrcong/RelTR) pré-treinado em VG-150. Recebe uma imagem PIL e retorna um `SceneGraph` com triplas `(sujeito, predicado, objeto)` + bounding boxes normalizados.

| Parâmetro | Default |
|---|---|
| `num_queries` | 20 triplas por imagem |
| `threshold` | 0.3 (score mínimo) |
| Vocabulário | 150 entidades + 50 predicados VG-150 |

```python
from models.SG.reltr_wrapper import RelTRWrapper

reltr = RelTRWrapper(checkpoint="checkpoints/reltr/reltr_vg.pth")
sg = reltr.predict(image)
G = sg.to_networkx()
```

---

### SemanticReRanker `models/rerank/semantic_rerank.py`

**Two-stage retrieval explicável.** Stage 1 (DINO+Aligner) traz top-K candidatos densos; stage 2 reordena esse mesmo conjunto usando similaridade entre a query e as triplas dos scene graphs gerados pelo RelTR.

```
para cada candidato (path, dense_score, scene_graph):
    triples_emb = Qwen("subject predicate object")        # [T, 4096], normalizado
    sims = triples_emb @ query_emb                         # [T]
    score_sg = pool(sims)                                  # max ou topn_mean

dense_norm = minmax(dense_scores)
sg_norm    = minmax(sg_scores)
score_final = α · dense_norm + (1-α) · sg_norm
```

**Importante:** o reranker só reordena os top-K já recuperados pelo stage 1 — não busca fora desse conjunto. Para considerar mais candidatos, aumente `--top_k`.

Cada `ReRankResult` guarda também as triplas mais alinhadas com a query (`explain_top_k`) para visualização explicativa.

---

### Módulos de Upsampling `models/ups/hr_conversions.py`

| Módulo | Método |
|---|---|
| `AnyUpModel` | PyTorch Hub `wimmerth/anyup` |
| `JBUStack` | 4 estágios JBU progressivos (×2 cada = ×16 total) |
| `upsampler="none"` | Sem upsampling (patches LR nativos do DINO) |

---

## Visualização

`eval_retrieval_sg*.py` salva uma figura por imagem recuperada:

- **Pré-rerank** (`rank{NN}_<img>.png`) — duas colunas: imagem com bboxes coloridos por confiança RelTR | dígrafo do scene graph com nós destacados quando aparecem na query
- **Pós-rerank** (`rerank{NN}_<img>.png`, se `--rerank`) — três colunas: imagem com bboxes | scene graph com triplas top-K destacadas | painel textual com scores antes/depois e as triplas que mais alinharam com a query

---

## Seleção Adaptativa de Patches (AnyUp 1024 → 196)

A análise de Dimensão Intrínseca (`analyze_id_local.py`) mostra que os 1024
patches AnyUp têm ID ≈ 1.88 (vs ID ≈ 5.0 para os 196 patches sem upsampler) —
isto é, os patches upsampled são altamente redundantes (manifold quasi-1D),
o que dificulta o cross-attention do aligner (entropia de atenção fica alta
e o treino estagna mais cedo). As duas variantes abaixo reduzem 1024 → K=196
patches por relevância **antes** do `LoRACrossAttentionAligner`, igualando o
número de tokens ao da variante sem upsampler.

| Script | Método | Aprendido? | Critério de seleção |
|---|---|---|---|
| [train_aligner_ats_h5.py](train_aligner_ats_h5.py) | `AdaptiveTokenSampler` (ATS) | Sim | score = content_proj(patch) + dot(patch, text_gate(query)); BCE straight-through |
| [train_aligner_pca_h5.py](train_aligner_pca_h5.py) | `PCAVarianceSampler` | Não | PCA por imagem (`torch.pca_lowrank`); score = norma da projeção nos top-64 componentes principais |

- **ATS** (`train_aligner_ats_h5.py`) — inspirado em Fayyaz et al. (2022), ECCV.
  Scorer aprendido (text-guided + content-based), treinado via loss adicional
  `Loss/ats` (BCE-with-logits straight-through). Checkpoint salva
  `sampler` + `aligner` juntos (`nn.ModuleDict`).
- **PCA** (`train_aligner_pca_h5.py`) — baseline clássico, determinístico,
  sem parâmetros extras. Seleciona os patches que mais contribuem para as
  direções de maior variância da imagem (PCA projection score).

Ambos reutilizam os shards `train_anyup`/`val_anyup` já gerados, mantêm a
mesma loss contrastiva + `entropy_reg` do `train_aligner_with_h5.py`, e
suportam resume via `RESUME_CHECKPOINT`/`START_EPOCH` no `__main__`.

---

## Análises Geométricas dos Embeddings

Scripts em `embeddings/` para comparar a estrutura geométrica dos embeddings com vs sem upsampling:

| Script | Análise | Pergunta que responde |
|---|---|---|
| [analyze_rsa_global.py](embeddings/analyze_rsa_global.py) | RSA sobre `visual_global` (CLS token) | A geometria GLOBAL do dataset muda com AnyUp? |
| [analyze_id_local.py](embeddings/analyze_id_local.py) | Dimensão Intrínseca TwoNN por imagem | A complexidade LOCAL dos patches muda? |
| [analyze_cka_cross.py](embeddings/analyze_cka_cross.py) | Centered Kernel Alignment visual↔texto | O alinhamento bruto visual↔texto muda? |

```bash
python embeddings/analyze_rsa_global.py --split test --n_samples 5000
python embeddings/analyze_id_local.py   --split test --n_images 500
python embeddings/analyze_cka_cross.py  --split test --n_samples 5000 --kernel linear
```

Saídas em `results/{rsa_global,id_local,cka}/` com JSON + PNGs comparativos.

---

## Estrutura do Projeto

```
SceneEmbed-Up/
├── models/
│   ├── encoders/
│   │   ├── dinov3_extrator.py          # DinoSceneEncoder (upsampler='anyup'|'featup'|'none')
│   │   └── qwen3_extrator.py           # QwenSceneEmbedder
│   ├── aligners/
│   │   └── lora_cross_attention.py     # LoRACrossAttentionAligner
│   ├── ups/
│   │   └── hr_conversions.py           # AnyUpModel, JBUStack
│   ├── rerank/
│   │   └── semantic_rerank.py          # SemanticReRanker (two-stage)
│   └── SG/
│       └── reltr_wrapper.py            # RelTRWrapper (pré-treinado, sem fine-tuning)
├── data/
│   ├── data_utils_pytorch.py           # Datasets/DataLoaders COYO + shards H5
│   ├── test_shards.py                  # Inspeção e comparação de shards
│   ├── get_metadados_coyo.py
│   ├── get_small_sample_coyo.py
│   └── extract_files_coyo.py
├── embeddings/
│   ├── generate_shards.py              # Shards COYO com AnyUp        [N,1024,768]
│   ├── generate_shards_no_up.py        # Shards COYO sem upsampler    [N, 196,768] + image_paths
│   ├── analyze_rsa_global.py           # RSA global (com vs sem AnyUp)
│   ├── analyze_id_local.py             # Dim. Intrínseca local (TwoNN)
│   └── analyze_cka_cross.py            # CKA visual↔texto
├── viz/
│   └── scene_graph_viz.py              # visualize_retrieval_result + visualize_rerank_result
├── utils/
│   ├── checkpoint.py
│   ├── early_stopping.py
│   ├── io_utils.py
│   ├── logging_utils.py
│   └── metrics.py
├── train_aligner_with_h5.py            # Treino aligner — shards com AnyUp
├── train_aligner_with_Images.py        # Treino aligner — imagens direto
├── train_aligner_no_up.py              # Treino aligner — shards SEM upsampler
├── train_aligner_ats_h5.py             # Treino aligner — AnyUp + ATS (1024→196, aprendido)
├── train_aligner_pca_h5.py             # Treino aligner — AnyUp + PCA (1024→196, clássico)
├── eval_retrieval_with_h5.py           # Recall@K bidirecional sobre shards
├── eval_retrieval_with_images.py       # Recall@K sobre imagens brutas
├── eval_retrieval_sg.py                # Retrieval + SG + rerank   (variante AnyUp)
└── eval_retrieval_sg_no_up.py          # Retrieval + SG + rerank   (variante sem upsampler)
```

---

## Dados

| Dataset | Uso |
|---|---|
| COYO-700M (~15M imgs) | Treino do Aligner |
| Visual Genome (VG-150) | Pesos pré-treinados do RelTR (download direto) |

**Shards H5 — Aligner (COYO):**

| Variante | `visual_feats` | `text_feats` | `visual_global` | `image_paths` | Tamanho/shard (5k) |
|---|---|---|---|---|---|
| `generate_shards.py` (AnyUp) | `[N, 1024, 768]` | `[N, 1, 4096]` | `[N, 768]` | ❌ | ~7 GB (gzip) |
| `generate_shards_no_up.py` | `[N, 196, 768]` | `[N, 1, 4096]` | `[N, 768]` | ✅ utf-8 vlen | ~1.4 GB (gzip) |

A versão sem upsampler é ~4× menor, ~2-3× mais rápida de gerar, e salva os caminhos das imagens originais para usos posteriores (e.g. retrieval com SG sem re-extrair).

---

## Métricas

| Tipo | Métrica | Descrição |
|---|---|---|
| Retrieval | `I2T/T2I Recall@K` | Top-K bidirecional image↔text |
| Retrieval | `Mean Recall@K` | Média I2T + T2I |
| Rerank | `score_dense, score_sg, score_final` | Componentes do score combinado |
| Análise | RSA Pearson/Spearman | Correlação entre RDMs com/sem AnyUp |
| Análise | TwoNN ID | Dimensão intrínseca local por imagem |
| Análise | CKA | Alinhamento entre Gram matrices |

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
- Facco et al., *Estimating the intrinsic dimension of datasets by a minimal neighborhood information*, Scientific Reports 2017
- Fayyaz et al., *Adaptive Token Sampling For Efficient Vision Transformers (ATS)*, ECCV 2022
- Kornblith et al., *Similarity of Neural Network Representations Revisited (CKA)*, ICML 2019
- Kriegeskorte et al., *Representational similarity analysis*, Frontiers in Systems Neuroscience 2008
- Nogueira & Cho, *Passage Re-ranking with BERT*, 2019
