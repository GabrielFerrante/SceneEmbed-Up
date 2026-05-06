# SceneEmbed-Up

## Visao Geral
Pipeline de pesquisa para alinhamento multimodal de alta resolucao e geracao de Scene Graphs.
Combina DINOv3 (encoder visual com upsampling HR) + Qwen3-Embedding-8B (encoder textual) + LoRA Cross-Attention Aligner.
Scene Graphs gerados pelo modelo pre-treinado RelTR (end-to-end, VG-150) para visualizacao e explicabilidade do retrieval.

## Tech Stack
- Python 3.10+
- PyTorch (CUDA 12.1) com bfloat16
- Transformers / HuggingFace
- H5PY para shards de embeddings
- NetworkX para grafos
- TensorBoard para logging
- RelTR (repositorio externo clonado em `reltr_repo/`)

## Estrutura Principal
- `models/encoders/` — DinoSceneEncoder, QwenSceneEmbedder
- `models/aligners/` — LoRACrossAttentionAligner (rank=64)
- `models/ups/` — AnyUpModel, JBUStack
- `models/SG/reltr_wrapper.py` — RelTRWrapper (modelo pre-treinado, sem fine-tuning)
- `data/` — Datasets, DataLoaders, extracao COYO
- `embeddings/` — Geracao de shards H5 (COYO)
- `viz/scene_graph_viz.py` — Visualizacao de retrieval + scene graph

## Pipeline de Execucao (ordem obrigatoria)
1. **Fase 0 — Dados COYO**: `get_metadados_coyo.py` → `get_small_sample_coyo.py` → `extract_files_coyo.py` → `embeddings/generate_shards.py`
2. **Fase 1 — Aligner**: `train_aligner_with_h5.py` (ou `train_aligner_with_Images.py` se VRAM > 16GB) → `checkpoints/best_aligner.pth`
3. **Fase 2 — RelTR**: Clonar repositorio e baixar pesos (ver instrucoes em `models/SG/reltr_wrapper.py`)
4. **Fase 3 — Avaliacao**: `eval_retrieval_with_h5.py` (Recall@K sobre H5) ou `eval_retrieval_sg.py` (retrieval + scene graph)

## Scripts e seus checkpoints
- `train_aligner_with_h5.py` — Treino do aligner com shards de embeddings → `best_aligner.pth`
- `train_aligner_with_Images.py` — Treino do aligner com imagens (VRAM > 16GB) → `best_aligner.pth`
- `eval_retrieval_with_h5.py` — Avaliacao Recall@K bidirecional sobre shards H5
- `eval_retrieval_with_images.py` — Avaliacao Recall@K sobre imagens brutas
- `eval_retrieval_sg.py` — Retrieval COYO + scene graph RelTR + visualizacao explicabilidade

## Setup RelTR
```bash
git clone https://github.com/yrcong/RelTR.git reltr_repo
mkdir -p checkpoints/reltr
# Baixar pesos VG de: https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD
# Salvar em: checkpoints/reltr/reltr_vg.pth
```

## Convencoes de Codigo
- Shapes documentados em todas as funcoes com `[B, C, H, W]`
- `@torch.no_grad()` em metodos de inferencia
- `bfloat16` para operacoes GPU e armazenamento H5
- Context managers obrigatorios para H5PY
- `tqdm` em todos os loops de processamento
- Erros de GPU: `torch.cuda.memory_summary()` antes de raise

## Arquitetura do Aligner
- visual_dim=768, text_dim=4096, rank=64
- Loss: `contrastive_loss + 0.05 * entropy_reg`
- Cross-Attention: Q=text, K=V=visual

## Dados
- COYO-700M (~15M imagens) — treino do aligner
- Shards H5 COYO: visual_feats [N,1024,768], text_feats [N,1,4096], visual_global [N,768] — 7GB por shard de 5k amostras (gzip)

## Delegacao para Subagentes
- **Ao modificar qualquer arquivo em `models/`** (incluindo `models/encoders/`, `models/aligners/`, `models/ups/`, `models/SG/`), delegar a tarefa para o subagente `cv-multimodal-expert` via Agent tool. Esse agente conhece o stack DINOv3 + Qwen3 + LoRA, scene graphs, knowledge graphs e metodos de upsampling, e deve liderar decisoes de arquitetura e revisao de codigo nessa area.
- Excecoes em que **nao** delegar: edicoes triviais (renomear variavel, ajustar import, corrigir typo) — resolver direto.
