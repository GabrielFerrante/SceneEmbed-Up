# SceneEmbed-Up

## Visao Geral
Pipeline de pesquisa para alinhamento multimodal de alta resolucao e geracao de Scene Graphs.
Combina DINOv3 (encoder visual com upsampling HR) + Qwen3-Embedding-8B (encoder textual) + LoRA Cross-Attention Aligner.

## Tech Stack
- Python 3.10+
- PyTorch (CUDA 12.1) com bfloat16
- Transformers / HuggingFace
- H5PY para shards de embeddings
- NetworkX para grafos
- TensorBoard para logging

## Estrutura Principal
- `models/encoders/` — DinoSceneEncoder, QwenSceneEmbedder
- `models/aligners/` — LoRACrossAttentionAligner (rank=64)
- `models/ups/` — AnyUpModel, JBUStack, LoftUpModel
- `models/SG/` — SceneGraphGenerator, KnowledgeGraphGenerator
- `data/` — Datasets, DataLoaders, extracao COYO
- `embeddings/` — Geracao e consolidacao de shards H5
- `train_with_h5.py` — Treino rapido (recomendado)
- `train_with_Images.py` — Treino fim-a-fim
- `eval_with_h5.py` / `eval_with_images.py` — Avaliacao

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
- loss_vg e apenas metrica de monitoramento (TensorBoard)
- Cross-Attention: Q=text, K=V=visual

## Dados
- COYO-700M (~15M imagens)
- Shards H5: visual_feats [N,1024,768], text_feats [N,1,4096], visual_global [N,768]
- 7GB por shard de 5k amostras (gzip)
