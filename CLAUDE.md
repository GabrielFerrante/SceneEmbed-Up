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
- `models/SG/` — AttributeClassifier, AttributeHead, KnowledgeGraphGenerator, RelationHead, HasBboxGrid, Detection, RelationPredictor
- `models/detectors` - DetrDetector
- `data/` — Datasets, DataLoaders, extracao COYO, extracao VisualGenome
- `embeddings/` — Geracao e consolidacao de shards H5
- `train_aligner_with_h5.py` — Treino do aligner com shards de embeddings
- `train_aligner_with_Images.py` — Treino do aligner com imagens (executar em caso de VRAM alta > 16gb)
- `train_attribute_head.py` — Treino do Classificador multi-label de atributos sobre features no espaco aligned (4096-d). 
- `train_detr_vg150.py` — Treino fine-tuning do modelo DETR-R50 com head pre-treinada para 150 classes de objetos
- `train_relation_head.py` — Treina a RelationHead sobre pares GT do Visual Genome (VG-150).
- `eval_retrieval_with_h5.py` / `eval_retrieval_with_images.py` — Avaliacao para Retrieval
- `eval_sg_vg` — Avalia a geração de grafos de cena e grafos de conhecimento (SG e KG)

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
- VisualGenome para geração de grafos
## Delegacao para Subagentes
- **Ao modificar qualquer arquivo em `models/`** (incluindo `models/encoders/`, `models/aligners/`, `models/ups/`, `models/SG/`, `models/detectors/`), delegar a tarefa para o subagente `cv-multimodal-expert` via Agent tool. Esse agente conhece o stack DINOv3 + Qwen3 + LoRA, scene graphs, knowledge graphs e metodos de upsampling, e deve liderar decisoes de arquitetura e revisao de codigo nessa area.
- Excecoes em que **nao** delegar: edicoes triviais (renomear variavel, ajustar import, corrigir typo) — resolver direto.
