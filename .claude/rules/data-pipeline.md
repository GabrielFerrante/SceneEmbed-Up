# Pipeline de Dados (data/ + embeddings/)

Regras especificas para geracao e consumo de shards H5 e datasets COYO.
Ver `code-style.md` para convencoes gerais (shapes, tqdm, context managers).

## Shards H5 — formato fixo

Datasets dentro de cada `shard_NNNNNN.h5` (nomes literais, nao alterar):

| Dataset         | Shape              | Dtype      | Obrigatorio |
|-----------------|--------------------|------------|-------------|
| `visual_feats`  | `[N, 1024, 768]`   | `float16`  | sim         |
| `text_feats`    | `[N, 1, 4096]`     | `float16`  | sim         |
| `visual_global` | `[N, 768]`         | `float16`  | opcional    |

**Atencao:** shards usam `float16`, **nao** `bfloat16`. A conversao para `bfloat16`
acontece no training loop via `tensor.to(autocast_dtype)`. Nao mudar o dtype dos
shards sem considerar o impacto em VRAM e precisao downstream.

## Convencoes de escrita

- Sempre usar `with h5py.File(...) as f:` ou a classe `ShardWriter` em
  `embeddings/generate_shards.py`.
- Datasets criados com `compression="gzip"` e `chunks` configurados (ver
  `ShardWriter._open_shard`). Manter consistencia: `chunks=(64, 1024, 768)` para
  visual, `(64, 1, 4096)` para texto, `(256, 768)` para global.
- Nome dos arquivos: `shard_NNNNNN.h5` com 6 digitos (zfill). **Nao** mudar o
  padrao — `_get_resume_state` faz parsing por esse formato.
- `samples_per_shard` padrao: `5_000` (≈7GB gzip). Alterar implica revalidar
  consumo de RAM e tempo de I/O.

## Resumption (retomada apos crash)

- `ShardWriter(resume=True)` detecta o ultimo shard e o numero de amostras ja
  escritas via `_get_resume_state`.
- Em codigo novo que consome shards, **nunca** assumir que o ultimo shard esta
  cheio — sempre ler `f["visual_feats"].shape[0]`.
- O loop de exportacao em `export_embeddings_sharded` faz skip por contagem
  global (`global_idx`); preservar essa logica em qualquer refator.

## Datasets / DataLoaders

- Classes em `data/data_utils_pytorch.py`: `CoyoExtractedDataset`,
  `ShardedH5Dataset_withHD`, `ShardedH5Dataset_withSSD`.
- Collate functions (`CoyoCollate`) **devem retornar tensores em CPU**. Mover
  para GPU eh responsabilidade do training loop com `.to(device, non_blocking=True)`.
- `CoyoCollate` ja filtra `None` (imagens corrompidas). Ao adicionar nova
  fonte de dados, preservar esse filtro para evitar crash em batches parciais.

## Paths

- Paths COYO (`F:/COYO/...`, `G:/coyo/...`) vivem em `CLAUDE.local.md`.
  **Nunca hardcoded** em codigo novo. Receber via argumento ou ler de config.
- O token HuggingFace (`tokenDINOV3.json`, `data/token-HuggingFace.json`) **nunca**
  deve ser commitado nem logado.
