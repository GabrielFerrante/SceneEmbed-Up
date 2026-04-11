# GPU Memory

Padroes para evitar OOM e leaks. Complementa `code-style.md` (que ja exige
`bfloat16` na GPU e `torch.cuda.memory_summary()` em erros CUDA).

## Antes de treinar

- Verificar VRAM total e livre antes de iniciar:
  ```python
  total = torch.cuda.get_device_properties(0).total_memory / 1e9
  free  = (total*1e9 - torch.cuda.memory_allocated()) / 1e9
  ```
  `train_with_h5.py` ja faz isso no entrypoint — replicar em scripts novos.
- `torch.cuda.empty_cache()` **antes** do treino e em pontos de transicao
  (entre epocas, ao rotacionar shards). Nunca dentro de hot loops.

## Dtype

- `bfloat16` e o dtype padrao na GPU. Conversao via:
  ```python
  autocast_dtype = torch.bfloat16 if device == "cuda" else torch.float16
  tensor.to(device, non_blocking=True).to(autocast_dtype)
  ```
- Operacoes treinamento dentro de `with torch.amp.autocast(device_type=device, dtype=autocast_dtype):`.
- Conversao para `float32` apenas onde for **numericamente necessario** (ex.:
  loss reductions criticas). Nao converter por habito.
- Shards H5 sao `float16` (ver `data-pipeline.md`); a conversao para `bfloat16`
  acontece no batch loading, nao na geracao.

## Tratamento de erro CUDA

Padrao obrigatorio (ja em `train_with_h5.py` e `generate_shards.py`):

```python
try:
    ...
except RuntimeError as e:
    if "CUDA" in str(e).upper() and torch.cuda.is_available():
        print(torch.cuda.memory_summary())
    raise
```

Sempre re-raisar — nao silenciar.

## DataLoader

- `pin_memory=True` quando alvo for GPU.
- `non_blocking=True` em `.to(device)` se `pin_memory=True`.
- `num_workers` afeta RAM, nao VRAM. Para shards H5 grandes em RAM
  (`ShardedH5Dataset_withHD`), usar `num_workers=0` para evitar duplicacao.

## Anti-padroes

- Nao acumular tensores com grafo computacional em listas de logging — sempre
  `.detach().item()` ou `.detach().cpu()`.
- Nao chamar `.cuda()` em hot loops sem `non_blocking=True`.
- Nao misturar `float32` e `bfloat16` em operacoes binarias sem cast explicito
  (broadcasting silencioso causa upcast e dobra a memoria).
- Nao usar `torch.cuda.empty_cache()` esperando recuperar de OOM no meio de um
  step — ja eh tarde, o batch precisa diminuir.

## Batch size de referencia

Ver `CLAUDE.local.md`:
- Treino fim-a-fim (com encoders ativos): batch_size = 4
- Treino com shards H5 (`train_with_h5.py`): batch_size = 64
- Geracao de shards (`generate_shards.py`): batch_size = 4 (encoders ativos)
