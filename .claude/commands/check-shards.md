# /project:check-shards

Valida a integridade de um diretorio de shards H5 gerados por
`embeddings/generate_shards.py`. Use antes de iniciar treino, apos crash do
gerador, ou para diagnosticar suspeita de corrupcao.

**Argumento:** caminho do diretorio (default: `F:/COYO/embeds/train_anyup/`,
ver `CLAUDE.local.md` para outros splits).

## Steps

1. **Listar shards** no diretorio com padrao `shard_*.h5` (ordenados).
   Reportar contagem total e nome do primeiro/ultimo.

2. **Para cada shard**, abrir com `with h5py.File(path, "r") as f:` e validar:
   - Datasets obrigatorios presentes: `visual_feats`, `text_feats`.
   - Dataset opcional: `visual_global` (shards antigos podem nao ter).
   - Shapes esperados (de `embeddings/generate_shards.py`):
     - `visual_feats`: `[N, 1024, 768]`, dtype `float16`
     - `text_feats`:   `[N, 1, 4096]`,   dtype `float16`
     - `visual_global`: `[N, 768]`,      dtype `float16` (se presente)
   - Eixo-0 consistente entre todos os datasets do mesmo shard.
   - `N <= 5000` (samples_per_shard padrao). Apenas o ultimo shard pode ter
     `N < 5000`; demais devem ter exatamente 5000.

3. **Saude numerica** (amostragem em shards grandes para nao explodir RAM):
   - `np.isnan(arr).any()` deve ser `False`.
   - `np.isinf(arr).any()` deve ser `False`.
   - Reportar quantas amostras tem norma zero (`(arr == 0).all(axis=-1).sum()`).
   - Reportar `min/max` por dataset.

4. **Resumo final:**
   - Total de samples (somar `N` de todos os shards).
   - Quantos shards completos vs incompletos.
   - Lista de shards com problemas (corrupcao, shapes errados, NaN/Inf).
   - Se tudo OK, imprimir "[OK] N shards validos, M samples totais".

## Referencias

- Logica de validacao de referencia: `_validate_shard()` em
  `embeddings/generate_shards.py:112`.
- Script ad-hoc existente: `data/test_shards.py` (pode ser reusado/expandido).
- Convencoes de shard: `.claude/rules/data-pipeline.md`.

## Observacoes

- **Nao** modificar nem deletar shards. Este comando eh **read-only**.
- Em diretorios muito grandes (> 100 shards), processar com `tqdm` e considerar
  amostragem em vez de leitura completa.
