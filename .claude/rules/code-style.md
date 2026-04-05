# Estilo de Codigo

- Documentar shapes de tensores em todas as funcoes: `[B, C, H, W]`
- Usar `@torch.no_grad()` em todos os metodos de inferencia
- Usar `bfloat16` como dtype padrao para GPU
- Context managers obrigatorios para H5PY (`with h5py.File(...) as f:`)
- Usar `tqdm` em todos os loops de processamento de dados
- Em erros de GPU, chamar `torch.cuda.memory_summary()` antes de raise
- Nomes de variaveis em ingles, comentarios podem ser em portugues
- Type hints em assinaturas de funcoes publicas
