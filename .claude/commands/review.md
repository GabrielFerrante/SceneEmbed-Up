# /project:review

Revise o codigo alterado verificando:

1. **Shapes**: Todos os tensores possuem shapes documentados e consistentes ao longo do pipeline
2. **Dtype**: Operacoes GPU usam bfloat16, sem mixed precision acidental
3. **Memory**: Sem leaks obvios (tensores acumulados em listas sem detach, grafos computacionais retidos)
4. **Loss**: Componentes de loss estao corretos (contrastive + entropy_reg, loss_vg apenas monitoring)
5. **Convencoes**: @torch.no_grad em inferencia, context managers para H5, tqdm em loops

Reporte problemas encontrados com severidade (critico/medio/baixo) e sugira correcoes.
