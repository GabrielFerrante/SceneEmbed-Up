# Regras de Teste

- Validar shapes de tensores apos cada transformacao critica
- Testar com batch_size=1 e batch_size>1 para detectar erros de broadcasting
- Usar `torch.allclose()` para comparacoes numericas com tolerancia
- Testes de GPU devem ter fallback para CPU com `@pytest.mark.skipif(not torch.cuda.is_available())`
- Verificar memory leaks com `torch.cuda.memory_allocated()` antes e depois
