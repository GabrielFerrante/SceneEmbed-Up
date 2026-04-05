# Configuracao de Deploy

## Checkpoint
- Path padrao: `checkpoints/best_aligner.pth`
- Formato: PyTorch state_dict (apenas parametros treinaveis do LoRA)

## Parametros do modelo para reconstrucao
- visual_dim: 768
- text_dim: 4096
- rank: 64
- num_heads: 8

## Exportacao
- Incluir: state_dict, hiperparametros, metricas de validacao
- Excluir: embeddings H5, dados COYO, logs TensorBoard
