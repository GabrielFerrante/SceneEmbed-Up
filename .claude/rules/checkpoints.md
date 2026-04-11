# Checkpoints

Convencoes para salvar e carregar pesos do Aligner. Ver `skills/deploy/` para
exportacao final (apenas LoRA).

## Diretorio e nomeacao

- Diretorio padrao: `checkpoints/` (criado via `os.makedirs(..., exist_ok=True)`).
- Checkpoint por epoca: `aligner_epoch_{N}.pth` (1-based), gerado por
  `utils.checkpoint.save_epoch_checkpoint()`.
- Melhor modelo: `best_aligner.pth`, gerado por `utils.early_stopping.EarlyStopping`.
- **Nao** salvar checkpoints com nomes ad-hoc — usar sempre as funcoes acima
  para garantir uniformidade.

## O que e salvo

- `save_epoch_checkpoint(model, epoch)` salva **`model.state_dict()` completo**
  do `LoRACrossAttentionAligner`. Isso inclui `visual_proj` (768→4096) alem dos
  parametros LoRA. **Nao** filtrar manualmente: o aligner inteiro e necessario
  para retomar treino.
- `EarlyStopping` recebe um `model_state` (dict) ja preparado pelo chamador e
  o persiste via `torch.save`. O training loop passa `aligner.state_dict()`.
- A reducao para apenas-LoRA acontece **na exportacao** (skill `deploy`),
  nunca durante o treino.

## Como carregar

- Usar `torch.load(path, map_location=device)` seguido de
  `aligner.load_state_dict(state, strict=True)`.
- Encoders (DINO/Qwen) nao sao restaurados de checkpoint — sao recriados a
  partir do HuggingFace toda vez. Apenas o aligner persiste.
- Apos `load_state_dict`, lembrar de mover para o dtype correto:
  `aligner.to(device).to(torch.bfloat16)`.

## EarlyStopping

- Default: `patience=5`, `min_delta=0.001`, monitora **loss** (quanto menor
  melhor). `train_with_h5.py` usa `patience=10`.
- Sempre passar `current_loss=avg_val_loss` (validacao), nao train loss.
- O contador reseta apenas quando `current_loss < best_loss - min_delta`.

## Anti-padroes

- Nao usar `torch.save(model, path)` (salva o objeto inteiro) — sempre
  `state_dict`.
- Nao salvar checkpoint a cada step — apenas por epoca.
- Nao deletar checkpoints antigos automaticamente; deixar essa decisao para o
  usuario (ver protecoes em `hooks/validate-bash.sh`).
