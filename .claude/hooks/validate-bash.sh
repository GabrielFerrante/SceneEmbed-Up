#!/bin/bash
# Hook: Valida comandos bash antes da execucao
# Bloqueia operacoes destrutivas acidentais

COMMAND="$1"

# Bloquear delecao recursiva de diretorios criticos
if echo "$COMMAND" | grep -qE "rm\s+-rf\s+(/|\.\.|\*|checkpoints|models|data)"; then
  echo "BLOCKED: Delecao recursiva de diretorio critico detectada"
  exit 1
fi

# Bloquear force push
if echo "$COMMAND" | grep -qE "git\s+push\s+.*--force"; then
  echo "BLOCKED: Force push nao permitido"
  exit 1
fi

# Bloquear reset hard
if echo "$COMMAND" | grep -qE "git\s+reset\s+--hard"; then
  echo "BLOCKED: git reset --hard nao permitido"
  exit 1
fi

exit 0
