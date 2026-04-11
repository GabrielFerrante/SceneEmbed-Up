#!/bin/bash
# Hook: Valida comandos bash antes da execucao
# Bloqueia operacoes destrutivas acidentais

COMMAND="$1"

# Bloquear delecao recursiva de diretorios criticos
if echo "$COMMAND" | grep -qE "rm\s+-rf\s+(/|\.\.|\*|checkpoints|models|data|embeddings|logs|results)"; then
  echo "BLOCKED: Delecao recursiva de diretorio critico detectada"
  exit 1
fi

# Bloquear delecao em paths COYO (datasets pesados em discos externos)
if echo "$COMMAND" | grep -qiE "rm\s+-rf\s+[a-z]:?/coyo"; then
  echo "BLOCKED: Delecao em path COYO (F:/COYO ou G:/coyo) bloqueada"
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

# Bloquear commit acidental do token HuggingFace do DINO
if echo "$COMMAND" | grep -qE "git\s+add\s+.*tokenDINOV3\.json"; then
  echo "BLOCKED: tokenDINOV3.json contem credenciais HuggingFace e nao deve ser commitado"
  exit 1
fi

# Bloquear commit acidental do token HuggingFace generico
if echo "$COMMAND" | grep -qE "git\s+add\s+.*token-HuggingFace\.json"; then
  echo "BLOCKED: token-HuggingFace.json contem credenciais e nao deve ser commitado"
  exit 1
fi

# Bloquear commit acidental do CLAUDE.local.md (paths locais)
if echo "$COMMAND" | grep -qE "git\s+add\s+.*CLAUDE\.local\.md"; then
  echo "BLOCKED: CLAUDE.local.md contem paths locais e nao deve ser commitado"
  exit 1
fi

exit 0
