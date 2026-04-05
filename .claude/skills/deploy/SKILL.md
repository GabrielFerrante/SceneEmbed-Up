# Skill: Deploy

Automatiza o processo de exportacao e deploy de checkpoints treinados.

## Trigger
Quando o usuario pedir para exportar, empacotar ou fazer deploy de um modelo treinado.

## Steps
1. Verificar que o checkpoint existe em `checkpoints/`
2. Validar o state_dict carregando o Aligner
3. Exportar apenas os pesos LoRA (parametros treinaveis)
4. Gerar metadata (epoch, metricas, hiperparametros)
5. Empacotar em formato distribuivel
