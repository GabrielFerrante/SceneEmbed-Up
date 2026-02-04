import os
from huggingface_hub import HfApi, hf_hub_download
import json
dadosAuth = None
with open('token-HuggingFace.json', 'r', encoding='utf-8') as file:
    dadosAuth = json.load(file)


# CONFIGURAÇÕES
REPO_ID = "kakaobrain/coyo-700m"
PASTA_METADADOS = "F:/COYO/coyo_500gb_meta"
# O COYO tem aprox 5.5M de URLs por arquivo. 
# Baixar 5 arquivos nos dá ~27M de URLs para tentar baixar.
QTD_ARQUIVOS_META = 5 

def baixar_metadados_limitados():
    api = HfApi()
    
    print(f"Buscando lista de arquivos em {REPO_ID}...")
    arquivos_repo = api.list_repo_files(repo_id=REPO_ID, repo_type="dataset")
    
    # Pega os primeiros 5 arquivos .parquet da pasta data/
    arquivos_parquet = [f for f in arquivos_repo if f.startswith("data/") and f.endswith(".parquet")]
    arquivos_parquet.sort() # Garante ordem (00000, 00001, etc)
    
    selecionados = arquivos_parquet[:QTD_ARQUIVOS_META]
    
    print(f"Selecionados {len(selecionados)} arquivos de metadados para baixar.")
    
    if not os.path.exists(PASTA_METADADOS):
        os.makedirs(PASTA_METADADOS)

    for arquivo in selecionados:
        print(f"Baixando metadado: {arquivo} ...")
        hf_hub_download(
            repo_id=REPO_ID,
            filename=arquivo,
            repo_type="dataset",
            local_dir=PASTA_METADADOS,
            local_dir_use_symlinks=False,
            token= dadosAuth["token"]
        )

    print("\nSucesso! Metadados prontos.")
    print(f"Local: {PASTA_METADADOS}/data")

if __name__ == "__main__":
    baixar_metadados_limitados()