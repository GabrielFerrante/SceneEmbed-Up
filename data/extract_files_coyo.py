import tarfile
import os
import glob
from tqdm import tqdm # Barra de progresso (pip install tqdm)

# --- CONFIGURAÇÕES ---
# Onde estão os arquivos .tar baixados pelo img2dataset?
PASTA_ORIGEM = "F:/COYO/coyo" 

# Onde você quer salvar as imagens soltas?
PASTA_DESTINO = "F:/COYO/coyo/extracted"

# Limite de segurança: Quantos TARs você quer extrair agora?
# Coloque None para extrair TUDO (Cuidado com espaço em disco!)
LIMITE_TARS = None

def extrair_dataset():
    # 1. Encontra todos os arquivos .tar na pasta de origem
    arquivos_tar = sorted(glob.glob(os.path.join(PASTA_ORIGEM, "*.tar")))
    
    if not arquivos_tar:
        print(f"Nenhum arquivo .tar encontrado em {PASTA_ORIGEM}")
        return

    print(f"Total de arquivos .tar encontrados: {len(arquivos_tar)}")
    
    # Aplica o limite se definido
    if LIMITE_TARS:
        arquivos_tar = arquivos_tar[:LIMITE_TARS]
        print(f"--- MODO DE TESTE ---")
        print(f"Extraindo apenas os primeiros {LIMITE_TARS} arquivos .tar.")
        print(f"Para extrair tudo, altere LIMITE_TARS = None no script.")
    
    # 2. Loop principal de extração
    for caminho_tar in tqdm(arquivos_tar, desc="Extraindo pacotes"):
        try:
            # Nome do arquivo sem extensão (ex: '00000')
            nome_base = os.path.splitext(os.path.basename(caminho_tar))[0]
            
            # Cria uma subpasta para este pacote específico
            # Ex: ./coyo_dataset_final/00000/
            subpasta_destino = os.path.join(PASTA_DESTINO, nome_base)
            
            if not os.path.exists(subpasta_destino):
                os.makedirs(subpasta_destino)
            
            # 3. Abre e extrai o TAR
            with tarfile.open(caminho_tar, "r:") as tar:
                # O img2dataset salva jpg, txt e json. Vamos extrair tudo.
                tar.extractall(path=subpasta_destino)
                
        except Exception as e:
            print(f"Erro ao extrair {caminho_tar}: {e}")

    print("\n--- Concluído ---")
    print(f"Imagens organizadas em: {os.path.abspath(PASTA_DESTINO)}")
    print("Cada subpasta contém os pares (imagem.jpg + imagem.txt)")

if __name__ == "__main__":
    extrair_dataset()