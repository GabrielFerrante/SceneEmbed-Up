import os
import requests
from PIL import Image
from io import BytesIO
from datasets import load_dataset

# CONFIGURAÇÕES
NUM_AMOSTRAS = 50        # Quantos itens queremos tentar baixar
PASTA_SAIDA = "F:/COYO"  # Onde salvar os arquivos
TIMEOUT = 5              # Segundos para esperar por uma imagem

def baixar_amostra():
    # 1. Cria a pasta de saída se não existir
    if not os.path.exists(PASTA_SAIDA):
        os.makedirs(PASTA_SAIDA)
    
    print(f"--- Iniciando conexão com COYO-700M (Modo Streaming) ---")
    
    # 2. Carrega o dataset em modo streaming (não baixa o arquivo gigante)
    # Isso permite ler item por item instantaneamente via internet
    ds = load_dataset("kakaobrain/coyo-700m", split="train", streaming=True)
    
    # Iterador para percorrer o dataset
    dataset_iter = iter(ds)
    
    sucessos = 0
    tentativas = 0

    print(f"--- Tentando baixar {NUM_AMOSTRAS} amostras... ---")

    while tentativas < NUM_AMOSTRAS:
        try:
            # Pega o próximo item do dataset (contém 'url', 'text', etc.)
            item = next(dataset_iter)
            url_imagem = item['url']
            texto_legenda = item['text']
            id_imagem = item['id'] # Ou gerar um ID sequencial se preferir
            
            tentativas += 1
            print(f"[{tentativas}/{NUM_AMOSTRAS}] Tentando URL: {url_imagem[:50]}...")

            # 3. Tenta baixar a imagem real
            headers = {'User-Agent': 'Mozilla/5.0'} # Para evitar bloqueios simples
            response = requests.get(url_imagem, headers=headers, timeout=TIMEOUT)
            
            if response.status_code == 200:
                # Processa a imagem
                img = Image.open(BytesIO(response.content))
                
                # Converte para RGB para evitar erros com PNGs transparentes ao salvar como JPG
                if img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # 4. Salva a Imagem
                nome_arquivo = f"{id_imagem}"
                caminho_img = os.path.join(PASTA_SAIDA, f"{nome_arquivo}.jpg")
                img.save(caminho_img)
                
                # 5. Salva a Legenda (Caption) em um arquivo de texto ao lado
                caminho_txt = os.path.join(PASTA_SAIDA, f"{nome_arquivo}.txt")
                with open(caminho_txt, "w", encoding="utf-8") as f:
                    f.write(texto_legenda)
                
                print(f"   -> Sucesso! Salvo em: {caminho_img}")
                sucessos += 1
            else:
                print(f"   -> Falha: Status Code {response.status_code}")

        except Exception as e:
            # Captura erros de conexão, URLs quebradas ou timeouts
            print(f"   -> Erro ao baixar: {e}")
            continue

    print("-" * 30)
    print(f"Processo finalizado.")
    print(f"Tentativas: {tentativas} | Downloads com Sucesso: {sucessos}")
    print(f"Verifique a pasta '{PASTA_SAIDA}'")

if __name__ == "__main__":
    baixar_amostra()