import h5py
import torch
import os
import sys
sys.path.append('..')
from tqdm import tqdm
from models.encoders.dinov3_extrator import DinoSceneEncoder
from models.encoders.qwen3_extrator import QwenSceneEmbedder
from data.data_utils_pytorch import create_all_dataloaders


def export_embeddings(dataloader, dino, qwen, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Conta quantos arquivos já existem para definir o ponto de partida
    # Isso permite que você feche o script e abra novamente
    existing_files = []
    for root, dirs, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".h5"):
                # Extrai o número do arquivo sample_123.h5 -> 123
                try:
                    num = int(file.split('_')[1].split('.')[0])
                    existing_files.append(num)
                except: continue
    
    start_idx = max(existing_files) + 1 if existing_files else 0
    print(f"Iniciando/Retomando do índice: {start_idx}")
    
    idx = 0
    samples_per_folder = 1000 
    
    # Garantir que os modelos estão em eval e na GPU

    with torch.no_grad():
        for imgs, texts in tqdm(dataloader, desc="Exportando"):
            
            batch_size = imgs.shape[0]
            
            # Verifica se todo este batch já foi processado
            if idx + batch_size <= start_idx:
                idx += batch_size
                continue
            
            try:
                
                # 1. Extração Visual (DinoV3)
                # globais: [B, 768], locais: [B, 768, H, W]
                globais, locais = dino.extract_features(imgs.to("cuda"))
                
                # 2. Processamento Local (Igual ao seu treino)
                # Reduz para 32x32 via pooling
                locais_small = torch.nn.functional.adaptive_avg_pool2d(locais, (32, 32))
                
                # 3. Extração de Texto (Qwen)
                # Formata textos como lista de listas para bater com a lógica interna
                formatted_texts = [[t] for t in texts]
                t_feat = qwen.embed_components(formatted_texts, normalize=False)
                
                # Garantir shape [B, 1, 4096] para o texto
                if t_feat.dim() == 2:
                    t_feat = t_feat.unsqueeze(1)

                batch_size = imgs.shape[0]
                
                for b in range(batch_size):
                    folder_idx = str(idx // samples_per_folder).zfill(5)
                    target_folder = os.path.join(output_dir, folder_idx)
                    os.makedirs(target_folder, exist_ok=True)
                    
                    sample_filename = os.path.join(target_folder, f"sample_{idx}.h5")
                    
                    # --- PROCESSAMENTO DA FEATURE LOCAL POR AMOSTRA ---
                    # locais_small[b] tem shape [768, 32, 32]
                    feat_b = locais_small[b] 
                    c, h, w = feat_b.shape
                    # Achata (reshape) e Transpõe: [768, 1024] -> [1024, 768]
                    flat_feat = feat_b.reshape(c, -1).transpose(0, 1) 
                    
                    with h5py.File(sample_filename, 'w') as f:
                        # Guardamos a feature visual já pronta para o Aligner [1024, 768]
                        f.create_dataset("visual_feats", data=flat_feat.cpu().numpy().astype('float16'), compression="gzip")
                        
                        # Guardamos a feature de texto [1, 4096]
                        f.create_dataset("text_feats", data=t_feat[b].cpu().numpy().astype('float16'), compression="gzip")
                        
                        # Opcional: Guardar a global caso precise no futuro
                        f.create_dataset("visual_global", data=globais[b].cpu().numpy().astype('float16'), compression="gzip")
                    
                    idx += 1
            except KeyboardInterrupt:
                print(f"\nInterrupção detectada! Processamento parado no índice {idx}. Pode fechar agora.")
                return # Sai da função de forma limpa
                

if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="anyup") 
    qwen_embedder = QwenSceneEmbedder(device="cuda")
    
    # Criar dataloaders
    train_dataloader, val_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=4, num_workers=8, t="train")
    
    # Agora passamos DIRETÓRIOS em vez de caminhos de arquivo únicos
    export_embeddings(train_dataloader, dino_encoder, qwen_embedder, output_dir="F:/COYO/embeds/train_anyup")
    export_embeddings(val_dataloader, dino_encoder, qwen_embedder, output_dir="F:/COYO/embeds/val_anyup")

