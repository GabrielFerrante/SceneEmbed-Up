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
    
    idx = 0
    samples_per_folder = 1000  # Limite de ficheiros por subpasta
    
    with torch.no_grad():
        for imgs, texts in tqdm(dataloader, desc=f"Exportando"):
            globais, locais = dino.extract_features(imgs.to("cuda"))
            t_feat = qwen.embed_components(texts)

            batch_size = imgs.shape[0]
            
            for b in range(batch_size):
                # Cálculo da subpasta: 0, 1, 2...
                folder_idx = str(idx // samples_per_folder).zfill(5)
                target_folder = os.path.join(output_dir, folder_idx)
                os.makedirs(target_folder, exist_ok=True)
                
                sample_filename = os.path.join(target_folder, f"sample_{idx}.h5")
                
                with h5py.File(sample_filename, 'w') as f:
                    # Guardar features em float16 para poupar espaço
                    f.create_dataset("visual_global", data=globais[b].cpu().numpy().astype('float16'), compression="gzip")
                    f.create_dataset("text_feats", data=t_feat[b].cpu().numpy().astype('float16'), compression="gzip")
                    
                    local_key = "visual_local" if dino.model_up == "anyup" else "visual_feats"
                    f.create_dataset(local_key, data=locais[b].cpu().numpy().astype('float16'), compression="gzip")
                
                idx += 1

if __name__ == "__main__":
    dino_encoder = DinoSceneEncoder(device="cuda", upsampler="anyup") 
    qwen_embedder = QwenSceneEmbedder(device="cuda")
    
    # Criar dataloaders
    train_dataloader, val_dataloader = create_all_dataloaders("F:/COYO/coyo/extracted", batch_size=2, num_workers=4, t="train")
    
    # Agora passamos DIRETÓRIOS em vez de caminhos de arquivo únicos
    export_embeddings(train_dataloader, dino_encoder, qwen_embedder, output_dir="F:/COYO/embeds/train_anyup")
    export_embeddings(val_dataloader, dino_encoder, qwen_embedder, output_dir="F:/COYO/embeds/val_anyup")

    #export_embeddings(train_dataloader, "F:/COYO/embeds/train_embeddings_featup.h5")
    #export_embeddings(val_dataloader, "F:/COYO/embeds/val_embeddings_featup.h5")
