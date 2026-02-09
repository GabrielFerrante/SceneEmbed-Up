import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image

# --- CONFIGURAÇÕES DE IMAGEM ---
def get_transforms(image_size=256):
    """
    Define o pipeline de pré-processamento da imagem.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)), # Garante tamanho fixo
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # Padrão ImageNet
            std=[0.229, 0.224, 0.225]
        ),
    ])
    
class CoyoCollate:
    def __init__(self, tokenizer=None, max_length=77):
        """
        Args:
            tokenizer: Instância de um tokenizer (ex: BertTokenizer, CLIPTokenizer).
                       Se None, retorna o texto cru (lista de strings).
            max_length: Tamanho máximo da sequência de tokens (padrão CLIP é 77).
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        """
        Esta função recebe uma lista de tuplas [(img, txt), (img, txt), None, (img, txt)...]
        """
        # 1. Filtragem de Erros (Remove Nones)
        # Se __getitem__ retornou None (imagem corrompida), removemos aqui.
        batch = [item for item in batch if item is not None]
        
        if len(batch) == 0:
            # Caso extremo: todo o batch estava corrompido
            return None
        
        # Separa imagens e textos
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]
        
        # 2. Processa Imagens
        # Empilha a lista de tensores em um único tensor [B, C, H, W]
        images_tensor = torch.stack(images, dim=0)
        
        # 3. Processa Textos (Tokenização)
        if self.tokenizer:
            # Tokeniza o batch inteiro de uma vez (mais eficiente)
            text_tokens = self.tokenizer(
                captions,
                padding="max_length",  # Preenche até max_length
                truncation=True,       # Corta se for maior que max_length
                max_length=self.max_length,
                return_tensors="pt"    # Retorna tensores PyTorch
            )
            return images_tensor, text_tokens
        else:
            # Se não tiver tokenizer, retorna as strings cruas
            return images_tensor, captions

class CoyoExtractedDataset(Dataset):
    def __init__(self, root_dir, transform=None, extensions=('.jpg', '.jpeg', '.png')):
        """
        Args:
            root_dir (str): Caminho base (ex: ./coyo_dataset_final)
            transform (callable, optional): Transformações do PyTorch.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        
        print(f"Iniciando varredura em: {root_dir} (Isso pode demorar um pouco...)")
        
        # Estratégia Eficiente: Scaneia subpastas recursivamente
        # Padrão esperado: root_dir/subpasta/imagem.jpg
        search_pattern = os.path.join(root_dir, "**", "*")
        
        # Glob recursivo para encontrar todas as imagens
        # O uso de iglob é mais eficiente em memória, mas precisamos da lista para __len__
        for ext in extensions:
            self.image_paths.extend(glob.glob(os.path.join(root_dir, "**", f"*{ext}"), recursive=True))
            
        
        print(f"Dataset carregado! Total de imagens encontradas: {len(self.image_paths)}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Caminho da Imagem
        img_path = self.image_paths[idx]
        
        # 2. Derivar Caminho do Texto e JSON
        # Assume que o txt tem o mesmo nome base da imagem
        base_path = os.path.splitext(img_path)[0]
        txt_path = base_path + ".txt"
        
        # 3. Carregar Imagem
        try:
            image = Image.open(img_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            # Em caso de imagem corrompida, retorna um tensor vazio ou trata o erro
            print(f"Erro ao ler imagem {img_path}: {e}")
            # Se der erro, retorna None para filtrar depois
            return None

        # 4. Carregar Texto
        caption = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
                    caption = f.read().strip()
            except Exception as e:
                print(f"Erro ao ler texto {txt_path}: {e}")
        
        # Retorna o par
        return image, caption

def create_dataloader(
    root_dir, 
    batch_size=64, 
    image_size=256, 
    num_workers=4, 
    shuffle=True,
    
):
    """
    Função helper para instanciar o DataLoader pronto para uso.
    """
    # 1. Cria as transformações
    transform = get_transforms(image_size)
    
    # 2. Instancia o Dataset
    dataset = CoyoExtractedDataset(root_dir=root_dir, transform=transform)
    collate_fn = CoyoCollate(tokenizer=None, max_length=77)
    # 3. Cria o DataLoader
    # pin_memory=True acelera a transferência da RAM para a VRAM (GPU)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True, # Evita problemas com batch incompleto no final
        collate_fn= collate_fn
    )
    
    return loader
"""
# --- TESTE RÁPIDO ---
if __name__ == "__main__":
    loader = create_dataloader("F:/COYO/coyo/extracted", batch_size=4, num_workers=0)
    
    batch = next(iter(loader))
    
    # Caso raro de batch vazio (tudo corrompido)
    if batch is None: 
        print("Batch vazio!")
    else:
        images, texts = batch
        
        print(f"Images Shape: {images.shape}") # Deve ser [4, 3, 256, 256]
        
        if isinstance(texts, list):
            print(f"Textos (Raw): {texts}")
        else:
            # Se usou tokenizer
            print(f"Input IDs Shape: {texts['input_ids'].shape}") # Deve ser [4, 77]
            print(f"Attention Mask: {texts['attention_mask'].shape}")
        
"""