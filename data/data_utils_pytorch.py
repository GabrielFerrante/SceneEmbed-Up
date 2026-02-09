import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torchvision import transforms
from PIL import Image

#NOTA: FOI CRIADO OUTRO ENV PARA EXEXUTAR A EXTRAÇÃO DOS DADOS

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

def create_all_dataloaders(
    root_dir, 
    batch_size=64, 
    image_size=256, 
    num_workers=4, 
    shuffle=True,
    t = "all"
    
):
    """
    Função helper para instanciar o DataLoader pronto para uso.
    """
    #  Cria as transformações
    transform = get_transforms(image_size)
    
    #  Instancia o Dataset
    full_dataset = CoyoExtractedDataset(root_dir=root_dir, transform=transform)
    total_size = len(full_dataset)
    collate_fn = CoyoCollate(tokenizer=None, max_length=77)
    
    #  Calcula os tamanhos das divisões (70/20/10)
    train_size = int(0.7 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    print(f"Divisão concluída:")
    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")
    print(f"  Teste: {len(test_dataset)} imagens")
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    if t == "all":
        return train_loader, val_loader, test_loader
    elif t == "train":
        return train_loader, val_loader
    elif t == "test":
        return test_loader

# --- TESTE RÁPIDO ---
if __name__ == "__main__":
    PATH = "F:/COYO/coyo/extracted"
    
    try:
        # Cria os 3 loaders usando a estratégia de split (70/20/10)
        train_loader, val_loader, test_loader = create_all_dataloaders(
            root_dir=PATH, 
            batch_size=4, 
            num_workers=0,  # 0 é melhor para debugar erros de caminho
            
        )
        
        # Lista para facilitar o loop de teste
        loaders = [
            ("TREINO", train_loader),
            ("VALIDAÇÃO", val_loader),
            ("TESTE", test_loader)
        ]
        
        print("\n" + "="*30)
        print("INICIANDO TESTE DOS LOADERS")
        print("="*30)

        for nome, loader in loaders:
            print(f"\n--- Testando Loader: {nome} ---")
            
            # Pega apenas o primeiro batch de cada loader
            batch = next(iter(loader))
            
            if batch is None:
                print(f"  [!] Aviso: Batch de {nome} retornou None (vazio ou corrompido).")
                continue
                
            images, texts = batch
            
            # Verificações de Shape
            print(f"  [OK] Imagens Shape: {images.shape}") # Esperado: [4, 3, 256, 256]
            
            if isinstance(texts, dict) and 'input_ids' in texts:
                # Caso com Tokenizer (Hugging Face)
                print(f"  [OK] Tokenizer Detectado (Input IDs): {texts['input_ids'].shape}")
                print(f"  [OK] Exemplo de IDs (Primeiro item): {texts['input_ids'][0][:10]}...")
            else:
                # Caso Texto Raw (Lista de strings)
                print(f"  [OK] Texto Raw Detectado. Quantidade: {len(texts)}")
                print(f"  [OK] Exemplo de Legenda: {texts[0][:50]}...")

        print("\n" + "="*30)
        print("TESTE FINALIZADO COM SUCESSO!")
        print("="*30)

    except FileNotFoundError as e:
        print(f"\n[ERRO] Caminho não encontrado: {e}")
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro inesperado: {e}")
        import traceback
        traceback.print_exc()
        
