import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
import threading
from torchvision import transforms
import random
from PIL import Image
import numpy as np
from tqdm import tqdm
import h5py
#NOTA: FOI CRIADO OUTRO ENV PARA EXEXUTAR A EXTRAÇÃO DOS DADOS

def get_transforms(image_size=256):
    """
    Define o pipeline de pré-processamento da imagem.
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((image_size, image_size)), # Garante tamanho fixo
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
    
class ShardedH5Dataset_withSSD(torch.utils.data.Dataset):
    """
    Dataset para shards H5 em SSD com cache de file handles.

    Mantém os arquivos h5 abertos para evitar o overhead de open/close
    por amostra. Handles são criados lazily — compatível com num_workers>0
    (cada worker abre seus próprios handles após o spawn).
    """

    def __init__(self, folder_path: str):
        search_pattern = os.path.join(folder_path, "**", "*.h5")
        shard_files = sorted(glob.glob(search_pattern, recursive=True))

        if not shard_files:
            raise RuntimeError(f"Nenhum arquivo .h5 encontrado em {folder_path}")

        self._index: list[tuple[str, int]] = []
        for path in shard_files:
            try:
                with h5py.File(path, "r") as f:
                    n = f["visual_feats"].shape[0]
                for local_idx in range(n):
                    self._index.append((path, local_idx))
            except Exception as e:
                print(f"[WARN] Shard ignorado: {path} — {e}")

        if not self._index:
            raise RuntimeError("Nenhuma amostra válida encontrada nos shards.")

        self._handles: dict[str, h5py.File] = {}
        print(f"[ShardedH5Dataset_withSSD] {len(self._index):,} amostras em {len(shard_files)} shards")

    def _get_handle(self, path: str) -> h5py.File:
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r")
        return self._handles[path]

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx):
        shard_path, local_idx = self._index[idx]
        f = self._get_handle(shard_path)
        visual = f["visual_feats"][local_idx]   # [1024, 768]
        text   = f["text_feats"][local_idx]     # [1, 4096]
        return torch.from_numpy(visual.copy()), torch.from_numpy(text.copy())

    def __del__(self):
        for f in self._handles.values():
            try:
                f.close()
            except Exception:
                pass
        
        
class ShardedH5Dataset_withHD(Dataset):
    """
    Dataset com buffer rotativo de shards em memória.

    Carrega `shards_in_memory` shards por vez na RAM. Quando todos os
    samples do buffer forem consumidos, o buffer é recarregado com os
    próximos shards automaticamente em background.

    Parameters
    ----------
    folder_path:
        Diretório raiz com os arquivos `.h5`.
    shards_in_memory:
        Quantos shards manter na RAM simultaneamente.
    shuffle_shards:
        Se True, embaralha a ordem dos shards a cada rotação.
    """

    def __init__(self, folder_path: str, shards_in_memory: int = 9, shuffle_shards: bool = True):
        search_pattern = os.path.join(folder_path, "**", "*.h5")
        self._all_shards = sorted(glob.glob(search_pattern, recursive=True))

        if not self._all_shards:
            raise RuntimeError(f"Nenhum arquivo .h5 encontrado em {folder_path}")

        self.shards_in_memory = shards_in_memory
        self.shuffle_shards   = shuffle_shards

        # Fila de shards ainda não carregados nesta epoch
        self._shard_queue: list[str] = []

        # Tensores ativos em RAM
        self._visual: torch.Tensor | None = None
        self._text:   torch.Tensor | None = None

        # Prefetch: próximo buffer sendo carregado em background
        self._next_visual: torch.Tensor | None = None
        self._next_text:   torch.Tensor | None = None
        self._prefetch_thread: threading.Thread | None = None

        # Carrega o primeiro buffer de forma síncrona
        self._reset_queue()
        self._load_next_buffer_sync()
        
        # Dispara prefetch do buffer seguinte
        self._start_prefetch()

    # ------------------------------------------------------------------
    # Gerenciamento de fila e buffers
    # ------------------------------------------------------------------

    def _reset_queue(self) -> None:
        """Recarrega a fila com todos os shards (embaralhados ou não)."""
        self._shard_queue = list(self._all_shards)
        if self.shuffle_shards:
            random.shuffle(self._shard_queue)

    def _load_shards(self, paths: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Lê uma lista de shards e retorna tensores concatenados."""
        visuals, texts = [], []
        for path in tqdm(paths, desc="  Carregando shards", leave=False):
            try:
                with h5py.File(path, "r") as f:
                    visuals.append(torch.from_numpy(f["visual_feats"][:]))
                    texts.append(torch.from_numpy(f["text_feats"][:]))
            except Exception as e:
                print(f"[WARN] Shard ignorado: {path} — {e}")

        if not visuals:
            raise RuntimeError("Nenhum shard válido carregado no buffer.")

        return torch.cat(visuals, dim=0), torch.cat(texts, dim=0)

    def _load_next_buffer_sync(self) -> None:
        """Carrega o próximo grupo de shards de forma síncrona para o buffer ativo."""
        if not self._shard_queue:
            self._reset_queue()

        batch = self._shard_queue[:self.shards_in_memory]
        self._shard_queue = self._shard_queue[self.shards_in_memory:]

        print(f"\n[Buffer] Carregando {len(batch)} Next's shards em memória...")
        self._visual, self._text = self._load_shards(batch)
        print(f"[Buffer] {len(self._visual):,} amostras disponíveis.")

    def _start_prefetch(self) -> None:
        """Dispara thread de background para pré-carregar o próximo buffer."""
        if not self._shard_queue:
            # Fila vazia: próxima carga virá de uma nova epoch
            return

        batch = self._shard_queue[:self.shards_in_memory]
        # Não remove da fila ainda — só remove quando o buffer for rotacionado

        def _worker():
            self._next_visual, self._next_text = self._load_shards(batch)

        self._prefetch_thread = threading.Thread(target=_worker, daemon=True)
        self._prefetch_thread.start()

    def rotate_buffer(self) -> None:
        """
        Descarta o buffer atual e ativa o próximo (pré-carregado).
        Deve ser chamado ao final de cada época ou quando desejado.
        """
        # Aguarda prefetch terminar (se ainda em andamento)
        if self._prefetch_thread is not None:
            self._prefetch_thread.join()
            self._prefetch_thread = None

        if self._next_visual is not None:
            # Remove os shards que já foram pré-carregados da fila
            self._shard_queue = self._shard_queue[self.shards_in_memory:]
            self._visual = self._next_visual
            self._text   = self._next_text
            self._next_visual = None
            self._next_text   = None
            print(f"\n[Buffer] Rotacionado — {len(self._visual):,} amostras em memória.")
        else:
            # Prefetch não tinha dados (fim de fila): recarrega do zero
            self._load_next_buffer_sync()

        # Dispara prefetch do próximo
        self._start_prefetch()

    # ------------------------------------------------------------------
    # Interface Dataset
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._visual)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self._visual[idx], self._text[idx]

def create_all_dataloaders( #USAR SE TIVER MEMÓRIA VRAM O SUFICIENTE
    root_dir, 
    batch_size, 
    image_size=256, 
    num_workers=8, 
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
    
   # 1. Definir os tamanhos das fatias desejadas
    train_size = int(500000)
    val_size   = int(10000)
    test_size  = int(10000)
    
    # 2. Calcular o resto (74% que não serão usados)
    unused_size = total_size - (train_size + val_size + test_size)
    
    # 3. Realizar o split com a lista completa que soma 100%
    generator = torch.Generator().manual_seed(42)
    
    # Criamos 4 divisões, mas ignoramos a última
    train_dataset, val_dataset, test_dataset, _ = random_split(
        full_dataset, 
        [train_size, val_size, test_size, unused_size],
        generator=generator
    )
    
    print(f"Divisão concluída:")
    print(f"  Treino: {len(train_dataset)} imagens")
    print(f"  Validação: {len(val_dataset)} imagens")
    print(f"  Teste: {len(test_dataset)} imagens")
    
    
    
    if t == "all":
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
        return train_loader, val_loader, test_loader
    elif t == "train":
        train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
    
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        return train_loader, val_loader
    elif t == "test":
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, 
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
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
        
