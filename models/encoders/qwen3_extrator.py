import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class QwenSceneEmbedder:
    def __init__(self, model_id='Qwen/Qwen3-Embedding-8B', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Usamos bfloat16 se a GPU suportar (A100, H100, RTX 30/40), senão float16
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if "cpu" in self.device: self.dtype = torch.float32

        print(f"Loading Qwen3 Embedding model on {self.device} em {self.dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=self.dtype,
            device_map="cuda"
        ).to(self.device)
        
        self.model.eval()
        self.task_sg = 'Extract semantic features for visual objects and relationships in a scene'

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        # Mantive sua lógica de pooling que está correta
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.inference_mode()
    def embed_components(self, batch_texts: list[list[str]], max_length=512, normalize=True) -> Tensor:
        """
        Args:
            batch_texts: Lista de listas, ex: [['gato', 'mesa'], ['cachorro', 'bola']]
        Returns:
            Tensor: [Batch, N_textos, 4096]
        """
        batch_size = len(batch_texts)
        n_texts_per_batch = len(batch_texts[0]) # Assume que cada imagem tem o mesmo N de termos no batch
        
        # 1. Achata a lista de listas para processar tudo em um único forward do Qwen
        flat_texts = [t for sublist in batch_texts for t in sublist]
        instructed_texts = [f'Instruct: {self.task_sg}\nQuery:{t}' for t in flat_texts]
        
        # 2. Tokenização em massa
        batch_dict = self.tokenizer(
            instructed_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(self.device)

        # 3. Forward
        outputs = self.model(**batch_dict)
        
        # 4. Pooling
        embeddings = self._last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)
            
        # 5. RESHAPE: De [Batch*N_textos, 4096] para [Batch, N_textos, 4096]
        return embeddings.view(batch_size, n_texts_per_batch, -1)
    
    
# Exemplo de uso em outro arquivo:
# from qwen_module import QwenSceneEmbedder
embedder = QwenSceneEmbedder()
vecs = embedder.embed_components(['black cat', 'on top of'])
print(vecs.shape)