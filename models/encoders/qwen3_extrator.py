import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

class QwenSceneEmbedder:
    def __init__(self, model_id='Qwen/Qwen3-Embedding-8B', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.dtype = torch.bfloat16

        print(f"Loading Qwen3 Embedding model on {self.device} em {self.dtype}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
        self.model = AutoModel.from_pretrained(
            model_id, 
            dtype=self.dtype,
            device_map="cuda"
        ).to(self.device)
        
        self.model.eval()
        self.task_sg = 'Extract semantic features for visual objects and relationships in a scene'

    def _last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
       
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    @torch.no_grad()
    def embed_components(self, batch_texts: list[list[str]] | list[str], max_length=512, normalize=True) -> Tensor:
        """
        Returns: Tensor [Batch, N_textos, 4096]
        """
        # 1. Normalização de Entrada: Garante que seja sempre lista de listas
        if isinstance(batch_texts[0], str):
            batch_texts = [batch_texts] # Transforma ['a', 'b'] em [['a', 'b']]
            
        batch_size = len(batch_texts)
        n_texts_per_batch = len(batch_texts[0]) 
        
        # 2. Achata para processamento em massa
        flat_texts = [t for sublist in batch_texts for t in sublist]
        instructed_texts = [f'Instruct: {self.task_sg}\nQuery:{t}' for t in flat_texts]
        
        # 3. Tokenização e Forward
        batch_dict = self.tokenizer(
            instructed_texts, 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**batch_dict)
            
        # 4. Pooling (Usando a sua lógica de Mean Pooling)
        embeddings = outputs.last_hidden_state
        mask_expanded = batch_dict['attention_mask'].unsqueeze(-1).expand_as(embeddings).float()
        
        sum_embeddings = torch.sum(embeddings * mask_expanded, dim=1)
        sum_mask = torch.clamp(torch.sum(mask_expanded, dim=1), min=1e-9)
        mean_pooled = sum_embeddings / sum_mask
        
        # 5. RESHAPE CRUCIAL: Volta para [B, N, 4096]
        # Isso separa os textos de volta para suas respectivas imagens
        final_embeddings = mean_pooled.view(batch_size, n_texts_per_batch, -1)
        
        if normalize:
            final_embeddings = torch.nn.functional.normalize(final_embeddings, p=2, dim=-1)
            
        return final_embeddings.float()
        
    
# Exemplo de uso em outro arquivo:
#from qwen_module import QwenSceneEmbedder
#embedder = QwenSceneEmbedder()
#vecs = embedder.embed_components(['black cat', 'on top of'])
#print(vecs.shape)

"""
torch.size([2,9,4096])
"""