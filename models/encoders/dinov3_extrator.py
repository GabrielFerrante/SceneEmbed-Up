
import torch
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login
import torchvision.transforms as T
from transformers.image_utils import load_image

class DinoSceneEncoder:
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", token_path='tokenDINOV3.json', device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(token_path, 'r', encoding='utf-8') as file:
            dados_auth = json.load(file)
        login(dados_auth['token'])

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="cuda").to(self.device)
        
        # AnyUp
        self.upsampler = torch.hub.load('wimmerth/anyup', 'anyup_multi_backbone', use_natten=True).to(self.device)
        
        self.model.eval()
        self.upsampler.eval()

        # Transformação para garantir que a imagem HR tenha a mesma normalização do processor
        self.hr_transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=self.processor.image_mean, std=self.processor.image_std)
        ])

    @torch.inference_mode()
    def extract_features(self, image: Image.Image):
        # 1. Preparar imagem HR com a normalização correta do modelo
        img_tensor = self.hr_transform(image).unsqueeze(0).to(self.device)
        
        # 2. Processar pelo DINO
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        
        last_hidden_state = outputs.last_hidden_state 
        B, N_total, C = last_hidden_state.shape
        
        cls_token = last_hidden_state[:, 0, :]
        
        # Cálculo dinâmico da grade
        h_feat = inputs['pixel_values'].shape[-2] // 16
        w_feat = inputs['pixel_values'].shape[-1] // 16
        n_spatial = h_feat * w_feat
        
        # Seleciona patches espaciais descartando CLS e potenciais Registers
        spatial_tokens = last_hidden_state[:, 1:n_spatial+1, :]
        
        # 3. Reshape para Grid 2D (B, C, H, W)
        lr_features = spatial_tokens.transpose(1, 2).reshape(B, C, h_feat, w_feat)
        
        # 4. Upsampling AnyUp
        hr_features = self.upsampler(img_tensor, lr_features)
        
        return cls_token, hr_features

# Exemplo de uso:
# from dino_module import DinoSceneEncoder
encoder = DinoSceneEncoder()

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
image = load_image(url)

cls_feat, spatial_feats = encoder.extract_features(image)
print(f"global:{cls_feat.shape} ")
print(f"local: {spatial_feats.shape}")


"""
SAIDA VISTA

global:torch.Size([1, 768])
local: torch.Size([1, 768, 686, 960])
"""
