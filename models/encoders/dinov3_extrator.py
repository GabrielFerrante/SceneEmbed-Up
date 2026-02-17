
import torch
import torch.nn as nn
import json
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import login
import torchvision.transforms as T
from transformers.image_utils import load_image
import sys
sys.path.append('..')
from models.ups.hr_conversions import AnyUpModel, load_featup_stack
"""
import hr_conversions as hr
""" #USAR COM FEATUP



class DinoSceneEncoder:
    def __init__(self, model_name="facebook/dinov3-vitb16-pretrain-lvd1689m", token_path='C:/Repositorios/SceneEmbed-Up/models/encoders/tokenDINOV3.json', 
                 device=None, upsampler = 'anyup'):
        
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        with open(token_path, 'r', encoding='utf-8') as file:
            dados_auth = json.load(file)
        login(dados_auth['token'])

        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="cuda").to(self.device)
        
        # Upsampler
        self.upsampler = upsampler
        self.model_up = upsampler
        self.adapterFeat = None
        
        if self.upsampler == "anyup":
       
            self.upsampler =  AnyUpModel()
            
        elif self.upsampler == "featup":
            self.upsampler =  load_featup_stack("C:/Repositorios/SceneEmbed-Up/models/encoders/vit_jbu_stack_cocostuff.ckpt", feat_dim=384)
            self.adapterFeat = nn.Conv2d(768, 384, kernel_size=1).to(self.device)
            
        self.model.eval()
        


    @torch.no_grad()
    def extract_features(self, img_tensor):
        
        # Processar pelo DINO
        inputs = self.processor(images=img_tensor, return_tensors="pt").to(self.device)
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
                
        #  Reshape para Grid 2D (B, C, H, W)
        lr_features = spatial_tokens.transpose(1, 2).reshape(B, C, h_feat, w_feat)
        hr_features = None
        if self.model_up == "anyup":
            hr_features = self.upsampler.up(img_tensor, lr_features)
        elif self.model_up == "featup":
            hr_features = self.upsampler(self.adapterFeat(lr_features), img_tensor)
                    
        return cls_token,  hr_features

# Exemplo de uso:

#encoder = DinoSceneEncoder()

#url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
#image = load_image(url)

#cls_feat, spatial_feats = encoder.extract_features(image)
#print(f"global:{cls_feat.shape} ")
#print(f"local: {spatial_feats.shape}")


"""
SAIDA VISTA
PARA O ANYUP

global:torch.Size([1, 768])
local: torch.Size([1, 768, 686, 960])

PARA O FEATUP
global:torch.Size([1, 768]) 
local: torch.Size([1, 384, 224, 224])
"""
