
import torch

class AnyUpModel:
    def __init__(self, repo = 'wimmerth/anyup', model =  'anyup_multi_backbone', natten = True, device = 'cuda'):
        self.natten = natten
        self.repo = repo
        self.model = model
        self.device = device
        
    def up(self, img_tensor, lr_features):
        anyup = torch.hub.load(self.repo, self.model, use_natten=self.natten).to(self.device)
        hr_features =  anyup(img_tensor, lr_features)
        return hr_features
    

class FeatUpModel: #TEM QUE MUDAR E COLOCAR PRA LER O MODELO E EXECUTAR
    def __init__(self, repo ="mhamilton723/FeatUp", model = 'vit',  use_norme = False, device = 'cuda'):
        self.repo = repo
        self.model = model
        self.device = device
        self.norme = use_norme
        
    def up(self, lr_features):
        featup = torch.hub.load(self.repo, self.model, use_norm=self.norme).to(self.device)
        hr_features = featup(lr_features)
        return hr_features

class FeatSharpModel: ## ESPERANDO LANÇAMENTO DOS MODELOS
    def __init__(self):
        pass