
import torch
import torch.nn as nn
import torch.nn.functional as F

class AnyUpModel:
    def __init__(self, repo = 'wimmerth/anyup', model =  'anyup_multi_backbone', natten = True, device = 'cuda'):
        self.natten = natten
        self.repo = repo
        self.model = model
        self.device = device
        self.torch_hub = torch.hub.load(self.repo, self.model, use_natten=self.natten).to(self.device)
        
    def up(self, img_tensor, lr_features):
        hr_features =  self.torch_hub(img_tensor, lr_features, output_size=(224, 224))
        return hr_features

class AdaptiveConv(torch.nn.Module):
    """
    Versão PyTorch do AdaptiveConv.
    Realiza convolução pixel-a-pixel usando kernels dinâmicos.
    """
    def __init__(self, radius):
        super().__init__()
        self.radius = radius
        self.diameter = 2 * radius + 1

    def forward(self, input, filters):
        """
        Args:
            input: [B, C, H_padded, W_padded] (Imagem de alta resolução com pad)
            filters: [B, H, W, D, D] (Kernels gerados pelo JBU)
        """
        B, C, H_pad, W_pad = input.shape
        B_f, H, W, D, D_f = filters.shape
        
        # 1. Extrair patches da imagem de entrada (Unfold)
        # O unfold transforma a imagem em uma sequência de blocos.
        # Resultado: [B, C * D * D, H * W]
        patches = F.unfold(input, kernel_size=self.diameter)
        
        # 2. Reshape dos patches para facilitar a multiplicação
        # [B, C, D*D, H, W]
        patches = patches.view(B, C, self.diameter * self.diameter, H, W)
        
        # 3. Reshape dos filtros (kernels dinâmicos)
        # [B, 1, D*D, H, W]
        filters = filters.permute(0, 3, 4, 1, 2).reshape(B, 1, self.diameter * self.diameter, H, W)
        
        # 4. Multiplicação Element-wise e Redução
        # Multiplicamos cada canal de cor pelo mesmo filtro espacial
        # O resultado é a soma ponderada dos pixels no patch pelo kernel do JBU
        result = (patches * filters).sum(dim=2) # Soma na dimensão do diâmetro (D*D)
        
        return result # [B, C, H, W]

class JBULearnedRange(torch.nn.Module):

    def __init__(self, guidance_dim, feat_dim, key_dim, scale=2, radius=3):
        super().__init__()
        self.scale = scale
        self.radius = radius
        self.diameter = self.radius * 2 + 1

        self.guidance_dim = guidance_dim
        self.key_dim = key_dim
        self.feat_dim = feat_dim

        self.range_temp = nn.Parameter(torch.tensor(0.0))
        self.range_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim, key_dim, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(key_dim, key_dim, 1, 1),
        )

        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Conv2d(guidance_dim + self.diameter ** 2, self.diameter ** 2, 1, 1),
            torch.nn.GELU(),
            torch.nn.Dropout2d(.1),
            torch.nn.Conv2d(self.diameter ** 2, self.diameter ** 2, 1, 1),
        )

        self.sigma_spatial = nn.Parameter(torch.tensor(1.0))

    def get_range_kernel(self, x):
        GB, GC, GH, GW = x.shape
        proj_x = self.range_proj(x)
        proj_x_padded = F.pad(proj_x, pad=[self.radius] * 4, mode='reflect')
        queries = torch.nn.Unfold(self.diameter)(proj_x_padded).reshape((GB, self.key_dim, self.diameter * self.diameter, GH, GW)).permute(0, 1, 3, 4, 2)
        pos_temp = self.range_temp.exp().clamp_min(1e-4).clamp_max(1e4)
        return F.softmax(pos_temp * torch.einsum("bchwp,bchw->bphw", queries, proj_x), dim=1)

    def get_spatial_kernel(self, device):
        dist_range = torch.linspace(-1, 1, self.diameter, device=device)
        x, y = torch.meshgrid(dist_range, dist_range)
        patch = torch.cat([x.unsqueeze(0), y.unsqueeze(0)], dim=0)
        return torch.exp(- patch.square().sum(0) / (2 * self.sigma_spatial ** 2)) \
            .reshape(1, self.diameter * self.diameter, 1, 1)

    def forward(self, source, guidance):
        GB, GC, GH, GW = guidance.shape
        SB, SC, SH, SQ = source.shape
        assert (SB == GB)

        spatial_kernel = self.get_spatial_kernel(source.device)
        range_kernel = self.get_range_kernel(guidance)

        combined_kernel = range_kernel * spatial_kernel
        combined_kernel /= combined_kernel.sum(1, keepdim=True).clamp(1e-7)

        combined_kernel += .1 * self.fixup_proj(torch.cat([combined_kernel, guidance], dim=1))
        combined_kernel = combined_kernel.permute(0, 2, 3, 1) \
            .reshape(GB, GH, GW, self.diameter, self.diameter)

        hr_source = torch.nn.Upsample((GH, GW), mode='bicubic', align_corners=False)(source)
        hr_source_padded = F.pad(hr_source, pad=[self.radius] * 4, mode='reflect')

    
        if not hasattr(self, 'adaptive_conv'):
            self.adaptive_conv = AdaptiveConv(radius=self.radius)
            
        result = self.adaptive_conv(hr_source_padded, combined_kernel)
        return result

class JBUStack(torch.nn.Module):

    def __init__(self, feat_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.up1 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up2 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up3 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.up4 = JBULearnedRange(3, feat_dim, 32, radius=3)
        self.fixup_proj = torch.nn.Sequential(
            torch.nn.Dropout2d(0.2),
            torch.nn.Conv2d(feat_dim, feat_dim, kernel_size=1))

    def upsample(self, source, guidance, up):
        _, _, h, w = source.shape
        small_guidance = F.adaptive_avg_pool2d(guidance, (h * 2, w * 2))
        upsampled = up(source, small_guidance)
        return upsampled

    def forward(self, source, guidance):
        source_2 = self.upsample(source, guidance, self.up1)
        source_4 = self.upsample(source_2, guidance, self.up2)
        source_8 = self.upsample(source_4, guidance, self.up3)
        source_16 = self.upsample(source_8, guidance, self.up4)
        return self.fixup_proj(source_16) * 0.1 + source_16
    
def load_featup_stack(checkpoint_path, feat_dim=384, device="cuda"):
    """
    Docstring for load_featup_stack
    
    :param checkpoint_path: caminho do modelo
    :param feat_dim: tamanho da feature inicial
    :param device: cuda ou cpu
    """
    #  Instancia a arquitetura. feat_dim deve ser 768 para o DINO ViT-B
    model = JBUStack(feat_dim=feat_dim).to(device)
        
    #  Carrega o checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
        
    # O FeatUp costuma salvar dentro de 'state_dict'
    state_dict = checkpoint.get('state_dict', checkpoint)
        
    # Limpeza de prefixos (comum se foi treinado com PyTorch Lightning)
    new_state_dict = {k.replace('upsampler.', ''): v for k, v in state_dict.items()}
        
    model.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model

