# SceneEmbed-UP
Este repositório implementa uma arquitetura avançada para Geração de Grafos de Cena (Scene Graphs), otimizada para imagens de ultra-alta resolução através da integração de encoders visuais e de embeddings. Objetivo é criar grafos que permitam a recuperação de imagens via texto (Image retrieval).

**Arquitetura**

O treinamento do sistema de alinhamento multimodal baseia-se em uma arquitetura de extração hierárquica e refinamento adaptativo, projetada para processar dados de alta fidelidade com eficiência computacional. O pipeline inicia-se com a extração de características visuais através do backbone **DINOv3** (https://arxiv.org/abs/2508.10104), que, integrado a um upsampler (**AnyUp** (https://arxiv.org/abs/2510.12764), **FeatUp** (https://arxiv.org/abs/2403.10516), **LoftUp** ), gera mapas de características locais de alta resolução e tokens globais semanticamente ricos. Simultaneamente, as descrições textuais são processadas pelo modelo **Qwen3-Embeddings** (https://arxiv.org/abs/2506.05176), resultando em vetores de texto de 4096 dimensões que servem como âncoras semânticas. Para viabilizar o processamento de mapas (tensores) de características densos $(768 \times 224 \times 224)$ (65K tokens) sem exceder os limites de memória de vídeo (VRAM), aplica-se uma camada de Adaptive Average Pooling sobre as características locais, reduzindo a dimensionalidade espacial para uma grade de $32 \times 32$ (1024 tokens) sem sacrificar a densidade de informação necessária para o alinhamento. O componente central, o LoRACrossAttentionAligner, opera através de uma projeção visual base congelada suplementada por camadas LoRA, que permitem o ajuste fino e eficiente dos parâmetros visuais para o domínio do texto. O mecanismo de Cross-Attention utiliza os embeddings do Qwen como query para interagir com as características visuais projetadas (key e value), forçando o modelo a filtrar e alinhar as informações visuais mais relevantes para cada termo textual. Todo o processo é otimizado através de uma perda contrastiva bidirecional (CLIP Loss), empregando precisão mista (BFloat16) para acelerar a convergência e um sistema de Early Stopping baseado na loss de validação para garantir a máxima generalização do modelo final.

![alt text](image.png)

![alt text](image-1.png)

![alt text](image-2.png)

![alt text](image-3.png)

![alt text](image-4.png)


**Dataset utilizado**

* COYO-700M (https://github.com/kakaobrain/coyo-dataset) subset com 7.794.790 de pares de imagens e captions.
    - 500k Treino
    - 10k Validação 
    - 10k Teste

**Frameworks**
 * Pytorch
 * image2dataset
 * Anaconda
 * HuggingFace models
