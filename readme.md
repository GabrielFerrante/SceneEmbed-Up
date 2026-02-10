# SceneEmbed-UP
Este repositório implementa uma arquitetura avançada para Geração de Grafos de Cena (Scene Graphs), otimizada para imagens de ultra-alta resolução através da integração de encoders visuais de última geração e LLMs de embedding.

## Método
A solução resolve o desafio de processar detalhes minúsculos em imagens grandes sem estourar a memória de vídeo (VRAM), utilizando uma técnica de Projeção via Cross-Attention com Adaptadores LoRA.

**Arquitetura**

* Visual Backbone (DINOv3): Utilizamos o DINOv3 para extração de features ricas. Aplicamos os upsamplers AnyUp e FeatUp para elevar a resolução dos mapas de features. Isso permite que o modelo "enxergue" objetos que desapareceriam em resoluções padrão (256x256).

* Semantic Embedding (Qwen3-8B): O Qwen3-Embedding é utilizado para converter labels de objetos e relações em vetores semânticos de alta dimensão ($4096$). As descrições são processadas com instruções específicas para tarefas de visão computacional.

* LoRA Cross-Attention Aligner: Em vez de uma projeção linear simples, implementamos uma camada de Cross-Attention.

    - Mecânica: O texto (Query) busca informações nos patches da imagem (Key/Value). Isso reduz a complexidade computacional, focando apenas nos pixels relevantes para cada objeto do grafo.

    - Eficiência: Utilizamos LoRA (Low-Rank Adaptation) para treinar as matrizes de projeção, permitindo um ajuste fino leve e rápido, mantendo os modelos base congelados.

**Dataset**

* COYO-700M como dataset de treino, validação e teste (70% | 20% | 10%)

**Tecnologias**
 * Pytorch
 * image2dataset
 * Python
 * Anaconda
 * HuggingFace models
