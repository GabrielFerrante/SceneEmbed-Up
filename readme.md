# SceneEmbed-UP
O SceneEmbed-UP é um mecanismo avançado de criação e refinamento de Grafos de Cena (Scene Graphs). O projeto integra embeddings multimodais (texto e imagem) e utiliza técnicas de upsampling (AnyUP) para aumentar a densidade e a fidelidade das representações latentes dos nós e das relações no grafo.

## Método
Fusão de embeddings via modelos como DINOV3 para imagens e Qwen3-Embeddings para representação semântica textual. Scene Graph Generation (SGG) para mapeamento de objetos, atributos e predicados relacionais. Latent Upsampling com AnyUP para melhorar a granularidade das representações de cena e Interoperabilidade para exportação de grafos para formatos compatíveis com Graph Neural Networks (GNNs).