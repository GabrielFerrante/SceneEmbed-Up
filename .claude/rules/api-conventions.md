# Convencoes de API

- Encoders (DinoSceneEncoder, QwenSceneEmbedder) sao classes com metodos `extract_features` / `embed_components`
- Encoders sao sempre congelados (sem grad) durante treino do Aligner
- Aligner retorna tupla: `(attn_output, attn_weights, v_features)`
- SceneGraphGenerator.generate() retorna dict com chaves `nodes` e `edges`
- KnowledgeGraphGenerator.generate_from_scene() recebe scene_graph dict e retorna knowledge_graph dict
- Shards H5 usam datasets: `visual_feats`, `text_feats`, `visual_global`
