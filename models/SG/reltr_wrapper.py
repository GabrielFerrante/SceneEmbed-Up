"""
reltr_wrapper.py
----------------
Wrapper para o modelo RelTR pré-treinado (https://github.com/yrcong/RelTR).

RelTR é um detector de scene graphs end-to-end baseado em DETR. Recebe uma
imagem e retorna triplas (sujeito, predicado, objeto) com bounding boxes.

Instalação dos pesos pré-treinados:
    mkdir -p checkpoints/reltr
    # Baixar de: https://drive.google.com/file/d/1id6oD_iwiNDD6HyCn2ORgRTIKkPD3tUD
    # Salvar em: checkpoints/reltr/reltr_vg.pth

Instalação do repositório RelTR:
    git clone https://github.com/yrcong/RelTR.git reltr_repo
    (não precisa instalar como pacote — usamos sys.path)
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


@dataclass
class SceneGraphTriple:
    subject: str
    predicate: str
    object: str
    subject_box: Tuple[float, float, float, float]   # (x1, y1, x2, y2) normalized
    object_box: Tuple[float, float, float, float]
    confidence: float


@dataclass
class SceneGraph:
    triples: List[SceneGraphTriple]
    image_width: int
    image_height: int

    def to_networkx(self):
        """Converte para grafo NetworkX (nodes=entidades, edges=relações)."""
        import networkx as nx
        G = nx.DiGraph()
        for t in self.triples:
            G.add_node(t.subject)
            G.add_node(t.object)
            G.add_edge(t.subject, t.object, label=t.predicate, confidence=t.confidence)
        return G


_TRANSFORM = transforms.Compose([
    transforms.Resize(800),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# VG-150 entity classes — igual ao CLASSES do inference.py oficial do RelTR.
# 151 entradas: N/A no índice 0, zebra no índice 150.
# O modelo tem 151 logits; [:, :-1] remove o último (background extra),
# deixando índices 0..149 que mapeiam direto para este vocabulário.
_VG_ENTITIES = [
    "N/A", "airplane", "animal", "arm", "bag", "banana", "basket",
    "beach", "bear", "bed", "bench", "bike", "bird", "board", "boat", "book",
    "boot", "bottle", "bowl", "box", "boy", "branch", "building", "bus", "cabinet",
    "cap", "car", "cat", "chair", "child", "clock", "coat", "counter", "cow",
    "cup", "curtain", "desk", "dog", "door", "drawer", "ear", "elephant", "engine",
    "eye", "face", "fence", "finger", "flag", "flower", "food", "fork", "fruit",
    "giraffe", "girl", "glass", "glove", "guy", "hair", "hand", "handle", "hat",
    "head", "helmet", "hill", "horse", "house", "jacket", "jean", "kid", "kite",
    "lady", "lamp", "laptop", "leaf", "leg", "letter", "light", "logo", "man",
    "men", "motorcycle", "mountain", "mouth", "neck", "nose", "number", "orange",
    "pant", "paper", "paw", "people", "person", "phone", "pillow", "pizza",
    "plane", "plant", "plate", "player", "pole", "post", "pot", "racket", "railing",
    "rock", "roof", "room", "screen", "seat", "sheep", "shelf", "shirt", "shoe",
    "short", "sidewalk", "sign", "sink", "skateboard", "ski", "skier", "sneaker",
    "snow", "sock", "stand", "street", "stripe", "surfboard", "table", "tail",
    "tie", "tile", "tire", "toilet", "towel", "tower", "track", "train", "tree",
    "truck", "trunk", "umbrella", "vase", "vegetable", "vehicle", "wave", "wheel",
    "window", "windshield", "wing", "wire", "woman", "zebra",
]

# VG-150 predicate classes — igual ao REL_CLASSES do inference.py oficial.
# O background é o índice 0 e é usado para filtrar; argmax direto sobre os 51 logits.
_VG_PREDICATES = [
    "__background__", "above", "across", "against", "along", "and", "at",
    "attached to", "behind", "belonging to", "between", "carrying", "covered in",
    "covering", "eating", "flying in", "for", "from", "growing on", "hanging from",
    "has", "holding", "in", "in front of", "laying on", "looking at", "lying on",
    "made of", "mounted on", "near", "of", "on", "on back of", "over", "painted on",
    "parked on", "part of", "playing", "riding", "says", "sitting on", "standing on",
    "to", "under", "using", "walking in", "walking on", "watching", "wearing",
    "wears", "with",
]


class RelTRWrapper:
    """
    Wrapper leve sobre o RelTR para geração de scene graphs.

    Parameters
    ----------
    checkpoint:
        Caminho para o arquivo .pth com os pesos pré-treinados do RelTR.
    reltr_repo:
        Caminho para o repositório clonado do RelTR (adicionado ao sys.path).
    device:
        'cuda' ou 'cpu'.
    num_queries:
        Número de triplas a retornar (top-K por confiança). Default: 20.
    threshold:
        Score mínimo para incluir uma tripla. Default: 0.3.
    """

    def __init__(
        self,
        checkpoint: str = "checkpoints/reltr/reltr_vg.pth",
        reltr_repo: str = "reltr_repo",
        device: str = "cuda",
        num_queries: int = 20,
        threshold: float = 0.3,
    ) -> None:
        self.device = device
        self.num_queries = num_queries
        self.threshold = threshold
        self._reltr_repo = os.path.abspath(reltr_repo)

        self.model = self._weights_only(checkpoint)

    def _weights_only(self, checkpoint: str):
        import argparse

        # Salva estado atual de sys.path e módulos conflitantes
        _saved_path = sys.path.copy()
        _saved_mods = {
            k: sys.modules.pop(k)
            for k in list(sys.modules.keys())
            if k == "models" or k.startswith("models.")
        }

        # Coloca reltr_repo na frente — "models" e "util" resolvem para o RelTR
        sys.path.insert(0, self._reltr_repo)

        try:
            from models import build_model  # type: ignore  (reltr_repo/models/__init__.py)
            from util.misc import nested_tensor_from_tensor_list  # type: ignore

            args = argparse.Namespace(
                # Arquitetura
                enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=256,
                dropout=0.1, nheads=8, num_entities=100, num_triplets=200,
                pre_norm=False, position_embedding="sine", backbone="resnet50",
                dilation=False, masks=False, return_interm_layers=False,
                # Dataset
                dataset="vg", ann_path="./data/vg/", img_folder="",
                # Treino (irrelevantes em inferência, mas exigidos pelo build)
                lr_backbone=0,
                aux_loss=False,
                bbox_loss_coef=5, giou_loss_coef=2, rel_loss_coef=1,
                eos_coef=0.1,
                set_cost_class=1, set_cost_bbox=5, set_cost_giou=2,
                set_iou_threshold=0.7,
                # Device
                device=self.device,
            )
            model, _, _ = build_model(args)

        except ImportError as e:
            raise ImportError(
                "RelTR não encontrado. Clone o repositório com:\n"
                "  git clone https://github.com/yrcong/RelTR.git reltr_repo\n"
                f"Erro original: {e}"
            )
        finally:
            # Restaura sys.path
            sys.path[:] = _saved_path
            # Remove módulos RelTR do cache para não poluir imports futuros
            for key in list(sys.modules.keys()):
                if key in ("models", "util") or key.startswith("models.") or key.startswith("util."):
                    sys.modules.pop(key, None)
            # Restaura os nossos módulos
            sys.modules.update(_saved_mods)

        self._nested_tensor = nested_tensor_from_tensor_list

        if os.path.exists(checkpoint):
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            model.load_state_dict(state["model"], strict=False)
            print(f"[RelTR] Pesos carregados de {checkpoint}")
        else:
            print(f"[RelTR] AVISO: checkpoint não encontrado em {checkpoint}. Usando pesos aleatórios.")

        model.to(self.device).eval()
        return model

    @torch.no_grad()
    def predict(self, image: Image.Image) -> SceneGraph:
        """
        Gera um scene graph a partir de uma imagem PIL.

        Parameters
        ----------
        image:
            Imagem PIL RGB.

        Returns
        -------
        SceneGraph
            Objeto com lista de triplas (sujeito, predicado, objeto) + bboxes.
        """
        w, h = image.size
        img_t = _TRANSFORM(image.convert("RGB")).unsqueeze(0).to(self.device)
        samples = self._nested_tensor([img_t[0]])

        outputs = self.model(samples)

        # Scores de entidades e triplas
        sub_logits = outputs["sub_logits"][0]    # [num_triplets, num_entity_classes]
        obj_logits = outputs["obj_logits"][0]
        rel_logits = outputs["rel_logits"][0]    # [num_triplets, num_predicate_classes]
        sub_boxes  = outputs["sub_boxes"][0]     # [num_triplets, 4] — cx,cy,w,h normalized
        obj_boxes  = outputs["obj_boxes"][0]

        # Igual ao inference.py oficial: remove o último logit (N/A) em entidades,
        # e remove o último logit em relações (background está no índice 0 mas
        # o modelo usa :-1 para o keep — seguimos o mesmo padrão).
        probas_sub = sub_logits.softmax(-1)[:, :-1]   # [T, 150]
        probas_obj = obj_logits.softmax(-1)[:, :-1]   # [T, 150]
        probas_rel = rel_logits.softmax(-1)[:, :-1]   # [T, 50]

        sub_conf, sub_idx = probas_sub.max(-1)
        obj_conf, obj_idx = probas_obj.max(-1)
        rel_conf, rel_idx = probas_rel.max(-1)

        # Filtra triplas onde todas as três partes superam o threshold
        keep = (sub_conf > self.threshold) & (obj_conf > self.threshold) & (rel_conf > self.threshold)
        keep_ids = torch.where(keep)[0]

        # Ordena por confiança combinada e pega top-K
        if keep_ids.numel() == 0:
            return SceneGraph(triples=[], image_width=w, image_height=h)

        combined = sub_conf[keep_ids] * rel_conf[keep_ids] * obj_conf[keep_ids]
        order = combined.argsort(descending=True)[:self.num_queries]
        keep_ids = keep_ids[order]

        triples: List[SceneGraphTriple] = []
        for tid in keep_ids.tolist():
            score = (sub_conf[tid] * rel_conf[tid] * obj_conf[tid]).item()

            s_name = _VG_ENTITIES[sub_idx[tid].item()]
            o_name = _VG_ENTITIES[obj_idx[tid].item()]
            p_name = _VG_PREDICATES[rel_idx[tid].item()]

            def cx_cy_to_xyxy(box):
                cx, cy, bw, bh = box
                return (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)

            triples.append(SceneGraphTriple(
                subject=s_name,
                predicate=p_name,
                object=o_name,
                subject_box=cx_cy_to_xyxy(sub_boxes[tid].tolist()),
                object_box=cx_cy_to_xyxy(obj_boxes[tid].tolist()),
                confidence=score,
            ))

        return SceneGraph(triples=triples, image_width=w, image_height=h)
