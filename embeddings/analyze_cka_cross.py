"""
analyze_cka_cross.py
--------------------
Centered Kernel Alignment (CKA) entre embeddings visuais e textuais.

Usa a biblioteca `ckatorch` (https://github.com/RistoAle97/centered-kernel-alignment),
que implementa CKA em PyTorch com suporte a minibatch HSIC.

CKA mede aderência geométrica entre dois espaços de representação mesmo
quando suas dimensões nativas são diferentes (768 vs 4096). Olha para as
relações entre N amostras, não para as dimensões de cada amostra.

Comparações executadas (texto é único — Qwen é determinístico):
  1. Visual_global (anyup) ↔ Text                  — alinhamento bruto com upsampling
  2. Visual_global (noup)  ↔ Text                  — alinhamento bruto sem upsampling
  3. Visual_global (anyup) ↔ Visual_global (noup)  — preservação geométrica do CLS

Instalação da dependência:
    pip install ckatorch

Uso:
    python embeddings/analyze_cka_cross.py
    python embeddings/analyze_cka_cross.py --n_samples 5000 --kernel linear
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Tuple

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# ckatorch — biblioteca oficial de CKA em PyTorch
try:
    from ckatorch import CKA
except ImportError as e:
    raise ImportError(
        "ckatorch nao encontrado. Instale com:\n"
        "  pip install ckatorch\n"
        f"Erro original: {e}"
    )

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.io_utils import ensure_dir


# ---------------------------------------------------------------------------
# Wrapper de chamada robusto à variação de API
# ---------------------------------------------------------------------------

class _PairedFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset que devolve dict {"x": [chunk, Dx], "y": [chunk, Dy]} por índice.

    A ckatorch.CKA espera ativações 3D [B, T, D] (torch.bmm interno) com T>1
    para evitar variância zero no HSIC. Como temos embeddings globais [N, D],
    agrupamos `chunk_size` amostras em cada item: cada índice devolve um
    "batch de tokens" pronto pra ser consumido como pseudo-transformer.

    Após DataLoader, batches ficam [B_loader, chunk, D] — ckatorch faz bmm
    sobre o eixo do meio (T = chunk) e o HSIC tem material para centralizar.
    """

    def __init__(self, X: torch.Tensor, Y: torch.Tensor, chunk_size: int = 64) -> None:
        assert X.shape[0] == Y.shape[0], "X e Y devem ter mesmo N"
        N = X.shape[0]
        # Trunca para múltiplo de chunk_size
        usable = (N // chunk_size) * chunk_size
        if usable < N:
            X = X[:usable]
            Y = Y[:usable]
        # Reshape para [N/chunk_size, chunk_size, D]
        self.X = X.reshape(-1, chunk_size, X.shape[-1])
        self.Y = Y.reshape(-1, chunk_size, Y.shape[-1])

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return {"x": self.X[idx], "y": self.Y[idx]}


class _FeatureCarrier(torch.nn.Module):
    """
    Modelo Identity-wrapper que devolve a feature da chave configurada.

    A ckatorch chama `model(**batch)` desempacotando o dict como kwargs.
    Cada batch chega como [B_loader, chunk, D] — formato 3D que a CKA
    consome via torch.bmm internamente.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self.key = key
        self.layer = torch.nn.Identity()

    def forward(self, **kwargs):
        return self.layer(kwargs[self.key])


def _cka_score(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",   # noqa: ARG001 — ckatorch usa linear por padrão; mantido p/ compat de API
    device: str = "cuda",
    batch_size: int = 8,
    chunk_size: int = 64,
    epochs: int = 10,
) -> float:
    """
    Calcula CKA(X, Y) usando ckatorch.CKA com dois modelos Identity-wrapper.

    A ckatorch.CKA exige (first_model, second_model, layers). Para tensores
    já extraídos, criamos modelos dummy que devolvem as próprias features
    via uma camada Identity. CKA registra hook nessa camada e calcula HSIC.

    Parameters
    ----------
    X, Y : np.ndarray  com shape [N, Dx] e [N, Dy].
    kernel : "linear" ou "rbf"  (mantido por compatibilidade; ckatorch usa kernel linear).
    device : torch device.

    Returns
    -------
    float em [0, 1]
    """
    _ = kernel  # ckatorch v0.x não expõe parâmetro de kernel pública na CKA principal
    Xt = torch.from_numpy(X).float()
    Yt = torch.from_numpy(Y).float()

    model_x = _FeatureCarrier(key="x").to(device).eval()
    model_y = _FeatureCarrier(key="y").to(device).eval()

    # DataLoader com dict[str, Tensor] — exigido pela ckatorch.CKA
    # chunk_size agrupa amostras em pseudo-tokens p/ HSIC ter material
    dataset = _PairedFeatureDataset(Xt, Yt, chunk_size=chunk_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # Tenta a assinatura padrão de ckatorch.CKA
    try:
        cka = CKA(
            first_model=model_x,
            second_model=model_y,
            layers=["layer"],          # nome da Identity em ambos modelos
            first_name="anyup",
            second_name="noup",
            device=device,
        )
    except TypeError:
        # Versões mais antigas/diferentes podem não ter first_name/second_name
        cka = CKA(
            first_model=model_x,
            second_model=model_y,
            layers=["layer"],
            device=device,
        )

    # Executa CKA. A assinatura interna do ckatorch.CKA.__call__ é
    # algo como `__call__(self, dataloader, epochs)` — epochs é o
    # SEGUNDO posicional, não-keyword. Passamos posicionalmente.
    with torch.no_grad():
        result = cka(loader, epochs)

    # Resultado costuma vir como tensor/matriz [n_layers_x, n_layers_y]
    if torch.is_tensor(result):
        if result.numel() == 1:
            return float(result.item())
        # Matriz — pega o [0, 0] que é a comparação layer↔layer
        return float(result[0, 0].item())
    if isinstance(result, np.ndarray):
        return float(result.flat[0])
    return float(result)


# ---------------------------------------------------------------------------
# Carregamento dos shards
# ---------------------------------------------------------------------------

def load_embeddings(
    folder: str,
    n_samples: int,
    keys: Tuple[str, ...] = ("visual_global", "text_feats"),
) -> dict:
    """
    Carrega N amostras emparelhadas de visual_global + text_feats.

    Returns
    -------
    dict com:
        visual_global: [N, 768]  float32
        text_feats:    [N, 4096] float32 (squeezed se vier [N, 1, 4096])
    """
    pattern = os.path.join(folder, "**", "*.h5")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        raise FileNotFoundError(f"Nenhum shard em {folder}")

    chunks = {k: [] for k in keys}
    collected = 0

    for f in files:
        if collected >= n_samples:
            break
        try:
            with h5py.File(f, "r") as h5:
                missing = [k for k in keys if k not in h5]
                if missing:
                    print(f"  [WARN] {os.path.basename(f)} sem {missing} — pulado")
                    continue

                n_avail = h5[keys[0]].shape[0]
                take = min(n_avail, n_samples - collected)

                for k in keys:
                    arr = h5[k][:take].astype(np.float32)
                    if k == "text_feats" and arr.ndim == 3:
                        arr = arr.squeeze(1)
                    chunks[k].append(arr)
                collected += take
        except Exception as e:
            print(f"  [WARN] Falha ao ler {f}: {e}")

    out = {k: np.concatenate(chunks[k], axis=0)[:n_samples] for k in keys}
    return out


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_cka_matrix(results: dict, save_path: str) -> None:
    """Barplot horizontal das CKAs computadas."""
    labels = list(results.keys())
    values = [results[k] for k in labels]

    fig, ax = plt.subplots(figsize=(10, max(3, len(labels) * 0.6)))
    bars = ax.barh(labels, values, color="#4A90D9", alpha=0.85)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("CKA (ckatorch)")
    ax.set_title("Centered Kernel Alignment")
    ax.axvline(0.5, color="gray", linestyle="--", lw=1, alpha=0.5, label="strong alignment >= 0.5")
    ax.grid(alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    ensure_dir(os.path.dirname(save_path))
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [viz] Saved to {save_path}")


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def run_cka_analysis(
    folder_anyup: str,
    folder_noup: str,
    n_samples: int,
    output_dir: str,
    split: str,
    kernel: str = "linear",
    device: str = "cuda",
) -> dict:
    """
    Computa CKA via ckatorch entre os pares relevantes:
      - visual_anyup ↔ text                  (alinhamento bruto com AnyUp)
      - visual_noup  ↔ text                  (alinhamento bruto sem AnyUp)
      - visual_anyup ↔ visual_noup           (preservação geométrica do CLS)
    """
    print(f"\n{'═' * 70}")
    print(f"  CKA (ckatorch)  —  split={split}  kernel={kernel}  N={n_samples}")
    print(f"{'═' * 70}")

    print(f"\n[1/3] Loading embeddings...")
    data_anyup = load_embeddings(folder_anyup, n_samples)
    data_noup  = load_embeddings(folder_noup,  n_samples)

    v_anyup = data_anyup["visual_global"]
    v_noup  = data_noup["visual_global"]
    t_anyup = data_anyup["text_feats"]
    t_noup  = data_noup["text_feats"]

    n = min(v_anyup.shape[0], v_noup.shape[0], t_anyup.shape[0], t_noup.shape[0])
    v_anyup, v_noup = v_anyup[:n], v_noup[:n]
    t_anyup, t_noup = t_anyup[:n], t_noup[:n]

    print(f"  visual_anyup: {v_anyup.shape}")
    print(f"  visual_noup : {v_noup.shape}")
    print(f"  text_anyup  : {t_anyup.shape}")
    print(f"  text_noup   : {t_noup.shape}")
    print(f"  Effective N : {n}")

    # Sanity check: textos devem ser idênticos (mesmo caption → mesmo Qwen)
    diff = float(np.linalg.norm(t_anyup - t_noup) / np.linalg.norm(t_anyup))
    print(f"\n  [sanity] ||t_anyup - t_noup|| / ||t_anyup|| = {diff:.6e}")
    if diff < 1e-4:
        print(f"  OK: Texts are identical -- shards aligned per sample. Using t_anyup.")
        t = t_anyup
    else:
        print(f"  [WARN] Texts diverge by {diff:.4f} -- shards NOT aligned per sample!")
        print(f"         Splits were shuffled in different orders during extraction.")
        print(f"         visual<->text CKA will be biased.")
        t = t_anyup

    print(f"\n[2/3] Computing CKAs via ckatorch ({kernel})...")

    pairs = {
        "visual_anyup <-> text":         (v_anyup, t),
        "visual_noup  <-> text":         (v_noup,  t),
        "visual_anyup <-> visual_noup":  (v_anyup, v_noup),
    }

    results: dict = {}
    for name, (X, Y) in pairs.items():
        score = _cka_score(X, Y, kernel=kernel, device=device)
        results[name] = score
        print(f"  {name:<32}  CKA = {score:.6f}")

    # Interpretação
    print(f"\n{'-' * 70}")
    print(f"  INTERPRETATION")
    print(f"{'-' * 70}")

    cka_v_cross = results["visual_anyup <-> visual_noup"]
    if cka_v_cross >= 0.95:
        v_msg = "Visual_global is PRACTICALLY IDENTICAL between the two extractions"
    elif cka_v_cross >= 0.85:
        v_msg = "Visual_global has HIGH alignment with small differences"
    elif cka_v_cross >= 0.70:
        v_msg = "Visual_global has MODERATE alignment -- AnyUp visibly alters the CLS"
    else:
        v_msg = "Visual_global DIVERGES substantially between extractions"
    print(f"  visual_anyup <-> visual_noup = {cka_v_cross:.4f}  ->  {v_msg}")

    a_vt = results["visual_anyup <-> text"]
    n_vt = results["visual_noup  <-> text"]
    delta = n_vt - a_vt
    if abs(delta) < 0.01:
        align_msg = "RAW visual<->text alignment is practically the same"
    elif delta > 0:
        align_msg = (f"No-AnyUp has higher RAW alignment by +{delta:.4f} "
                     f"-> confirms that AnyUp introduces noise")
    else:
        align_msg = (f"AnyUp has higher RAW alignment by +{abs(delta):.4f} "
                     f"-> upsampling helps in the raw space")
    print(f"  Delta alignment (noup - anyup) = {delta:+.4f}  ->  {align_msg}")

    # Salva relatório
    report = {
        "split": split,
        "kernel": kernel,
        "device": device,
        "library": "ckatorch",
        "n_samples": int(n),
        "folder_anyup": folder_anyup,
        "folder_noup": folder_noup,
        "cka": results,
    }

    ensure_dir(output_dir)
    json_path = os.path.join(output_dir, f"cka_{split}_{kernel}.json")
    with open(json_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)
    print(f"\n  [report] saved to {json_path}")

    png_path = os.path.join(output_dir, f"cka_{split}_{kernel}.png")
    plot_cka_matrix(results, png_path)

    return report


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CKA via ckatorch entre visual e text embeddings")
    parser.add_argument("--n_samples", type=int, default=50000)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--kernel", choices=["linear", "rbf"], default="rbf",
                        help="Tipo de kernel para Gram matrices (default: linear)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--folder_anyup", type=str, default=None)
    parser.add_argument("--folder_noup", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results/cka")
    args = parser.parse_args()

    defaults = {
        "train": ("F:/COYO/embeds/train_anyup/", "E:/COYO/embeds/train_noup/"),
        "val":   ("G:/coyo/embeds/val_anyup/",   "E:/COYO/embeds/val_noup/"),
        "test":  ("G:/coyo/embeds/test_anyup/",  "E:/COYO/embeds/test_noup/"),
    }

    splits_to_run = ["train", "val", "test"] if args.split == "all" else [args.split]
    all_reports = {}

    for split in splits_to_run:
        f_anyup, f_noup = defaults[split]
        if args.folder_anyup:
            f_anyup = args.folder_anyup
        if args.folder_noup:
            f_noup = args.folder_noup

        if not os.path.isdir(f_anyup):
            print(f"\n[SKIP] {split}: {f_anyup} does not exist")
            continue
        if not os.path.isdir(f_noup):
            print(f"\n[SKIP] {split}: {f_noup} does not exist")
            continue

        report = run_cka_analysis(
            folder_anyup=f_anyup,
            folder_noup=f_noup,
            n_samples=args.n_samples,
            output_dir=args.output_dir,
            split=split,
            kernel=args.kernel,
            device=args.device,
        )
        all_reports[split] = report

    if len(all_reports) > 1:
        print(f"\n\n{'═' * 70}")
        print(f"  FINAL SUMMARY -- CKA (ckatorch) per split")
        print(f"{'═' * 70}")
        pairs_keys = list(next(iter(all_reports.values()))["cka"].keys())
        header = f"  {'Split':<8} " + "".join(f"{k[:24]:>26}" for k in pairs_keys)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for split, rep in all_reports.items():
            row = f"  {split:<8} " + "".join(f"{rep['cka'][k]:>26.4f}" for k in pairs_keys)
            print(row)
