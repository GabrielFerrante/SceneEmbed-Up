from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class EarlyStopping:
    """
    Critério de parada antecipada baseado em métrica escalar (tipicamente loss).

    Parâmetros
    ----------
    save_dir:
        Diretório onde o melhor `state_dict` será persistido.
    filename:
        Nome do arquivo de checkpoint a ser salvo.
    patience:
        Número máximo de épocas consecutivas sem melhora aceitável.
    min_delta:
        Variação mínima (melhora) necessária para resetar o contador de paciência.
    """

    save_dir: str = "checkpoints"
    filename: str = "best_aligner.pth"
    patience: int = 5
    min_delta: float = 0.001

    def __post_init__(self) -> None:
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.early_stop: bool = False
        os.makedirs(self.save_dir, exist_ok=True)

    def __call__(self, current_loss: float, model_state: dict[str, Any], epoch: int) -> bool:
        """
        Atualiza o estado de parada antecipada.

        Parameters
        ----------
        current_loss:
            Métrica atual (quanto menor, melhor).
        model_state:
            `state_dict` do modelo a ser persistido quando houver melhora.
        epoch:
            Época corrente (0‑based).

        Returns
        -------
        bool
            `True` se o treinamento deve ser encerrado.
        """
        if current_loss < (self.best_loss - self.min_delta):
            self.best_loss = current_loss
            self.counter = 0
            path = os.path.join(self.save_dir, self.filename)
            torch.save(model_state, path)
            print(f" >>> Época {epoch + 1}: Novo melhor modelo salvo! (Loss: {current_loss:.4f})")
        else:
            self.counter += 1
            print(f" >>> Época {epoch + 1}: Sem melhora na Loss. ({self.counter}/{self.patience})")

            if self.counter >= self.patience:
                self.early_stop = True
                print(" !!! Early Stopping acionado! Encerrando treinamento.")

        return self.early_stop

