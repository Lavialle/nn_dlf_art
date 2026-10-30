"""Early stopping on the validation loss, keeping a copy of the best weights."""
from __future__ import annotations

import torch
from torch import nn


class EarlyStopping:
    """Call step() after every epoch. It remembers the epoch with the lowest val loss and a copy of its weights;
    should_stop turns True after `patience` epochs without improvement (never, if patience is None).
    """

    def __init__(self, patience: int | None = None):
        self.patience = patience
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.best_state: dict[str, torch.Tensor] | None = None
        self.bad_epochs = 0  # epochs since the last improvement

    def step(self, val_loss: float, epoch: int, model: nn.Module) -> bool:
        """Record one epoch; return True if it is the new best."""
        if val_loss < self.best_loss:
            self.best_loss, self.best_epoch, self.bad_epochs = val_loss, epoch, 0
            # a copy, on the CPU: the model keeps training and would overwrite references to its own tensors
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience is not None and self.bad_epochs >= self.patience
