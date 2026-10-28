"""Training loop written by hand: one epoch of learning, one pass of evaluation."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader

from artguide.diagnostics.monitors import TrainingMonitor


def train_one_epoch(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                    device: torch.device, max_grad_norm: float | None = None,
                    monitor: TrainingMonitor | None = None) -> float:
    """One pass over the training set, updating the weights after every batch. Returns the mean training loss.

    A monitor, if given, records gradient norms and update sizes at every step (diagnostics/monitors.py).
    """
    model.train()  # dropout active, BatchNorm normalises with the statistics of the current batch
    total_loss, n_seen = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()  # 1. forget the previous batch's gradients (PyTorch adds them up otherwise)
        logits = model(x)  # 2. forward: one score per class for each image
        loss = loss_fn(logits, y)  # 3. how wrong the scores are, averaged over the batch
        loss.backward()  # 4. backward: gradient of the loss with respect to every weight
        if monitor is not None:
            monitor.after_backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # cap the size of the gradient
        optimizer.step()  # 5. move every weight a little against its gradient
        if monitor is not None:
            monitor.after_step()
        total_loss += loss.item() * len(y)  # .item(): a plain number; x batch size to average over images
        n_seen += len(y)
    return total_loss / n_seen


@torch.no_grad()  # no gradients needed: faster and lighter
def evaluate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, device: torch.device,
             eval_mode: bool = True) -> tuple[float, torch.Tensor, torch.Tensor]:
    """Mean loss, true labels and predicted labels over a whole split, without changing the weights.

    eval_mode=False exists only to reproduce the "forgot model.eval()" failure (Phase 4).
    """
    if eval_mode:
        model.eval()  # dropout off, BatchNorm uses the statistics accumulated during training
    total_loss, labels, preds = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * len(y)
        labels.append(y.cpu())
        preds.append(logits.argmax(dim=1).cpu())  # predicted class = the highest score
    labels = torch.cat(labels)
    return total_loss / len(labels), labels, torch.cat(preds)
