"""Sanity checks, run before any tuning: they tell a bug in the pipeline apart from a weak model."""
from __future__ import annotations

import math

import torch
from torch import nn

INITIAL_LOSS_TOLERANCE = 0.05  # |initial loss - ln K| allowed
OVERFIT_STEPS = 300
OVERFIT_MAX_LOSS = 0.05  # measured on style with v0 (SGD, lr 0.01): 0.12 after 100 steps, 0.017 after 300


def expected_initial_loss(n_classes: int) -> float:
    """Cross-entropy of a model that gives probability 1/K to every class: -ln(1/K) = ln K."""
    return math.log(n_classes)


def overfit_one_batch(model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module,
                      optimizer: torch.optim.Optimizer, steps: int = OVERFIT_STEPS) -> tuple[list[float], float]:
    """Train on the same batch again and again. Returns the loss at every step and the final batch accuracy.

    A working pipeline memorises one batch (loss -> ~0, accuracy -> 100 %). If it can't, the cause is a bug
    (labels misaligned, gradients not flowing, a missing step...), not a lack of capacity or data.
    """
    model.train()
    losses = []
    for _ in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    model.eval()
    with torch.no_grad():
        accuracy = (model(x).argmax(dim=1) == y).float().mean().item()
    return losses, accuracy
