"""Sanity checks, run before any tuning: they tell a bug in the pipeline apart from a weak model."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.utils.data import DataLoader

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


def lr_range_test(model: nn.Module, loader: DataLoader, loss_fn: nn.Module, optimizer: torch.optim.Optimizer,
                  device: torch.device, start_lr: float = 1e-6, end_lr: float = 10.0, steps: int = 200,
                  smoothing: float = 0.05, divergence: float = 4.0) -> tuple[list[float], list[float]]:
    """Train with a learning rate growing exponentially from start_lr to end_lr; record the smoothed loss.

    Stops early once the loss exceeds `divergence` times its best value. Destroys the model's weights:
    run it on a fresh model, never on one you want to keep.
    """
    model.train()
    growth = (end_lr / start_lr) ** (1 / (steps - 1))  # the LR is multiplied by this at every step
    lrs, losses = [], []
    avg, best, batches = 0.0, float("inf"), iter(loader)
    for step in range(steps):
        lr = start_lr * growth ** step
        for group in optimizer.param_groups:
            group["lr"] = lr
        try:
            x, y = next(batches)
        except StopIteration:  # end of an epoch: start a new pass over the training set
            batches = iter(loader)
            x, y = next(batches)
        optimizer.zero_grad()
        loss = loss_fn(model(x.to(device)), y.to(device))
        loss.backward()
        optimizer.step()

        avg = smoothing * loss.item() + (1 - smoothing) * avg  # exponential moving average of the loss
        smoothed = avg / (1 - (1 - smoothing) ** (step + 1))  # bias correction: avg starts at 0
        lrs.append(lr)
        losses.append(smoothed)
        best = min(best, smoothed)
        if not math.isfinite(smoothed) or smoothed > divergence * best:
            break
    return lrs, losses


def suggest_lr(lrs: list[float], losses: list[float]) -> float:
    """A tenth of the learning rate where the smoothed loss is lowest: safely below the edge of divergence."""
    finite = [(loss, lr) for lr, loss in zip(lrs, losses) if math.isfinite(loss)]
    return min(finite)[1] / 10
