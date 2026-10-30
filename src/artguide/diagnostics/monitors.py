"""Training monitors: what happens inside the network, epoch by epoch.

- gradient norm of each layer: exploding (huge) or vanishing (tiny in the first layers) gradients;
- update/weight ratio ||Δw|| / ||w|| per step: ~1e-3 is healthy; far above = LR too high, far below = too slow;
- activations of each hidden layer: mean, std and the share of dead ReLUs (never active, on any image).
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn

EPS = 1e-12  # avoids 0 / 0 when the weights are all zeros


def linear_layers(model: nn.Module) -> dict[str, nn.Linear]:
    """The Linear layers in forward order, named fc1, fc2, ... and out (the last one)."""
    layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    return {**{f"fc{i}": layer for i, layer in enumerate(layers[:-1], start=1)}, "out": layers[-1]}


class TrainingMonitor:
    """Passed to train_one_epoch, which calls after_backward() and after_step() at every step.

    epoch_stats() returns the averages over the steps of the epoch, then starts a new epoch.
    """

    def __init__(self, model: nn.Module):
        self.layers = linear_layers(model)
        self._reset()

    def _reset(self) -> None:
        self._sums: dict[str, float] = defaultdict(float)
        self._steps = 0
        self._before: dict[str, torch.Tensor] = {}

    def after_backward(self) -> None:
        """Gradient norms (before any clipping), and a copy of the weights to measure the update."""
        for name, layer in self.layers.items():
            if layer.weight.grad is not None:
                self._sums[f"grad_norm/{name}"] += layer.weight.grad.norm().item()
            self._before[name] = layer.weight.detach().clone()

    def after_step(self) -> None:
        for name, layer in self.layers.items():
            w = layer.weight.detach()
            self._sums[f"update_ratio/{name}"] += ((w - self._before[name]).norm() / (w.norm() + EPS)).item()
        self._steps += 1

    def epoch_stats(self) -> dict[str, float]:
        stats = {key: total / self._steps for key, total in self._sums.items()} if self._steps else {}
        self._reset()
        return stats


@torch.no_grad()
def activation_stats(model: nn.Module, x: torch.Tensor) -> dict[str, float]:
    """Forward one fixed batch in eval mode and describe the output of every hidden activation function.

    Dead ReLU = a unit whose output is 0 for every image of the batch: it passes no signal and gets no gradient.
    """
    activations = [m for m in model.modules() if isinstance(m, (nn.ReLU, nn.Sigmoid, nn.Tanh))]
    outputs: list[torch.Tensor] = []
    hooks = [m.register_forward_hook(lambda _m, _inp, out: outputs.append(out)) for m in activations]
    was_training = model.training
    model.eval()
    try:
        model(x)
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)

    stats = {}
    for i, (module, out) in enumerate(zip(activations, outputs), start=1):
        stats[f"act_mean/fc{i}"] = out.mean().item()
        stats[f"act_std/fc{i}"] = out.std().item()
        if isinstance(module, nn.ReLU):
            stats[f"dead_relu/fc{i}"] = (out <= 0).all(dim=0).float().mean().item()
    return stats
