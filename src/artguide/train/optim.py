"""Optimizer factory: the config names an optimizer, this builds it."""
from __future__ import annotations

from collections.abc import Iterable

import torch


def make_optimizer(params: Iterable[torch.nn.Parameter], name: str, lr: float, momentum: float = 0.0,
                   nesterov: bool = False, weight_decay: float = 0.0) -> torch.optim.Optimizer:
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)  # weight decay = L2 penalty, distorted
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)  # decoupled weight decay
    raise ValueError(f"unknown optimizer {name!r}: expected sgd, adam or adamw")


def make_scheduler(optimizer: torch.optim.Optimizer, name: str,
                   epochs: int) -> torch.optim.lr_scheduler.LRScheduler | None:
    """LR schedule, stepped once per epoch. None means a constant learning rate."""
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)  # lr -> 0 at the last epoch
    raise ValueError(f"unknown scheduler {name!r}: expected none or cosine")
