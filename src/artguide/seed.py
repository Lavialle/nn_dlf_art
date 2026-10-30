"""Seeding: the same seed gives the same initial weights, the same batch order and the same dropout masks."""
from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = True) -> torch.Generator:
    """Seed Python, numpy and torch (CPU and every GPU); return a generator for the DataLoader's shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
    if deterministic:
        torch.backends.cudnn.deterministic = True  # pick deterministic GPU algorithms (slightly slower)
        torch.backends.cudnn.benchmark = False  # don't let cuDNN pick the fastest, possibly varying, algorithm
    return torch.Generator().manual_seed(seed)  # batch order independent of how many random numbers the model used
