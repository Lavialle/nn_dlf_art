"""Build the device, data loaders and model described by a Config: shared by every script, so they all agree."""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from artguide.config import Config
from artguide.data.dataset import WikiArtSplit
from artguide.data.transforms import channel_stats, mlp_transform
from artguide.models.mlp import MLP


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(cfg: Config, generator: torch.Generator) -> tuple[DataLoader, DataLoader, list[str]]:
    """Train loader (shuffled) and val loader, normalised with train-only statistics; plus the class names."""
    train = WikiArtSplit(cfg.data_dir, cfg.splits_path, "train")
    transform = mlp_transform(*channel_stats(train.images), size=cfg.image_size)
    train.transform = transform
    val = WikiArtSplit(cfg.data_dir, cfg.splits_path, "val", transform=transform)
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=generator)
    val_loader = DataLoader(val, batch_size=256)
    return train_loader, val_loader, train.classes


def make_model(cfg: Config, n_classes: int) -> MLP:
    return MLP(3 * cfg.image_size ** 2, n_classes, hidden=cfg.hidden, activation=cfg.activation,
               batchnorm=cfg.batchnorm, dropout=cfg.dropout, init=cfg.init)
