"""Build the device, data loaders and model described by a Config: shared by every script, so they all agree."""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch.utils.data import DataLoader

from artguide.config import Config
from artguide.data.dataset import WikiArtSplit
from artguide.data.transforms import channel_stats, mlp_transform
from artguide.models.mlp import MLP


class Loaders(NamedTuple):
    train: DataLoader  # shuffled, augmented if cfg.augment: for training
    train_eval: DataLoader  # the same images, never augmented: train scores comparable with val scores
    val: DataLoader
    classes: list[str]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(cfg: Config, generator: torch.Generator) -> Loaders:
    """Train and val loaders, normalised with train-only statistics; plus the class names."""
    train = WikiArtSplit(cfg.data_dir, cfg.splits_path, "train")
    stats = channel_stats(train.images)
    transform = mlp_transform(*stats, size=cfg.image_size)
    train_eval = copy.copy(train)  # shares the images in memory, only the transform differs
    train_eval.transform = transform
    train.transform = mlp_transform(*stats, size=cfg.image_size, augment=cfg.augment)
    val = WikiArtSplit(cfg.data_dir, cfg.splits_path, "val", transform=transform)
    return Loaders(train=DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=generator),
                   train_eval=DataLoader(train_eval, batch_size=256),
                   val=DataLoader(val, batch_size=256),
                   classes=train.classes)


def make_model(cfg: Config, n_classes: int) -> MLP:
    return MLP(3 * cfg.image_size ** 2, n_classes, hidden=cfg.hidden, activation=cfg.activation,
               batchnorm=cfg.batchnorm, dropout=cfg.dropout, init=cfg.init)
