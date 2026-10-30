"""Build the device, data loaders, model and optimizer described by a Config: shared by every script."""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from artguide.config import Config
from artguide.data.dataset import WikiArtSplit
from artguide.data.transforms import channel_stats, image_transform
from artguide.models.cnn import SmallCNN
from artguide.models.mlp import MLP
from artguide.models.resnet import ResNetClassifier, imagenet_stats
from artguide.train.optim import make_optimizer


class Loaders(NamedTuple):
    train: DataLoader  # shuffled, augmented if cfg.augment: for training
    train_eval: DataLoader  # the same images, never augmented: train scores comparable with val scores
    val: DataLoader
    classes: list[str]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_loaders(cfg: Config, generator: torch.Generator) -> Loaders:
    """Train and val loaders, plus the class names. Normalised with train-only statistics for the MLP, with the
    ImageNet statistics for the pre-trained ResNet (the ones its weights were trained with)."""
    train = WikiArtSplit(cfg.data_dir, cfg.splits_path, "train")
    if cfg.failure == "shuffle_labels":  # provoked failure: the labels no longer match the images
        order = torch.randperm(len(train.labels), generator=torch.Generator().manual_seed(cfg.seed))
        train.labels = train.labels[order]
    stats = imagenet_stats() if cfg.model == "resnet18" else channel_stats(train.images)
    transform = image_transform(*stats, size=cfg.image_size)
    train_eval = copy.copy(train)  # shares the images in memory, only the transform differs
    train_eval.transform = transform
    train.transform = image_transform(*stats, size=cfg.image_size, augment=cfg.augment)
    val = WikiArtSplit(cfg.data_dir, cfg.splits_path, "val", transform=transform)
    return Loaders(train=DataLoader(train, batch_size=cfg.batch_size, shuffle=True, generator=generator),
                   train_eval=DataLoader(train_eval, batch_size=256),
                   val=DataLoader(val, batch_size=256),
                   classes=train.classes)


def make_model(cfg: Config, n_classes: int) -> nn.Module:
    if cfg.model == "resnet18":
        return ResNetClassifier(n_classes, finetune=cfg.finetune)
    if cfg.model == "cnn":
        return SmallCNN(n_classes, channels=cfg.channels, dropout=cfg.dropout)
    if cfg.model != "mlp":
        raise ValueError(f"unknown model {cfg.model!r}: expected mlp, cnn or resnet18")
    return MLP(3 * cfg.image_size ** 2, n_classes, hidden=cfg.hidden, activation=cfg.activation,
               batchnorm=cfg.batchnorm, dropout=cfg.dropout, init=cfg.init)


def make_model_optimizer(cfg: Config, model: nn.Module) -> torch.optim.Optimizer:
    """Every MLP weight at cfg.lr; for ResNet, only the trainable layers: head at lr, backbone at backbone_lr."""
    if isinstance(model, ResNetClassifier):
        params = model.param_groups(cfg.lr, cfg.backbone_lr or cfg.lr)
    else:
        params = model.parameters()
    return make_optimizer(params, cfg.optimizer, cfg.lr, cfg.momentum, cfg.nesterov, cfg.weight_decay)
