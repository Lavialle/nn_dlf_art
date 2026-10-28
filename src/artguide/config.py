"""Experiment configuration: one YAML file per experiment, loaded into a typed dataclass."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml

FAILURES = (None, "shuffle_labels", "forget_eval")


@dataclass(frozen=True)
class Config:
    name: str  # run name in MLflow, e.g. "mlp_v0"
    data_dir: str  # e.g. "data/wikiart_style5"
    split_file: str | None = None  # default: splits/<subset>.csv
    seed: int = 0
    group: str = "mlp"  # MLflow experiment <subset>-<group>: mlp, failures, resnet

    # Model (see models/mlp.py and models/resnet.py)
    model: str = "mlp"  # mlp, resnet18
    finetune: str = "probe"  # resnet18 only: probe, partial, full
    hidden: tuple[int, ...] = (512,)
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    init: str = "kaiming"
    image_size: int = 64
    augment: bool = False  # random crop + horizontal flip on the training images

    # Training
    optimizer: str = "sgd"  # sgd, adam, adamw
    lr: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 30
    max_grad_norm: float | None = None  # gradient clipping, off by default
    label_smoothing: float = 0.0  # training targets 0.9 / 0.025 instead of 1 / 0 (with 0.1 and 5 classes)
    scheduler: str = "none"  # none, cosine
    patience: int | None = None  # early stopping: stop after this many epochs without a better val loss
    backbone_lr: float | None = None  # resnet18: LR of the pre-trained layers (default: lr)

    # Provoked failures (Phase 4), off by default
    failure: str | None = None  # shuffle_labels, forget_eval

    @property
    def subset(self) -> str:
        """data/wikiart_style5_224 -> wikiart_style5: the 224 px copy holds the same paintings and split."""
        return Path(self.data_dir).name.removesuffix("_224")

    @property
    def experiment(self) -> str:
        return f"{self.subset}-{self.group}"  # e.g. "wikiart_style5-mlp"

    @property
    def splits_path(self) -> Path:
        return Path(self.split_file) if self.split_file else Path("splits") / f"{self.subset}.csv"

    @property
    def checkpoint_path(self) -> Path:
        return Path("checkpoints") / self.subset / f"{self.name}-seed{self.seed}.pt"


def parse_overrides(pairs: list[str]) -> dict:
    """['dropout=0.2', 'hidden=[256]'] -> {'dropout': 0.2, 'hidden': [256]}: each value is read as YAML."""
    overrides = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            raise ValueError(f"expected KEY=VALUE, got {pair!r}")
        overrides[key.strip()] = yaml.safe_load(value)
    return overrides


def load_config(path: str | Path, overrides: dict | None = None) -> Config:
    """Read a YAML file into a Config. An unknown key (a typo) raises an error instead of being ignored.

    Overrides replace values of the file; unless they set the name, it gets a suffix (mlp_v3 -> mlp_v3-dropout0.2)
    so that the run cannot be mistaken for the config as written.
    """
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overrides:
        if "name" not in overrides:
            overrides = {**overrides, "name": values["name"] + "".join(f"-{k}{v}" for k, v in overrides.items())}
        values = {**values, **overrides}
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    if values.get("failure") not in FAILURES:
        raise ValueError(f"{path}: unknown failure {values['failure']!r}, expected one of {FAILURES[1:]}")
    if "hidden" in values:
        values["hidden"] = tuple(values["hidden"])  # YAML gives a list; the dataclass is frozen, so a tuple
    return Config(**values)
