"""Experiment configuration: one YAML file per experiment, loaded into a typed dataclass."""
from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Config:
    name: str  # run name in MLflow, e.g. "mlp_v0"
    data_dir: str  # e.g. "data/wikiart_style5"
    split_file: str | None = None  # default: splits/<data_dir name>.csv
    seed: int = 0

    # Model (see models/mlp.py)
    hidden: tuple[int, ...] = (512,)
    activation: str = "relu"
    batchnorm: bool = False
    dropout: float = 0.0
    init: str = "kaiming"
    image_size: int = 64

    # Training
    optimizer: str = "sgd"  # sgd, adam, adamw
    lr: float = 0.01
    momentum: float = 0.0
    nesterov: bool = False
    weight_decay: float = 0.0
    batch_size: int = 64
    epochs: int = 30
    max_grad_norm: float | None = None  # gradient clipping, off by default

    @property
    def experiment(self) -> str:
        return f"{Path(self.data_dir).name}-mlp"  # e.g. "wikiart_style5-mlp"

    @property
    def splits_path(self) -> Path:
        return Path(self.split_file) if self.split_file else Path("splits") / f"{Path(self.data_dir).name}.csv"


def load_config(path: str | Path) -> Config:
    """Read a YAML file into a Config. An unknown key (a typo) raises an error instead of being ignored."""
    values = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    unknown = set(values) - {f.name for f in fields(Config)}
    if unknown:
        raise ValueError(f"{path}: unknown config keys {sorted(unknown)}")
    if "hidden" in values:
        values["hidden"] = tuple(values["hidden"])  # YAML gives a list; the dataclass is frozen, so a tuple
    return Config(**values)
