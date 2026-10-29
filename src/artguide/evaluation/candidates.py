"""The final candidates for the kiosk, rebuilt for the single test-set pass and the latency benchmark.

Every choice (the C of a logistic regression, the MLP version, the ResNet fine-tuning level) is made on the
validation scores in MLflow, never on the test set. Each candidate takes raw uint8 images of its subset (128 px
for the MLP and the baselines, 224 px for ResNet) and returns class indices: test scores and latency cover the
whole path from the image to the answer.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from sklearn.dummy import DummyClassifier
from torch import nn

from artguide.config import Config
from artguide.data.dataset import WikiArtSplit
from artguide.data.transforms import channel_stats, image_transform
from artguide.evaluation.latency import size_mb
from artguide.models.baselines import hsv_histogram_features, logistic_regression, pixel_features
from artguide.models.resnet import embedding_model, imagenet_stats
from artguide.reporting import latest_per_name, runs
from artguide.train.builders import make_model

SEEDS = (0, 1, 2)
MLP_CONFIGS = ["mlp_v0", "mlp_v1_sgd", "mlp_v1_nesterov", "mlp_v1_adamw", "mlp_v2", "mlp_v3", "mlp_v4", "mlp_v5",
               "mlp_v6"]
FINETUNED = ["resnet_partial", "resnet_full"]


@dataclass
class Candidate:
    name: str  # e.g. "MLP (mlp_v5)"
    family: str  # baseline, mlp, resnet
    data_dir: str  # the images it reads
    prepare: Callable[[torch.Tensor], object]  # uint8 (N, 3, H, W) -> what the model takes
    model: object  # nn.Module in eval mode, or a fitted sklearn model
    seed: int | None = None
    other_parts: list = field(default_factory=list)  # also shipped to the kiosk (e.g. a frozen backbone)

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> np.ndarray:
        x = self.prepare(images)
        if isinstance(self.model, nn.Module):
            return self.model(x).argmax(dim=1).numpy()
        return self.model.predict(x)

    def size_mb(self) -> float:
        return sum(size_mb(part) for part in [self.model, *self.other_parts])


def mean_val_loss(df, name: str) -> float:
    found = latest_per_name(df, [f"{name}-seed{s}" for s in SEEDS])
    return found["metrics.best_val_loss"].mean() if len(found) == len(SEEDS) else float("inf")


def best_config(df, names: list[str]) -> str:
    """The config with the lowest mean val loss over the 3 seeds: the criterion of early stopping too. Val
    accuracy moves by 0.45 points per image on 222 images; the loss is a smoother signal."""
    return min(names, key=lambda name: mean_val_loss(df, name))


def best_c(df, prefix: str) -> float:
    """C of the logistic regression run with the best val accuracy among runs named <prefix>-C<value>.

    Ties go to the smallest C (the strongest regularisation): on the ResNet embeddings, C = 0.01 and C = 10 tie on
    val, but C = 10 is fragile (features computed on CPU instead of GPU change 5 val predictions), C = 0.01 is not.
    """
    found = df[df["tags.mlflow.runName"].str.startswith(f"{prefix}-C")].copy()
    found["C"] = found["params.C"].astype(float)
    return float(found.sort_values(["metrics.val_accuracy", "C"], ascending=[False, True]).iloc[0]["C"])


def stack(transform) -> Callable[[torch.Tensor], torch.Tensor]:
    return lambda images: torch.stack([transform(image) for image in images])


def torch_candidates(subset: str, name: str, family: str, label: str) -> list[Candidate]:
    """One candidate per seed, from the checkpoints saved by scripts/train.py (CPU, eval mode)."""
    out = []
    for seed in SEEDS:
        checkpoint = torch.load(Path("checkpoints") / subset / f"{name}-seed{seed}.pt", map_location="cpu")
        values = checkpoint["config"]
        cfg = Config(**{**values, "hidden": tuple(values["hidden"])})
        model = make_model(cfg, len(checkpoint["classes"]))
        model.load_state_dict(checkpoint["state_dict"])
        if cfg.model == "resnet18":
            stats = imagenet_stats()
        else:  # the MLP was normalised with the statistics of its training images: recompute them, same data
            stats = channel_stats(WikiArtSplit(cfg.data_dir, cfg.splits_path, "train").images)
        out.append(Candidate(label, family, cfg.data_dir, stack(image_transform(*stats, size=cfg.image_size)),
                             model.eval(), seed))
    return out


def build_candidates(subset: str) -> list[Candidate]:
    data_dir, data_dir_224 = f"data/{subset}", f"data/{subset}_224"
    split_file = Path("splits") / f"{subset}.csv"
    train = WikiArtSplit(data_dir, split_file, "train")
    y_train = train.labels.numpy()
    baselines, mlp, resnet = runs(f"{subset}-baselines"), runs(f"{subset}-mlp"), runs(f"{subset}-resnet")

    candidates = [Candidate("Majority class", "baseline", data_dir, lambda images: np.zeros((len(images), 1)),
                            DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_train), 1)), y_train))]
    for features, label in [(pixel_features, "pixels"), (hsv_histogram_features, "hsv")]:
        model = logistic_regression(best_c(baselines, f"logreg-{label}")).fit(features(train.images), y_train)
        candidates.append(Candidate(f"Logistic regression ({label})", "baseline", data_dir, features, model))

    mlp_name = best_config(mlp, MLP_CONFIGS)
    candidates += torch_candidates(subset, mlp_name, "mlp", f"MLP ({mlp_name})")

    backbone = embedding_model()
    embed = stack(image_transform(*imagenet_stats(), size=224))
    to_features = lambda images: backbone(embed(images)).numpy()  # noqa: E731
    with torch.no_grad():
        train_features = to_features(WikiArtSplit(data_dir_224, split_file, "train").images)
    model = logistic_regression(best_c(resnet, "resnet_embed_logreg")).fit(train_features, y_train)
    candidates.append(Candidate("ResNet18 embeddings + logistic regression", "resnet", data_dir_224,
                                torch.no_grad()(to_features), model, other_parts=[backbone]))

    candidates += torch_candidates(subset, "resnet_probe", "resnet", "ResNet18 linear probe")
    finetuned = best_config(resnet, FINETUNED)
    candidates += torch_candidates(subset, finetuned, "resnet", f"ResNet18 fine-tuned ({finetuned})")
    return candidates
