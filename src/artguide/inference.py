"""Prediction on new photos with the recommended model: frozen ResNet18 embeddings + logistic regression.

A photo is prepared exactly as the training images were (make_wikiart_subset.py: shorter side -> 224 px, centre
crop), then goes through the same transform and backbone as in the final evaluation (evaluation/candidates.py).
Only the fitted logistic regression is saved: the backbone is rebuilt from torchvision's ImageNet weights.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.pipeline import Pipeline
from torch import nn

from artguide.data.transforms import image_transform
from artguide.models.resnet import WEIGHTS, embedding_model, imagenet_stats

SIZE = 224  # the input size of the ResNet subsets (data/<subset>_224)
CHUNK = 64  # images per forward pass, as in the final evaluation (same features, bit for bit)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def model_path(subset: str) -> Path:
    return Path("models") / f"{subset}_resnet_embed.pkl"


def load_photo(path: str | Path, size: int = SIZE) -> torch.Tensor:
    """A photo file -> (3, size, size) uint8 tensor, prepared as the training images were: upright (phones store
    the rotation in the EXIF tag instead of turning the pixels), RGB, shorter side resized to `size` and
    centre-cropped (ImageOps.fit, as make_wikiart_subset.py does). image_transform alone would only resize the
    shorter side: a non-square photo would stay non-square, unlike every training image."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
        im = ImageOps.fit(im, (size, size), method=Image.Resampling.BICUBIC)
        return torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1).contiguous()  # (H, W, 3) -> (3, H, W)


@dataclass
class Predictor:
    """Frozen backbone (image -> 512 numbers) + a scikit-learn classifier on those numbers."""
    classifier: Pipeline  # StandardScaler + logistic regression (models/baselines.py)
    classes: list[str]  # class names, in the order of the label indices
    info: dict = field(default_factory=dict)  # provenance: subset, C, git commit, backbone weights
    backbone: nn.Module = field(default_factory=embedding_model, repr=False)

    def __post_init__(self):
        self.transform = image_transform(*imagenet_stats(), size=SIZE)  # ImageNet statistics: what the weights expect

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        """(N, 3, H, W) uint8 -> (N, n_features) floats."""
        self.backbone.eval()
        return np.concatenate([self.backbone(torch.stack([self.transform(image) for image in chunk])).numpy()
                               for chunk in images.split(CHUNK)])

    def fit(self, images: torch.Tensor, labels: np.ndarray) -> Predictor:
        self.classifier.fit(self.embed(images), labels)
        # predict_proba's columns follow classifier.classes_: they must be the label indices 0..K-1, in order
        assert list(self.classifier.classes_) == list(range(len(self.classes))), "a class has no training image"
        return self

    def probabilities(self, images: torch.Tensor) -> np.ndarray:
        """(N, K): the probability of each class for each image (not calibrated: read it as a ranking)."""
        return self.classifier.predict_proba(self.embed(images))

    def predict(self, images: torch.Tensor) -> np.ndarray:
        return self.probabilities(images).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        """The classifier, the class names and the provenance; not the backbone (45 MB, rebuilt from torchvision)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump({"classifier": self.classifier, "classes": self.classes,
                         "info": {**self.info, "weights": str(WEIGHTS)}}, f)

    @classmethod
    def load(cls, path: str | Path, backbone: nn.Module | None = None) -> Predictor:
        """Only load files written by save(): unpickling runs code from the file."""
        with Path(path).open("rb") as f:
            saved = pickle.load(f)
        extra = {} if backbone is None else {"backbone": backbone}
        return cls(saved["classifier"], saved["classes"], saved["info"], **extra)
