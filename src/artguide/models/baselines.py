"""Classical baselines: what a simple model achieves without a neural network."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from matplotlib.colors import rgb_to_hsv
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.v2 import functional as F

from artguide.data.transforms import MLP_SIZE

HSV_BINS = (8, 4, 4)  # hue x saturation x value -> 128 joint bins


def pixel_features(images: torch.Tensor, size: int = MLP_SIZE) -> np.ndarray:
    """(N, 3, H, W) uint8 -> (N, 3 * size * size) floats in [0, 1]: exactly the input the MLP sees."""
    small = F.resize(images, [size, size], antialias=True)
    return small.flatten(start_dim=1).float().numpy() / 255


def hsv_histogram_features(images: torch.Tensor, bins: tuple[int, int, int] = HSV_BINS) -> np.ndarray:
    """(N, 3, H, W) uint8 -> (N, prod(bins)) colour histograms summing to 1: the palette, without any layout."""
    rgb = images.permute(0, 2, 3, 1).float().numpy() / 255  # (N, H, W, 3), the layout rgb_to_hsv expects
    hsv = rgb_to_hsv(rgb).reshape(len(images), -1, 3)  # (N, H * W, 3): each image as a list of pixels
    histograms = np.stack([np.histogramdd(pixels, bins=bins, range=[(0, 1)] * 3)[0].ravel() for pixels in hsv])
    return histograms / histograms.sum(axis=1, keepdims=True)


def aspect_ratio_features(ids: list[str], metadata: pd.DataFrame) -> np.ndarray:
    """(N, 1) log(height / width) of the original painting, which the square crop erased: > 0 vertical,
    < 0 horizontal, symmetric (2:1 and 1:2 give +0.69 and -0.69)."""
    sizes = metadata.set_index("id").loc[ids, ["orig_height", "orig_width"]].to_numpy(dtype=float)
    return np.log(sizes[:, :1] / sizes[:, 1:])


def logistic_regression(C: float) -> Pipeline:
    """Standardise each feature (statistics fitted on train only), then a multinomial logistic regression."""
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))
