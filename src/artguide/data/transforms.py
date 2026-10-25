"""Image transforms: uint8 (3, H, W) tensors -> normalised float tensors ready for a model."""
from __future__ import annotations

import torch
from torchvision.transforms import v2

MLP_SIZE = 64  # 3 x 64 x 64 = 12 288 inputs for the MLP (49 152 at 128 px)


def channel_stats(images: torch.Tensor) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of a (N, 3, H, W) uint8 tensor, on the [0, 1] scale."""
    x = images.float() / 255
    return x.mean(dim=(0, 2, 3)).tolist(), x.std(dim=(0, 2, 3)).tolist()


def mlp_transform(mean: list[float], std: list[float], size: int = MLP_SIZE) -> v2.Compose:
    """Resize to size x size, scale to [0, 1], then standardise each channel with the train statistics."""
    return v2.Compose([
        v2.Resize(size, antialias=True),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
        v2.Normalize(mean, std),  # (x - mean) / std, channel by channel
    ])
