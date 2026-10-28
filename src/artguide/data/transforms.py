"""Image transforms: uint8 (3, H, W) tensors -> normalised float tensors ready for a model (MLP or ResNet)."""
from __future__ import annotations

import torch
from torchvision.transforms import v2

MLP_SIZE = 64  # 3 x 64 x 64 = 12 288 inputs for the MLP (49 152 at 128 px)


def channel_stats(images: torch.Tensor) -> tuple[list[float], list[float]]:
    """Per-channel mean and std of a (N, 3, H, W) uint8 tensor, on the [0, 1] scale."""
    x = images.float() / 255
    return x.mean(dim=(0, 2, 3)).tolist(), x.std(dim=(0, 2, 3)).tolist()


def augmentation(size: int) -> list[v2.Transform]:
    """Random crop (60-100 % of the area) resized to size x size, and a horizontal flip half of the time.

    Chosen for paintings: no colour jitter (the palette IS a clue to the style), no rotation, no vertical flip.
    """
    return [v2.RandomResizedCrop(size, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip()]


def image_transform(mean: list[float], std: list[float], size: int = MLP_SIZE,
                    augment: bool = False) -> v2.Compose:
    """Resize to size x size (randomly cropped and flipped if augment), scale to [0, 1], then standardise each
    channel: with the train statistics for the MLP, with ImageNet's for the pre-trained ResNet."""
    return v2.Compose([
        *(augmentation(size) if augment else [v2.Resize(size, antialias=True)]),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
        v2.Normalize(mean, std),  # (x - mean) / std, channel by channel
    ])
