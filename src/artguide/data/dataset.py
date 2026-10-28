"""In-memory PyTorch dataset for one split (train, val or test) of a WikiArt subset."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


def load_image(path: Path) -> np.ndarray:
    """One image as a (H, W, 3) uint8 array."""
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


class WikiArtSplit(Dataset):
    """All the images of one split, decoded once and kept in memory as a uint8 tensor."""

    def __init__(self, data_dir: str | Path, split_file: str | Path, split: str, transform=None):
        table = pd.read_csv(split_file)
        self.classes = table.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()
        table = table[table["split"] == split].reset_index(drop=True)

        arrays = [load_image(Path(data_dir) / p) for p in table["path"]]
        self.images = torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()  # (N, 3, H, W)
        self.labels = torch.tensor(table["label_idx"].to_numpy(), dtype=torch.long)
        self.ids = table["id"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.images[i]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[i]
