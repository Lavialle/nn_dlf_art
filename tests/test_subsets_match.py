"""The 224 px subset (ResNet) must hold exactly the same paintings as the base subset (MLP), so one split serves both."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = sorted((ROOT / "splits").glob("*.csv"))  # splits/<subset>.csv serves data/<subset>/ and data/<subset>_224/
SAME_COLUMNS = ["id", "path", "label", "label_idx", "artist"]
MAX_PIXEL_GAP = 15  # mean |difference| on 0-255; measured: same painting <= 5, two different paintings >= 38
N_CHECKED = 20


def subset_dirs(split_file: Path) -> tuple[Path, Path]:
    base, big = ROOT / "data" / split_file.stem, ROOT / "data" / f"{split_file.stem}_224"
    if not (base / "metadata.csv").exists() or not (big / "metadata.csv").exists():
        pytest.skip(f"{base.name} or {big.name} not downloaded")
    return base, big


def as_array(path: Path, size: int | None = None) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize((size, size), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.float32)


each_split_file = pytest.mark.parametrize("split_file", SPLIT_FILES, ids=lambda p: p.stem)


@each_split_file
def test_same_metadata(split_file):
    base, big = subset_dirs(split_file)
    a, b = pd.read_csv(base / "metadata.csv"), pd.read_csv(big / "metadata.csv")
    assert a[SAME_COLUMNS].equals(b[SAME_COLUMNS])


@each_split_file
def test_big_subset_is_224_px(split_file):
    _, big = subset_dirs(split_file)
    first = pd.read_csv(big / "metadata.csv")["path"].iloc[0]
    assert as_array(big / first).shape == (224, 224, 3)


@each_split_file
def test_same_paintings(split_file):
    base, big = subset_dirs(split_file)
    paths = pd.read_csv(split_file)["path"].sample(N_CHECKED, random_state=0)
    for p in paths:
        small = as_array(base / p)
        shrunk = as_array(big / p, size=small.shape[0])  # 224 px brought back to the base size
        gap = np.abs(small - shrunk).mean()
        assert gap < MAX_PIXEL_GAP, f"{p}: mean pixel gap {gap:.1f}"
