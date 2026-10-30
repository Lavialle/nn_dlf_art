"""Train/val/test split of the WikiArt subset, stratified by class and grouped by artist."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

UNKNOWN_ARTIST = "Unknown Artist"
N_FOLDS = 20  # 20 folds of 5 %: 3 -> test (15 %), 3 -> val (15 %), the other 14 -> train (70 %)
N_TEST_FOLDS = 3
N_VAL_FOLDS = 3


def artist_groups(metadata: pd.DataFrame) -> np.ndarray:
    """One group per named artist; each 'Unknown Artist' image is its own group."""
    unknown = metadata["artist"] == UNKNOWN_ARTIST
    return np.where(unknown, metadata["id"], metadata["artist"])


def make_splits(metadata: pd.DataFrame, seed: int = 0, grouped: bool = True) -> pd.DataFrame:
    """Return a copy of metadata with a 'split' column: 'train', 'val' or 'test'.

    grouped=False puts every image in its own group (a plain stratified split, an artist can then be on both
    sides): only to measure the leakage that grouping prevents (Phase 4).
    """
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    groups = artist_groups(metadata) if grouped else metadata["id"]
    fold = np.empty(len(metadata), dtype=int)
    for k, (_, fold_idx) in enumerate(sgkf.split(metadata, metadata["label_idx"], groups)):
        fold[fold_idx] = k
    split = np.select([fold < N_TEST_FOLDS, fold < N_TEST_FOLDS + N_VAL_FOLDS], ["test", "val"], default="train")
    return metadata.assign(split=split)
