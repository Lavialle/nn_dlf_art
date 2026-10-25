"""The frozen splits must not leak information between train, val and test."""
from pathlib import Path

import pandas as pd
import pytest

from artguide.data.splits import UNKNOWN_ARTIST, make_splits

ROOT = Path(__file__).resolve().parents[1]
SPLIT_FILES = sorted((ROOT / "splits").glob("*.csv"))  # splits/<subset>.csv is built from data/<subset>/
EXPECTED = {"train": 0.70, "val": 0.15, "test": 0.15}
TOLERANCE = 0.05  # a large artist group can shift a class by a few points

each_split_file = pytest.mark.parametrize("split_file", SPLIT_FILES, ids=lambda p: p.stem)


@each_split_file
def test_each_image_in_exactly_one_split(split_file):
    s = pd.read_csv(split_file)
    assert s["id"].is_unique
    assert set(s["split"]) == set(EXPECTED)


@each_split_file
def test_no_named_artist_in_two_splits(split_file):
    s = pd.read_csv(split_file)
    named = s[s["artist"] != UNKNOWN_ARTIST]
    n_splits_per_artist = named.groupby("artist")["split"].nunique()
    leaking = n_splits_per_artist[n_splits_per_artist > 1]
    assert leaking.empty, f"artists in several splits: {list(leaking.index)}"


@each_split_file
def test_each_class_is_stratified(split_file):
    s = pd.read_csv(split_file)
    proportions = pd.crosstab(s["label"], s["split"], normalize="index")  # one row per class, sums to 1
    for split, expected in EXPECTED.items():
        assert (proportions[split] - expected).abs().max() < TOLERANCE, proportions


@each_split_file
def test_frozen_split_matches_make_splits(split_file):
    metadata_file = ROOT / "data" / split_file.stem / "metadata.csv"
    if not metadata_file.exists():
        pytest.skip(f"{metadata_file} not downloaded")
    frozen = pd.read_csv(split_file).set_index("id")["split"]
    recomputed = make_splits(pd.read_csv(metadata_file), seed=0).set_index("id")["split"]
    assert set(frozen.index) == set(recomputed.index)
    assert (frozen == recomputed.loc[frozen.index]).all()
