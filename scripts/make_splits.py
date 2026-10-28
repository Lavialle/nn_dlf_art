"""Freeze the train/val/test split of a WikiArt subset into a CSV committed to git.

    uv run python scripts/make_splits.py --data-dir data/wikiart_style5 --out splits/wikiart_style5.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from artguide.data.splits import make_splits

COLUMNS = ["id", "path", "label", "label_idx", "artist", "split"]  # identifiers only, no image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True, help="subset folder containing metadata.csv")
    p.add_argument("--out", type=Path, required=True, help="output CSV")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    metadata = pd.read_csv(args.data_dir / "metadata.csv")
    splits = make_splits(metadata, seed=args.seed)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    splits[COLUMNS].to_csv(args.out, index=False)

    print(pd.crosstab(splits["label"], splits["split"], margins=True))
    print(f"\nSaved {len(splits)} rows to {args.out} (seed={args.seed})")


if __name__ == "__main__":
    main()
