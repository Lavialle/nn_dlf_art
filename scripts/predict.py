"""Name the style or the genre of the paintings in new photos, with the model saved by scripts/export_model.py.

    uv run python scripts/predict.py --subset wikiart_genre5 photo.jpg
    uv run python scripts/predict.py --subset wikiart_style5 photos/ --top 2 --json

Files and folders (their images, not recursive) can be mixed. Works on the CPU, without the dataset or MLflow.
The scores are the logistic regression's probabilities: a ranking, not a calibrated confidence. Best results
when the painting fills the photo: a frame or a wall around it is not what the model was trained on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import UnidentifiedImageError

from artguide.inference import IMAGE_SUFFIXES, Predictor, load_photo, model_path


def image_files(paths: list[str]) -> list[Path]:
    files = []
    for path in map(Path, paths):
        if path.is_dir():
            files += sorted(f for f in path.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        else:
            files.append(path)
    return files


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", help="image files or folders")
    p.add_argument("--subset", default="wikiart_style5", help="the task: wikiart_style5 or wikiart_genre5")
    p.add_argument("--model", help="default: models/<subset>_resnet_embed.pkl")
    p.add_argument("--top", type=int, default=3, help="number of classes shown per photo")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args()
    path = Path(args.model) if args.model else model_path(args.subset)
    if not path.exists():
        raise SystemExit(f"{path} not found: run scripts/export_model.py --subset {args.subset} first")
    predictor = Predictor.load(path)

    files, photos = [], []
    for f in image_files(args.paths):
        try:
            photos.append(load_photo(f))
            files.append(f)
        except (OSError, UnidentifiedImageError) as error:  # missing, unreadable or not an image: skip it
            print(f"{f}: skipped ({error})", file=sys.stderr)
    if not photos:
        raise SystemExit("no image to classify")

    probabilities = predictor.probabilities(torch.stack(photos))
    results = []
    for f, probs in zip(files, probabilities):
        best = probs.argsort()[::-1][:args.top]  # class indices, most likely first
        results.append({"file": str(f), "predictions": [{"label": predictor.classes[k], "probability": float(probs[k])}
                                                         for k in best]})
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        for r in results:
            ranked = " · ".join(f"{c['label']} {c['probability']:.0%}" for c in r["predictions"])
            print(f"{r['file']}  ->  {ranked}")


if __name__ == "__main__":
    main()
