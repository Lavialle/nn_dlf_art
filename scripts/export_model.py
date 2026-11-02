"""Fit the recommended model (frozen ResNet18 embeddings + logistic regression) once and save it for predict.py.

    uv run python scripts/export_model.py --subset wikiart_genre5

The model is rebuilt as the final evaluation built it (evaluation/candidates.py), which never saved it: same
training images (data/<subset>_224), same features, same C. By default C is read from MLflow (best val accuracy,
ties to the smallest C: candidates.best_c); without those runs, the C the evaluation used (EVALUATED_C), or --C.

Checks, on the validation set, then against the committed test predictions of the evaluation
(reports/tables/<subset>_test_predictions.csv): the predictions are compared, no test score is computed or used
to choose anything. Identical predictions = the saved file is the model that was evaluated.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from artguide.data.dataset import WikiArtSplit
from artguide.evaluation.candidates import best_c
from artguide.evaluation.metrics import scores
from artguide.inference import Predictor, model_path
from artguide.models.baselines import logistic_regression
from artguide.reporting import runs
from artguide.tracking import _git

EVALUATED = "ResNet18 embeddings + logistic regression | seed None"  # its column in the test predictions
# C kept by the evaluation, for a machine without its MLflow runs: each one reproduces the val accuracy in
# reports/tables/<subset>_resnet.md and every committed test prediction (222 / 222 and 227 / 227)
EVALUATED_C = {"wikiart_style5": 0.01, "wikiart_genre5": 0.001}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    p.add_argument("--C", type=float, help="regularisation of the logistic regression (default: from MLflow)")
    p.add_argument("--out", help="default: models/<subset>_resnet_embed.pkl")
    args = p.parse_args()
    data_dir, split_file = f"data/{args.subset}_224", Path("splits") / f"{args.subset}.csv"
    out = Path(args.out) if args.out else model_path(args.subset)

    if args.C is None:
        resnet = runs(f"{args.subset}-resnet")
        if not resnet.empty and resnet["tags.mlflow.runName"].str.startswith("resnet_embed_logreg-C").any():
            args.C = best_c(resnet, "resnet_embed_logreg")
        elif args.subset in EVALUATED_C:
            args.C = EVALUATED_C[args.subset]
            print(f"no resnet_embed_logreg runs in MLflow: C of the evaluation, {args.C:g}")
        else:
            raise SystemExit(f"no resnet_embed_logreg runs in MLflow for {args.subset}: run "
                             f"scripts/resnet_embeddings.py --subset {args.subset} first, or pass --C")

    train = WikiArtSplit(data_dir, split_file, "train")
    info = {"subset": args.subset, "C": args.C, "n_train": len(train), "git_commit": _git("rev-parse", "HEAD")}
    print(f"fitting on {len(train)} training images, C = {args.C:g} ...")
    predictor = Predictor(logistic_regression(args.C), train.classes, info).fit(train.images, train.labels.numpy())

    val = WikiArtSplit(data_dir, split_file, "val")
    val_scores = scores(val.labels.numpy(), predictor.predict(val.images))
    print(f"val accuracy {val_scores['accuracy']:.3f}, macro-F1 {val_scores['macro_f1']:.3f}")

    committed = Path("reports/tables") / f"{args.subset}_test_predictions.csv"
    if committed.exists():
        test = WikiArtSplit(data_dir, split_file, "test")
        expected = pd.read_csv(committed).set_index("id").loc[test.ids, EVALUATED].to_numpy()
        same = int((predictor.predict(test.images) == expected).sum())
        print(f"test predictions identical to the evaluated model: {same} / {len(test)}")
        if same != len(test):
            print("  differences: another C, or features computed on another device (see best_c)")

    predictor.save(out)
    print(f"-> {out}. Next: uv run python scripts/predict.py --subset {args.subset} <photo>")


if __name__ == "__main__":
    main()
