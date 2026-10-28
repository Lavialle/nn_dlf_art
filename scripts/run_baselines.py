"""Train the classical baselines on train, score them on val, log every run in MLflow. The test set is not touched.

    uv run python scripts/run_baselines.py --data-dir data/wikiart_style5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from sklearn.dummy import DummyClassifier

from artguide.data.dataset import WikiArtSplit
from artguide.evaluation.metrics import scores
from artguide.models.baselines import hsv_histogram_features, logistic_regression, pixel_features
from artguide.tracking import log_scores, start_run

FEATURES = {"pixels": pixel_features, "hsv": hsv_histogram_features}
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]  # inverse regularisation strength: small C = strong regularisation


def fit_and_log(experiment, run_name, params, model, X_train, y_train, X_val, y_val, data_dir) -> dict[str, float]:
    with start_run(experiment, run_name, params, data_dir=data_dir):
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        log_scores({"fit_seconds": time.perf_counter() - t0}, "train")
        log_scores(scores(y_train, model.predict(X_train)), "train")
        val = scores(y_val, model.predict(X_val))
        log_scores(val, "val")
    return val


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--split-file", type=Path, help="default: splits/<data-dir name>.csv")
    args = p.parse_args()
    split_file = args.split_file or Path("splits") / f"{args.data_dir.name}.csv"
    experiment = f"{args.data_dir.name}-baselines"

    train = WikiArtSplit(args.data_dir, split_file, "train")
    val = WikiArtSplit(args.data_dir, split_file, "val")
    y_train, y_val = train.labels.numpy(), val.labels.numpy()
    results = {}

    majority = DummyClassifier(strategy="most_frequent")  # always answers the most frequent train class
    results["majority"] = fit_and_log(experiment, "majority", {"model": "majority"}, majority,
                                      train.ids, y_train, val.ids, y_val, args.data_dir)

    for name, featurize in FEATURES.items():
        X_train, X_val = featurize(train.images), featurize(val.images)
        for C in C_GRID:
            params = {"model": "logreg", "features": name, "n_features": X_train.shape[1], "C": C}
            results[f"logreg-{name} C={C}"] = fit_and_log(experiment, f"logreg-{name}-C{C}", params,
                                                         logistic_regression(C), X_train, y_train, X_val, y_val,
                                                         args.data_dir)

    print(f"\n{'run':24s} {'val acc':>8s} {'val F1':>8s}")
    for run, val_scores in results.items():
        print(f"{run:24s} {val_scores['accuracy']:8.3f} {val_scores['macro_f1']:8.3f}")


if __name__ == "__main__":
    main()
