"""Classification metrics shared by every model (baselines, MLP, ResNet), so that all are scored by the same code;
bootstrap confidence intervals for the final test-set pass."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def scores(y_true, y_pred) -> dict[str, float]:
    """Accuracy and macro-F1: the two numbers logged for every model."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def confusion(y_true, y_pred, n_classes: int) -> np.ndarray:
    """(n_classes, n_classes) counts: row = true class, column = predicted class."""
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


def bootstrap_ci(y_true, y_pred, metric: str = "accuracy", n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> tuple[float, float, float]:
    """Score and its (1 - alpha) confidence interval: re-draw the test images with replacement n_boot times and
    take the percentiles of the scores. It answers "how much would this number move with other test images?"."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    samples = [rng.integers(0, len(y_true), len(y_true)) for _ in range(n_boot)]
    boot = [scores(y_true[i], y_pred[i])[metric] for i in samples]
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return scores(y_true, y_pred)[metric], float(lo), float(hi)


def paired_bootstrap(y_true, pred_a, pred_b, metric: str = "accuracy", n_boot: int = 2000, alpha: float = 0.05,
                     seed: int = 0) -> tuple[float, float, float]:
    """Difference score(a) - score(b) and its confidence interval, both models scored on the SAME resampled
    images. If the interval contains 0, the test set cannot tell the two models apart."""
    y_true, pred_a, pred_b = np.asarray(y_true), np.asarray(pred_a), np.asarray(pred_b)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y_true), len(y_true))
        diffs.append(scores(y_true[i], pred_a[i])[metric] - scores(y_true[i], pred_b[i])[metric])
    point = scores(y_true, pred_a)[metric] - scores(y_true, pred_b)[metric]
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)
