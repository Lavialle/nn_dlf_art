"""Classification metrics shared by every model (baselines, MLP, ResNet), so that all are scored by the same code."""
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
