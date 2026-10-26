"""Metrics checked against a small example computed by hand."""
import pytest

from artguide.evaluation.metrics import confusion, scores

# A lazy model that predicts class 0 for almost everything.
Y_TRUE = [0, 0, 0, 0, 1, 1, 2, 2]
Y_PRED = [0, 0, 0, 0, 0, 0, 0, 2]


def test_accuracy_is_share_of_correct_predictions():
    assert scores(Y_TRUE, Y_PRED)["accuracy"] == pytest.approx(5 / 8)


def test_macro_f1_averages_per_class_f1():
    f1_class_0 = 2 * (4 / 7) * 1 / (4 / 7 + 1)  # precision 4/7, recall 4/4
    f1_class_1 = 0.0                             # never predicted
    f1_class_2 = 2 * 1 * (1 / 2) / (1 + 1 / 2)   # precision 1/1, recall 1/2
    expected = (f1_class_0 + f1_class_1 + f1_class_2) / 3
    assert scores(Y_TRUE, Y_PRED)["macro_f1"] == pytest.approx(expected)


def test_confusion_rows_are_true_classes():
    assert confusion(Y_TRUE, Y_PRED, n_classes=3).tolist() == [
        [4, 0, 0],  # the four 0s: all predicted 0
        [2, 0, 0],  # the two 1s: both predicted 0
        [1, 0, 1],  # the two 2s: one predicted 0, one predicted 2
    ]
