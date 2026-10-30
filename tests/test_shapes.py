"""The MLP builds the layers its config asks for, and outputs one logit per class."""
import pytest
import torch
from torch import nn

from artguide.models.mlp import MLP, n_parameters

IN_FEATURES, N_CLASSES = 3 * 64 * 64, 5
BATCH = torch.randn(8, 3, 64, 64)


@pytest.mark.parametrize("config", [
    {},
    {"hidden": (512, 256)},
    {"batchnorm": True, "dropout": 0.3},
    {"hidden": (64,) * 8, "activation": "sigmoid", "init": "xavier"},
    {"hidden": ()},  # no hidden layer: a multinomial logistic regression
])
def test_one_logit_per_class(config):
    assert MLP(IN_FEATURES, N_CLASSES, **config)(BATCH).shape == (8, N_CLASSES)


def test_v0_parameter_count():
    # Linear(12288 -> 512): weights + biases, then Linear(512 -> 5)
    assert n_parameters(MLP(IN_FEATURES, N_CLASSES)) == 12288 * 512 + 512 + 512 * 5 + 5


def test_layer_order():
    model = MLP(IN_FEATURES, N_CLASSES, batchnorm=True, dropout=0.3)
    types = [type(m) for m in model.net]
    assert types == [nn.Flatten, nn.Linear, nn.BatchNorm1d, nn.ReLU, nn.Dropout, nn.Linear]


def test_outputs_logits_not_probabilities():
    assert isinstance(MLP(IN_FEATURES, N_CLASSES).net[-1], nn.Linear)  # no softmax at the end


def test_zero_init_makes_all_neurons_identical():
    model = MLP(IN_FEATURES, N_CLASSES, init="zeros")
    assert all(torch.all(m.weight == 0) for m in model.modules() if isinstance(m, nn.Linear))


def test_unknown_init_is_rejected():
    with pytest.raises(ValueError):
        MLP(IN_FEATURES, N_CLASSES, init="he")
