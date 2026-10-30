"""The small CNN outputs one logit per class, starts at loss ln K, and its size does not depend on the image size."""
import math

import torch
from torch import nn

from artguide.models.cnn import SmallCNN
from artguide.models.mlp import n_parameters

N_CLASSES = 5


def test_one_logit_per_class_at_any_input_size():
    model = SmallCNN(N_CLASSES)
    for size in (64, 128):
        assert model(torch.randn(4, 3, size, size)).shape == (4, N_CLASSES)


def test_parameter_count():
    # per block: 3x3 conv weights (no bias) + BatchNorm scale and shift; then Linear(256 -> 5)
    widths = [3, 32, 64, 128, 256]
    convs = sum(9 * c_in * c_out + 2 * c_out for c_in, c_out in zip(widths, widths[1:]))
    assert n_parameters(SmallCNN(N_CLASSES)) == convs + 256 * N_CLASSES + N_CLASSES


def test_initial_loss_is_ln_k():
    torch.manual_seed(0)
    model = SmallCNN(N_CLASSES).eval()
    loss = nn.CrossEntropyLoss()(model(torch.randn(64, 3, 64, 64)), torch.randint(0, N_CLASSES, (64,)))
    assert abs(loss.item() - math.log(N_CLASSES)) < 0.01


def test_dropout_only_before_the_head():
    assert not any(isinstance(m, nn.Dropout) for m in SmallCNN(N_CLASSES).modules())
    assert isinstance(SmallCNN(N_CLASSES, dropout=0.3).head[1], nn.Dropout)
