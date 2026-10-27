"""Sanity checks on synthetic data: no dataset needed, so they run anywhere in a few seconds."""
import pytest
import torch
from torch import nn

from artguide.diagnostics.sanity import expected_initial_loss, overfit_one_batch
from artguide.models.mlp import MLP
from artguide.seed import seed_everything

N_CLASSES = 5


def test_initial_loss_is_ln_k():
    seed_everything(0)
    x, y = torch.randn(512, 3, 16, 16), torch.randint(0, N_CLASSES, (512,))
    model = MLP(3 * 16 * 16, N_CLASSES, hidden=(64,))
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(x), y).item()
    assert loss == pytest.approx(expected_initial_loss(N_CLASSES), abs=0.05)


def test_mlp_memorises_one_batch_even_with_random_labels():
    seed_everything(0)
    x, y = torch.randn(64, 3, 16, 16), torch.randint(0, N_CLASSES, (64,))  # pure noise: nothing to "understand"
    model = MLP(3 * 16 * 16, N_CLASSES, hidden=(64,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    losses, accuracy = overfit_one_batch(model, x, y, nn.CrossEntropyLoss(), optimizer, steps=300)
    assert losses[-1] < 0.05
    assert accuracy == 1.0


def test_broken_pipeline_cannot_memorise():
    seed_everything(0)
    x, y = torch.randn(64, 3, 16, 16), torch.randint(0, N_CLASSES, (64,))
    model = MLP(3 * 16 * 16, N_CLASSES, hidden=(64,))
    for p in model.parameters():
        p.requires_grad_(False)  # a classic bug: the weights are frozen, gradients never reach them
    model.net[-1].bias.requires_grad_(True)  # only the 5 output biases can learn
    optimizer = torch.optim.SGD([model.net[-1].bias], lr=0.05)
    losses, accuracy = overfit_one_batch(model, x, y, nn.CrossEntropyLoss(), optimizer, steps=300)
    assert losses[-1] > 1.0  # stuck: the check catches the bug
