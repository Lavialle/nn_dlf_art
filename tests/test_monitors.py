"""Training monitors on a tiny MLP and random data: names, gradient norms, update ratios, dead ReLUs."""
import pytest
import torch
from torch import nn

from artguide.diagnostics.monitors import TrainingMonitor, activation_stats, linear_layers
from artguide.models.mlp import MLP
from artguide.seed import seed_everything


def tiny_mlp(**kwargs) -> MLP:
    seed_everything(0)
    return MLP(3 * 4 * 4, 5, hidden=(32, 16), **kwargs)


def test_layers_are_named_in_forward_order():
    assert list(linear_layers(tiny_mlp())) == ["fc1", "fc2", "out"]


def test_update_ratio_is_lr_times_gradient_over_weight_for_plain_sgd():
    model, lr = tiny_mlp(), 0.1
    x, y = torch.randn(64, 3, 4, 4), torch.randint(0, 5, (64,))
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    monitor = TrainingMonitor(model)
    optimizer.zero_grad()
    nn.CrossEntropyLoss()(model(x), y).backward()
    monitor.after_backward()
    grad_norm = model.net[1].weight.grad.norm().item()  # fc1
    optimizer.step()
    monitor.after_step()
    stats = monitor.epoch_stats()
    assert stats["grad_norm/fc1"] == pytest.approx(grad_norm)
    assert stats["update_ratio/fc1"] == pytest.approx(lr * grad_norm / model.net[1].weight.norm().item(), rel=1e-4)
    assert monitor.epoch_stats() == {}  # a new epoch starts empty


def test_dead_relus_are_counted():
    model, x = tiny_mlp(), torch.randn(256, 3, 4, 4)
    healthy = activation_stats(model, x)
    assert healthy["dead_relu/fc1"] < 0.1
    with torch.no_grad():
        model.net[1].bias.fill_(-1e6)  # every fc1 unit gets a huge negative input: ReLU outputs 0 for all images
    assert activation_stats(model, x)["dead_relu/fc1"] == 1.0
    assert model.training  # the monitor put the model back in its previous mode
