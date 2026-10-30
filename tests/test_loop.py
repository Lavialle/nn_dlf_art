"""The hand-written loop learns, and evaluation measures without learning."""
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from artguide.models.mlp import MLP
from artguide.seed import seed_everything
from artguide.train.loop import evaluate, train_one_epoch

CPU = torch.device("cpu")


def toy_data(n: int = 200) -> TensorDataset:
    """Two easy classes: images whose pixels are mostly positive vs mostly negative."""
    y = torch.arange(n) % 2
    x = torch.randn(n, 3, 8, 8) + (2 * y - 1).view(-1, 1, 1, 1).float()
    return TensorDataset(x, y)


def test_training_reduces_the_loss():
    seed_everything(0)
    data, model = toy_data(), MLP(3 * 8 * 8, 2, hidden=(16,))
    loader = DataLoader(data, batch_size=32, shuffle=True)
    loss_fn, optimizer = nn.CrossEntropyLoss(), torch.optim.SGD(model.parameters(), lr=0.1)
    first = train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    for _ in range(9):
        last = train_one_epoch(model, loader, loss_fn, optimizer, CPU)
    assert last < first / 2


def test_evaluate_does_not_change_the_weights():
    seed_everything(0)
    model = MLP(3 * 8 * 8, 2, hidden=(16,), batchnorm=True, dropout=0.5)
    before = {k: v.clone() for k, v in model.state_dict().items()}  # includes BatchNorm's running statistics
    evaluate(model, DataLoader(toy_data(), batch_size=32), nn.CrossEntropyLoss(), CPU)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_evaluate_loss_does_not_depend_on_batch_size():
    seed_everything(0)
    data, model, loss_fn = toy_data(), MLP(3 * 8 * 8, 2, hidden=(16,)), nn.CrossEntropyLoss()
    loss_7, labels, preds = evaluate(model, DataLoader(data, batch_size=7), loss_fn, CPU)  # last batch smaller
    loss_200, _, _ = evaluate(model, DataLoader(data, batch_size=200), loss_fn, CPU)
    assert abs(loss_7 - loss_200) < 1e-6
    assert len(labels) == len(preds) == len(data)
