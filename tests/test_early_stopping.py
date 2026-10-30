"""Early stopping and LR schedules, on fake val losses and a tiny model: no dataset needed."""
import pytest
import torch
from torch import nn

from artguide.train.early_stopping import EarlyStopping
from artguide.train.optim import make_scheduler


def test_keeps_the_best_epoch_and_stops_after_patience():
    model = nn.Linear(2, 1)
    stopper = EarlyStopping(patience=3)
    stopped_at = None
    for epoch, val_loss in enumerate([1.0, 0.8, 0.9, 0.85, 0.95, 0.7], start=1):
        with torch.no_grad():
            model.weight.fill_(epoch)  # the weights change at every epoch, like during training
        stopper.step(val_loss, epoch, model)
        if stopper.should_stop:
            stopped_at = epoch
            break
    assert stopped_at == 5  # epochs 3, 4, 5 did not beat 0.8: the 0.7 of epoch 6 is never seen
    assert stopper.best_epoch == 2 and stopper.best_loss == 0.8
    assert torch.all(stopper.best_state["weight"] == 2)  # a copy of epoch 2, not the current weights (5)


def test_without_patience_it_never_stops():
    stopper = EarlyStopping(patience=None)
    for epoch in range(1, 51):
        stopper.step(1.0 + epoch, epoch, nn.Linear(2, 1))  # the loss only gets worse
    assert not stopper.should_stop
    assert stopper.best_epoch == 1


def test_cosine_goes_from_lr_to_zero():
    optimizer = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1)
    scheduler = make_scheduler(optimizer, "cosine", epochs=10)
    lrs = []
    for _ in range(10):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    assert lrs[0] == pytest.approx(0.1)
    assert lrs[5] == pytest.approx(0.05)  # half-way down the cosine
    assert all(b < a for a, b in zip(lrs, lrs[1:]))
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-9)


def test_no_scheduler_and_unknown_scheduler():
    optimizer = torch.optim.SGD(nn.Linear(2, 1).parameters(), lr=0.1)
    assert make_scheduler(optimizer, "none", epochs=10) is None
    with pytest.raises(ValueError, match="step"):
        make_scheduler(optimizer, "step", epochs=10)
