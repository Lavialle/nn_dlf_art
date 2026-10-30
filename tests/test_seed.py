"""Same seed -> same initial weights and same batch order; different seed -> different."""
import torch
from torch import nn

from artguide.seed import seed_everything


def initial_state(seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = seed_everything(seed)
    weights = nn.Linear(10, 5).weight.detach().clone()
    batch_order = torch.randperm(100, generator=generator)  # what a shuffling DataLoader does
    return weights, batch_order


def test_same_seed_same_start():
    w1, order1 = initial_state(0)
    w2, order2 = initial_state(0)
    assert torch.equal(w1, w2)
    assert torch.equal(order1, order2)


def test_different_seed_different_start():
    w1, order1 = initial_state(0)
    w2, order2 = initial_state(1)
    assert not torch.equal(w1, w2)
    assert not torch.equal(order1, order2)


def test_batch_order_independent_of_model_size():
    _, order_small = initial_state(0)
    generator = seed_everything(0)
    nn.Linear(1000, 1000)  # a bigger model consumes many more random numbers from the global generator
    assert torch.equal(torch.randperm(100, generator=generator), order_small)
