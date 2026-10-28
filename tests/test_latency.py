"""Latency and size measurements on tiny models."""
import pytest
import torch
from sklearn.linear_model import LogisticRegression

from artguide.evaluation.latency import measure_latency, size_mb
from artguide.models.mlp import MLP


def test_latency_is_positive_and_restores_the_thread_count():
    before = torch.get_num_threads()
    model = MLP(3 * 8 * 8, 5, hidden=(16,)).eval()
    stats = measure_latency(model, torch.randn(1, 3, 8, 8), threads=1, warmup=2, repeats=20)
    assert 0 < stats["median_ms"] <= stats["p95_ms"]
    assert torch.get_num_threads() == before


def test_torch_size_is_four_bytes_per_parameter():
    model = MLP(3 * 64 * 64, 5, hidden=(512,))
    n = sum(p.numel() for p in model.parameters())
    assert size_mb(model) == pytest.approx(n * 4 / 1e6, rel=0.01)  # float32 weights, plus a small header


def test_sklearn_model_size():
    model = LogisticRegression().fit([[0.0, 1.0], [1.0, 0.0]], [0, 1])
    assert 0 < size_mb(model) < 0.01
