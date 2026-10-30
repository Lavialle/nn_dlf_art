"""Kiosk simulation: CPU latency of one prediction and size of the model on disk."""
from __future__ import annotations

import io
import pickle
import statistics
import time
from collections.abc import Callable

import torch
from torch import nn

KIOSK_THREADS = 2  # a modest kiosk CPU: two cores


def measure_latency(predict: Callable[[torch.Tensor], object], image: torch.Tensor, threads: int = KIOSK_THREADS,
                    warmup: int = 10, repeats: int = 100) -> dict[str, float]:
    """Milliseconds to answer for ONE image (batch of 1, as a visitor's photo), on the CPU with `threads` threads.

    The first calls are slower (memory allocation, caches): they are run but not counted. Median and 95th
    percentile, since a few calls are slowed down by the operating system.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        for _ in range(warmup):
            predict(image)
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            predict(image)
            times.append((time.perf_counter() - t0) * 1000)
    finally:
        torch.set_num_threads(previous)
    return {"median_ms": statistics.median(times), "p95_ms": sorted(times)[int(0.95 * len(times)) - 1]}


def size_mb(model: object) -> float:
    """Size of the saved model in MB: the weights of a torch model, the pickled object of a sklearn one."""
    buffer = io.BytesIO()
    if isinstance(model, nn.Module):
        torch.save(model.state_dict(), buffer)
    else:
        pickle.dump(model, buffer)
    return buffer.getbuffer().nbytes / 1e6
