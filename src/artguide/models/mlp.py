"""Configurable MLP: Flatten -> [Linear -> (BatchNorm) -> activation -> (Dropout)] per hidden layer -> Linear (logits)."""
from __future__ import annotations

from torch import nn

ACTIVATIONS = {"relu": nn.ReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh}
INITS = ("kaiming", "xavier", "default", "zeros")
OUTPUT_INIT_STD = 1e-3  # tiny output weights -> near-uniform first predictions -> initial loss ~ ln(n_classes)
# measured on the style train set (seeds 0-2): output std 1e-2 -> 1.62-1.64, 1e-3 -> 1.610 (ln 5 = 1.609)


class MLP(nn.Module):
    def __init__(self, in_features: int, n_classes: int, hidden: tuple[int, ...] = (512,), activation: str = "relu",
                 batchnorm: bool = False, dropout: float = 0.0, init: str = "kaiming"):
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]  # (N, 3, 64, 64) -> (N, 12288): the spatial structure is lost here
        width = in_features
        for size in hidden:
            layers.append(nn.Linear(width, size))
            if batchnorm:
                layers.append(nn.BatchNorm1d(size))
            layers.append(ACTIVATIONS[activation]())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            width = size
        layers.append(nn.Linear(width, n_classes))  # raw logits: CrossEntropyLoss applies log-softmax itself
        self.net = nn.Sequential(*layers)
        init_weights(self, init, activation)

    def forward(self, x):
        return self.net(x)


def init_weights(model: nn.Module, init: str, activation: str) -> None:
    """kaiming (for ReLU), xavier (for sigmoid/tanh), default (PyTorch's own), zeros (deliberate failure)."""
    if init not in INITS:
        raise ValueError(f"init must be one of {INITS}, got {init!r}")
    if init == "default":
        return
    linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
    for layer in linears:
        if init == "kaiming":
            nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
        elif init == "xavier":
            nn.init.xavier_normal_(layer.weight, gain=nn.init.calculate_gain(activation))
        else:
            nn.init.zeros_(layer.weight)  # every neuron identical: they all get the same gradient (symmetry)
        nn.init.zeros_(layer.bias)
    if init != "zeros":
        nn.init.normal_(linears[-1].weight, std=OUTPUT_INIT_STD)


def n_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
