"""Small CNN trained from scratch: [Conv 3x3 -> BatchNorm -> ReLU -> MaxPool 2] per block -> global average pool
-> (Dropout) -> Linear (logits).

It sits between the MLP and the pre-trained ResNet: same 64 px input and training recipe as the MLP, but
convolutions (one small filter shared over the whole image) instead of one weight per pixel, and no pre-training.
"""
from __future__ import annotations

from torch import nn

from artguide.models.mlp import OUTPUT_INIT_STD


class SmallCNN(nn.Module):
    def __init__(self, n_classes: int, channels: tuple[int, ...] = (32, 64, 128, 256), dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        width = 3  # RGB
        for size in channels:
            layers += [nn.Conv2d(width, size, kernel_size=3, padding=1, bias=False),  # bias: BatchNorm has one
                       nn.BatchNorm2d(size), nn.ReLU(), nn.MaxPool2d(2)]  # 2x smaller image, 2x more channels
            width = size
        self.features = nn.Sequential(*layers)
        # average of each channel over the image: no Linear(12288 -> ...) as in the MLP, and the size of the
        # input does not change the number of parameters
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(nn.Flatten(), *([nn.Dropout(dropout)] if dropout > 0 else []),
                                  nn.Linear(width, n_classes))  # raw logits
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")  # as torchvision's ResNet
        nn.init.normal_(self.head[-1].weight, std=OUTPUT_INIT_STD)  # initial loss = ln K, as for the MLP
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))
