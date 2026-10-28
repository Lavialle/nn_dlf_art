"""ResNet18 pre-trained on ImageNet, with a new 5-class head, at three levels of fine-tuning.

- probe: the backbone is frozen, only the head (fc, 512 -> 5) learns: "are ImageNet features enough?";
- partial: the last block (layer4) and the head learn;
- full: everything learns, the backbone with a smaller LR than the head (differential learning rates).
"""
from __future__ import annotations

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from artguide.models.mlp import OUTPUT_INIT_STD

FINETUNE = {"probe": ("fc",), "partial": ("layer4", "fc"), "full": None}  # trainable top-level blocks (None = all)
WEIGHTS = ResNet18_Weights.IMAGENET1K_V1


class ResNetClassifier(nn.Module):
    def __init__(self, n_classes: int, finetune: str = "probe", pretrained: bool = True):
        super().__init__()
        if finetune not in FINETUNE:
            raise ValueError(f"unknown finetune {finetune!r}: expected one of {list(FINETUNE)}")
        self.net = resnet18(weights=WEIGHTS if pretrained else None)
        self.net.fc = nn.Linear(self.net.fc.in_features, n_classes)  # replaces the 1000 ImageNet classes
        nn.init.normal_(self.net.fc.weight, std=OUTPUT_INIT_STD)  # as for the MLP: initial loss = ln K
        nn.init.zeros_(self.net.fc.bias)
        trainable = FINETUNE[finetune]
        for name, p in self.net.named_parameters():
            p.requires_grad = trainable is None or name.split(".")[0] in trainable

    def train(self, mode: bool = True) -> ResNetClassifier:
        """Frozen BatchNorm layers stay in eval mode: they keep ImageNet's statistics instead of drifting
        towards our batches while their weights cannot adapt."""
        super().train(mode)
        for m in self.net.modules():
            if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
                m.eval()
        return self

    def forward(self, x):
        return self.net(x)

    def param_groups(self, lr: float, backbone_lr: float) -> list[dict]:
        """Head at lr, trainable backbone layers at backbone_lr (pre-trained weights only need small changes)."""
        head = list(self.net.fc.parameters())
        backbone = [p for n, p in self.net.named_parameters() if p.requires_grad and not n.startswith("fc.")]
        return [{"params": head, "lr": lr}] + ([{"params": backbone, "lr": backbone_lr}] if backbone else [])


def imagenet_stats() -> tuple[list[float], list[float]]:
    """The per-channel mean and std the pre-trained weights expect (ImageNet's, not ours)."""
    t = WEIGHTS.transforms()
    return list(t.mean), list(t.std)


def embedding_model() -> nn.Module:
    """The frozen pre-trained backbone without its head: image -> 512 numbers (for a logistic regression)."""
    net = resnet18(weights=WEIGHTS)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad = False
    return net.eval()
