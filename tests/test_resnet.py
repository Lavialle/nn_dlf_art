"""ResNet18 fine-tuning levels: what is trainable, BatchNorm of frozen layers, LR groups, initial loss."""
import math

import pytest
import torch
from torch import nn

from artguide.models.resnet import ResNetClassifier


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    return {ft: ResNetClassifier(5, finetune=ft) for ft in ("probe", "partial", "full")}


def trainable(model):
    return {name.split(".")[1] for name, p in model.named_parameters() if p.requires_grad}


def test_what_learns_at_each_level(models):
    assert trainable(models["probe"]) == {"fc"}
    assert trainable(models["partial"]) == {"layer4", "fc"}
    assert trainable(models["full"]) == {"conv1", "bn1", "layer1", "layer2", "layer3", "layer4", "fc"}


def test_frozen_batchnorm_stays_in_eval_mode(models):
    probe = models["probe"].train()
    assert all(not m.training for m in probe.modules() if isinstance(m, nn.BatchNorm2d))
    partial = models["partial"].train()
    assert partial.net.layer4[0].bn1.training and not partial.net.layer3[0].bn1.training


def test_lr_groups(models):
    groups = models["full"].param_groups(lr=1e-3, backbone_lr=1e-4)
    assert [g["lr"] for g in groups] == [1e-3, 1e-4]
    assert len(models["probe"].param_groups(lr=1e-3, backbone_lr=1e-4)) == 1  # nothing else to train


def test_initial_loss_is_ln_k(models):
    x, y = torch.randn(16, 3, 224, 224), torch.randint(0, 5, (16,))
    model = models["probe"].eval()
    with torch.no_grad():
        loss = nn.CrossEntropyLoss()(model(x), y).item()
    assert loss == pytest.approx(math.log(5), abs=0.05)
