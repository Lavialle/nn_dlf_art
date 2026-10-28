"""Run the sanity checks of a config before training it for real; log them in MLflow.

    uv run python scripts/sanity_checks.py --config configs/style/mlp_v0.yaml

Check 1: the initial loss is close to ln K (the model starts by "knowing nothing", honestly).
Check 2: the model can memorise a single batch (loss -> ~0, accuracy -> 100 %).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from artguide.config import load_config
from artguide.diagnostics.sanity import (INITIAL_LOSS_TOLERANCE, OVERFIT_MAX_LOSS, expected_initial_loss,
                                         overfit_one_batch)
from artguide.seed import seed_everything
from artguide.tracking import log_scores, start_run
from artguide.train.builders import get_device, make_loaders, make_model
from artguide.train.loop import evaluate
from artguide.train.optim import make_optimizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)

    device = get_device()
    generator = seed_everything(cfg.seed)
    loaders = make_loaders(cfg, generator)
    train_loader, classes = loaders.train, loaders.classes
    model = make_model(cfg, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model.parameters(), cfg.optimizer, cfg.lr, cfg.momentum, cfg.nesterov,
                               cfg.weight_decay)

    experiment = f"{Path(cfg.data_dir).name}-sanity"
    with start_run(experiment, f"{cfg.name}-seed{cfg.seed}", {"config_file": args.config, "lr": cfg.lr},
                   data_dir=cfg.data_dir):
        # Check 1: loss before any training, on the whole training set
        initial, _, _ = evaluate(model, train_loader, loss_fn, device)
        expected = expected_initial_loss(len(classes))
        ok_initial = abs(initial - expected) < INITIAL_LOSS_TOLERANCE
        log_scores({"loss": initial, "expected_loss": expected, "passed": float(ok_initial)}, "initial")
        print(f"Check 1 - initial loss {initial:.3f} vs ln {len(classes)} = {expected:.3f}: "
              f"{'OK' if ok_initial else 'FAILED'}")

        # Check 2: memorise one batch
        x, y = next(iter(train_loader))
        losses, accuracy = overfit_one_batch(model, x.to(device), y.to(device), loss_fn, optimizer)
        for step, loss in enumerate(losses, start=1):
            log_scores({"loss": loss}, "overfit_batch", step=step)
        ok_overfit = losses[-1] < OVERFIT_MAX_LOSS and accuracy == 1.0
        log_scores({"accuracy": accuracy, "passed": float(ok_overfit)}, "overfit_batch")
        print(f"Check 2 - one batch of {len(y)} after {len(losses)} steps: loss {losses[-1]:.4f}, "
              f"accuracy {accuracy:.0%}: {'OK' if ok_overfit else 'FAILED'}")


if __name__ == "__main__":
    main()
