"""Train an MLP from a YAML config; log every epoch in MLflow. The test set is not touched.

    uv run python scripts/train.py --config configs/style/mlp_v0.yaml
    uv run python scripts/train.py --config configs/style/mlp_v0.yaml --seed 1
"""
from __future__ import annotations

import argparse
import dataclasses
import time

from torch import nn

from artguide.config import load_config
from artguide.evaluation.metrics import scores
from artguide.models.mlp import n_parameters
from artguide.seed import seed_everything
from artguide.tracking import log_scores, start_run
from artguide.train.builders import get_device, make_loaders, make_model
from artguide.train.loop import evaluate, train_one_epoch
from artguide.train.optim import make_optimizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, help="overrides the seed of the config")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg = dataclasses.replace(cfg, seed=args.seed)

    device = get_device()
    generator = seed_everything(cfg.seed)
    train_loader, val_loader, classes = make_loaders(cfg, generator)
    model = make_model(cfg, len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model.parameters(), cfg.optimizer, cfg.lr, cfg.momentum, cfg.nesterov,
                               cfg.weight_decay)

    params = {**dataclasses.asdict(cfg), "config_file": args.config, "n_parameters": n_parameters(model)}
    with start_run(cfg.experiment, f"{cfg.name}-seed{cfg.seed}", params, data_dir=cfg.data_dir):
        best_acc, best_epoch = 0.0, 0
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device, cfg.max_grad_norm)
            _, y_train, p_train = evaluate(model, train_loader, loss_fn, device)  # train scores in eval mode
            val_loss, y_val, p_val = evaluate(model, val_loader, loss_fn, device)
            val_scores = scores(y_val, p_val)

            log_scores({"loss": train_loss, **scores(y_train, p_train)}, "train", step=epoch)
            log_scores({"loss": val_loss, **val_scores}, "val", step=epoch)
            log_scores({"seconds": time.perf_counter() - t0}, "epoch", step=epoch)
            if val_scores["accuracy"] > best_acc:
                best_acc, best_epoch = val_scores["accuracy"], epoch
            print(f"epoch {epoch:3d} | train loss {train_loss:.3f} | val loss {val_loss:.3f}"
                  f" acc {val_scores['accuracy']:.3f} F1 {val_scores['macro_f1']:.3f}")

        log_scores({"accuracy": best_acc, "epoch": best_epoch}, "best_val")
        print(f"\nBest val accuracy {best_acc:.3f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()
