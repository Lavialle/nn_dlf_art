"""Train a model (MLP or ResNet) from a YAML config; log every epoch in MLflow. The test set is not touched.

    uv run python scripts/train.py --config configs/style/mlp_v0.yaml
    uv run python scripts/train.py --config configs/style/mlp_v0.yaml --seeds 0 1 2
    uv run python scripts/train.py --config configs/style/mlp_v3.yaml --set dropout=0.2   (run name: mlp_v3-dropout0.2)

The kept model is the epoch with the lowest val loss (not the last one); its weights go to checkpoints/.
For the MLP, the monitors of diagnostics/monitors.py are logged at every epoch too.
"""
from __future__ import annotations

import argparse
import dataclasses
import math
import statistics
import time

import mlflow
import torch
from torch import nn

from artguide.config import Config, load_config, parse_overrides
from artguide.diagnostics.monitors import TrainingMonitor, activation_stats
from artguide.evaluation.metrics import scores
from artguide.models.mlp import n_parameters
from artguide.seed import seed_everything
from artguide.tracking import log_scores, start_run
from artguide.train.builders import get_device, make_loaders, make_model, make_model_optimizer
from artguide.train.early_stopping import EarlyStopping
from artguide.train.loop import evaluate, train_one_epoch
from artguide.train.optim import make_scheduler

N_MONITOR_IMAGES = 512  # fixed training images on which the activations are described at every epoch


def train(cfg: Config, config_file: str) -> dict[str, float] | None:
    """One run: train, keep the best val-loss epoch, save it, log everything. Returns its val scores
    (None if the run diverged before any epoch could be kept)."""
    device = get_device()
    generator = seed_everything(cfg.seed)
    loaders = make_loaders(cfg, generator)
    model = make_model(cfg, len(loaders.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)  # what the optimizer minimises
    eval_loss_fn = nn.CrossEntropyLoss()  # what we report: plain cross-entropy, comparable across versions
    optimizer = make_model_optimizer(cfg, model)
    scheduler = make_scheduler(optimizer, cfg.scheduler, cfg.epochs)
    stopper = EarlyStopping(cfg.patience)
    eval_mode = cfg.failure != "forget_eval"  # provoked failure: validation without model.eval()
    monitor, monitor_x = None, None
    if cfg.model == "mlp":
        monitor = TrainingMonitor(model)
        dataset = loaders.train_eval.dataset
        monitor_x = torch.stack([dataset[i][0] for i in range(min(N_MONITOR_IMAGES, len(dataset)))]).to(device)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    params = {**dataclasses.asdict(cfg), "config_file": config_file, "n_parameters": n_parameters(model),
              "n_trainable": n_trainable}
    best_scores = None
    with start_run(cfg.experiment, f"{cfg.name}-seed{cfg.seed}", params, data_dir=cfg.data_dir):
        for epoch in range(1, cfg.epochs + 1):
            t0 = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]  # the LR used during this epoch (the head's, for ResNet)
            train_loss = train_one_epoch(model, loaders.train, loss_fn, optimizer, device, cfg.max_grad_norm,
                                         monitor)
            if scheduler is not None:
                scheduler.step()  # once per epoch, after the optimizer steps
            _, y_train, p_train = evaluate(model, loaders.train_eval, eval_loss_fn, device, eval_mode)
            val_loss, y_val, p_val = evaluate(model, loaders.val, eval_loss_fn, device, eval_mode)
            val_scores = scores(y_val, p_val)

            log_scores({"loss": train_loss, **scores(y_train, p_train)}, "train", step=epoch)
            log_scores({"loss": val_loss, **val_scores}, "val", step=epoch)
            log_scores({"seconds": time.perf_counter() - t0, "lr": lr}, "epoch", step=epoch)
            if monitor is not None:
                mlflow.log_metrics({**monitor.epoch_stats(), **activation_stats(model, monitor_x)}, step=epoch)
            if stopper.step(val_loss, epoch, model):
                best_scores = val_scores
            print(f"epoch {epoch:3d} | lr {lr:.1e} | train loss {train_loss:.3f} | val loss {val_loss:.3f}"
                  f" acc {val_scores['accuracy']:.3f} F1 {val_scores['macro_f1']:.3f}"
                  f"{'  *' if stopper.best_epoch == epoch else ''}")  # * = new best val loss
            if not math.isfinite(train_loss):
                mlflow.set_tag("diverged", "True")
                print("Diverged: the training loss is no longer a number")
                break
            if stopper.should_stop:
                print(f"Early stop: no better val loss for {cfg.patience} epochs")
                break

        log_scores({"epochs": epoch}, "trained")
        if best_scores is None:
            return None
        cfg.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": stopper.best_state, "config": dataclasses.asdict(cfg),
                    "classes": loaders.classes, "epoch": stopper.best_epoch}, cfg.checkpoint_path)
        mlflow.set_tag("checkpoint", cfg.checkpoint_path.as_posix())
        log_scores({"loss": stopper.best_loss, **best_scores, "epoch": stopper.best_epoch}, "best_val")
        print(f"\nBest val loss {stopper.best_loss:.3f} at epoch {stopper.best_epoch}: accuracy"
              f" {best_scores['accuracy']:.3f}, macro-F1 {best_scores['macro_f1']:.3f} -> {cfg.checkpoint_path}")
    return best_scores


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", help="override the seed of the config; one run per seed")
    p.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                   help="override config values, e.g. --set dropout=0.2 lr=1.0e-3 (the run name gets a suffix)")
    args = p.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    seeds = args.seeds or [cfg.seed]

    results = []
    for seed in seeds:
        print(f"\n=== {cfg.name}, seed {seed} ===")
        result = train(dataclasses.replace(cfg, seed=seed), args.config)
        if result is not None:
            results.append(result)
    if len(results) > 1:
        print(f"\n{cfg.name} over seeds {seeds} (model kept = best val loss):")
        for metric in ("accuracy", "macro_f1"):
            values = [r[metric] for r in results]
            print(f"  val {metric}: {statistics.mean(values):.3f} ± {statistics.stdev(values):.3f}")


if __name__ == "__main__":
    main()
