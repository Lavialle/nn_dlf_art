"""LR range test of a config: find a good learning rate by measuring, not guessing. Saves a figure, logs to MLflow.

    uv run python scripts/lr_range_test.py --config configs/style/mlp_v0.yaml

The learning rate grows exponentially while training; the loss first falls faster and faster, then explodes.
Suggested LR = a tenth of the LR where the loss is lowest. Run on several seeds: the minimum moves a little.
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib
import mlflow

matplotlib.use("Agg")  # draw to a file, no window
import matplotlib.pyplot as plt
from torch import nn

from artguide.config import load_config
from artguide.diagnostics.sanity import lr_range_test, suggest_lr
from artguide.seed import seed_everything
from artguide.tracking import log_scores, start_run
from artguide.train.builders import get_device, make_loaders, make_model, make_model_optimizer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = p.parse_args()
    cfg = load_config(args.config)
    subset = Path(cfg.data_dir).name
    figure = Path("reports/figures") / subset / f"lr_range_{cfg.name}.png"
    device = get_device()

    fig, ax = plt.subplots(figsize=(7, 4))
    suggestions, lowest, start = [], [], []
    for seed in args.seeds:
        generator = seed_everything(seed)
        loaders = make_loaders(cfg, generator)
        model = make_model(cfg, len(loaders.classes)).to(device)  # a fresh model: the test destroys its weights
        optimizer = make_model_optimizer(cfg, model)
        loss_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        lrs, losses = lr_range_test(model, loaders.train, loss_fn, optimizer, device)
        suggestions.append(suggest_lr(lrs, losses))
        lowest.append(min(losses))
        start.append(losses[0])
        ax.plot(lrs, losses, label=f"seed {seed}")
        print(f"seed {seed}: suggested lr {suggestions[-1]:.1e}")

    suggested = statistics.median(suggestions)
    ax.axvline(suggested, color="black", linestyle="--", label=f"suggested {suggested:.1e}")
    ax.set(xscale="log", xlabel="learning rate (log scale)", ylabel="smoothed training loss",
           title=f"LR range test - {cfg.name} ({cfg.optimizer})",
           ylim=(0.95 * min(lowest), 1.15 * max(start)))  # zoom on the useful zone, not on the explosion
    ax.legend()
    figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure, dpi=150, bbox_inches="tight")

    with start_run(f"{subset}-sanity", f"{cfg.name}-lr-range", {"config_file": args.config,
                                                                "optimizer": cfg.optimizer,
                                                                "seeds": args.seeds}, data_dir=cfg.data_dir):
        log_scores({"lr": suggested, **{f"lr_seed{s}": v for s, v in zip(args.seeds, suggestions)}}, "suggested")
        mlflow.log_artifact(str(figure))
    print(f"\nSuggested learning rate (median over seeds): {suggested:.1e}  -> figure: {figure}")


if __name__ == "__main__":
    main()
