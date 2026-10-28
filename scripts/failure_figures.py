"""Figures and summary table of the provoked failures: symptom -> evidence -> fix, read from MLflow.

    uv run python scripts/failure_figures.py --subset wikiart_style5

Run the failure configs first (configs/style/failures/, see docs). Writes one figure per failure in
reports/figures/<subset>/failures/ and the table reports/tables/<subset>_failures.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn

from artguide.config import load_config
from artguide.reporting import BLUE, INK, INK_MUTED, ORANGE, history, latest_per_name, mean_std, runs, style
from artguide.seed import seed_everything
from artguide.train.builders import get_device, make_loaders, make_model
from artguide.train.loop import evaluate

CONFIGS = Path("configs/style/failures")
REF = INK_MUTED  # the healthy reference is always drawn in grey, the failure in blue, its fix in orange


class Runs:
    """Latest seed-0 run of each failure config, and per-epoch histories on demand."""

    def __init__(self, subset: str):
        self.df = runs(f"{subset}-failures")

    def get(self, name: str, seed: int = 0):
        found = latest_per_name(self.df, [f"{name}-seed{seed}"])
        if found.empty:
            raise SystemExit(f"run {name}-seed{seed} not found: train configs/style/failures/{name}.yaml first")
        return found.iloc[0]

    def curve(self, name: str, metric: str, seed: int = 0) -> tuple[list[int], list[float]]:
        h = history(self.get(name, seed).run_id, metric)
        return list(h), list(h.values())


def two_panels(title: str) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=12)
    for ax in axes:
        style(ax)
    return fig, axes


def plot(ax, r: Runs, name: str, metric: str, color: str, label: str, **kw) -> None:
    epochs, values = r.curve(name, metric)
    ax.plot(epochs, values, color=color, label=label, linewidth=1.8, **kw)


def save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def first_value(r: Runs, name: str, metric: str) -> float:
    return r.curve(name, metric)[1][0]


def lr_too_high(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("1. Learning rate x 100: the loss explodes")
    for name, color, label in [("reference", REF, "reference (lr 2.5e-3)"), ("lr_x100", BLUE, "lr x 100 (0.25)"),
                               ("lr_x100_clipped", ORANGE, "lr x 100 + clipping at 1")]:
        plot(a, r, name, "train_loss", color, label, marker="o", markersize=4)  # lr x 100 has a single point
        plot(b, r, name, "grad_norm/fc1", color, label, marker="o", markersize=4)
    for ax, metric in [(a, "train_loss"), (b, "grad_norm/fc1")]:
        ax.annotate("NaN from epoch 2", (1, first_value(r, "lr_x100", metric)), xytext=(10, -4),
                    textcoords="offset points", color=BLUE, fontsize=8)
    a.set(yscale="log", xlabel="epoch", ylabel="training loss (log)", title="Symptom")
    b.set(yscale="log", xlabel="epoch", ylabel="gradient norm, 1st layer (log)", title="Evidence")
    a.legend(frameon=False)
    save(fig, out / "1_lr_x100.png")
    diverged = r.get("lr_x100").get("tags.diverged") == "True"
    return {"failure": "LR x 100", "symptom": "loss explodes to NaN" if diverged else "loss explodes",
            "evidence": f"1st-layer gradient norm {first_value(r, 'lr_x100', 'grad_norm/fc1'):.3g} at epoch 1 "
                        f"vs {first_value(r, 'reference', 'grad_norm/fc1'):.3g}",
            "fix": f"lower the LR (range test); clipping alone: val acc "
                   f"{r.get('lr_x100_clipped')['metrics.best_val_accuracy'] * 100:.0f} %"}


def lr_too_low(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("2. Learning rate / 1000: nothing moves")
    for name, color, label in [("reference", REF, "reference (lr 2.5e-3)"), ("lr_div1000", BLUE, "lr / 1000")]:
        plot(a, r, name, "train_loss", color, label)
        plot(b, r, name, "update_ratio/fc1", color, label)
    b.axhline(1e-3, color=INK, linestyle=":", linewidth=1)
    b.text(1, 1.3e-3, "healthy ~1e-3", color=INK, fontsize=8)
    a.set(xlabel="epoch", ylabel="training loss", title="Symptom")
    b.set(yscale="log", xlabel="epoch", ylabel="||Δw|| / ||w|| per step, 1st layer (log)", title="Evidence")
    a.legend(frameon=False)
    save(fig, out / "2_lr_div1000.png")
    _, losses = r.curve("lr_div1000", "train_loss")
    return {"failure": "LR / 1000", "symptom": f"training loss {losses[0]:.3f} -> {losses[-1]:.3f} in "
                                               f"{len(losses)} epochs",
            "evidence": f"update ratio {first_value(r, 'lr_div1000', 'update_ratio/fc1'):.1e} "
                        f"vs {first_value(r, 'reference', 'update_ratio/fc1'):.1e}",
            "fix": "LR from the range test"}


def zero_init(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("3. All weights initialised to zero: symmetry, nothing learns")
    for name, color, label in [("reference", REF, "reference (Kaiming init)"), ("zero_init", BLUE, "zero init")]:
        plot(a, r, name, "val_accuracy", color, label)
        plot(b, r, name, "act_std/fc1", color, label)
    a.set(xlabel="epoch", ylabel="val accuracy", title="Symptom")
    b.set(xlabel="epoch", ylabel="std of the 1st-layer activations", title="Evidence: all units identical")
    a.legend(frameon=False)
    save(fig, out / "3_zero_init.png")
    run = r.get("zero_init")
    return {"failure": "Zero initialisation",
            "symptom": f"val acc stuck at {run['metrics.val_accuracy'] * 100:.0f} %, "
                       f"macro-F1 {run['metrics.val_macro_f1'] * 100:.0f} % (one class predicted)",
            "evidence": f"activation std {r.curve('zero_init', 'act_std/fc1')[1][-1]:.1e}, "
                        f"1st-layer gradient {first_value(r, 'zero_init', 'grad_norm/fc1'):.1e}",
            "fix": "random init (Kaiming for ReLU) breaks the symmetry"}


def deep_sigmoid(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("4. 8 hidden layers with sigmoid, no BatchNorm: vanishing gradients")
    for name, color, label in [("deep_sigmoid", BLUE, "8 x sigmoid, Xavier"), ("deep_relu_bn", ORANGE,
                                                                                "8 x ReLU + BatchNorm, Kaiming")]:
        plot(a, r, name, "train_loss", color, label)
        layers = [f"fc{i}" for i in range(1, 9)] + ["out"]
        norms = [first_value(r, name, f"grad_norm/{layer}") for layer in layers]
        b.plot(layers, norms, color=color, marker="o", markersize=5, linewidth=1.8, label=label)
    a.set(xlabel="epoch", ylabel="training loss", title="Symptom")
    b.set(yscale="log", xlabel="layer (input -> output)", ylabel="gradient norm, epoch 1 (log)",
          title="Evidence: the gradient fades towards the input")
    a.legend(frameon=False)
    save(fig, out / "4_deep_sigmoid.png")
    sig = lambda layer: first_value(r, "deep_sigmoid", f"grad_norm/{layer}")  # noqa: E731
    return {"failure": "8 sigmoid layers, no BN", "symptom": "training loss flat at ln 5",
            "evidence": f"gradient norm {sig('out'):.2g} at the output, {sig('fc1'):.1g} at layer 1 "
                        f"(x{sig('out') / sig('fc1'):.0e})",
            "fix": "ReLU + Kaiming + BatchNorm: "
                   f"val acc {r.get('deep_relu_bn')['metrics.best_val_accuracy'] * 100:.0f} %"}


def overfitting(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("5. No regularisation, no early stopping, 200 epochs: overfitting")
    plot(a, r, "overfit_200", "train_loss", BLUE, "train", linestyle="--")
    plot(a, r, "overfit_200", "val_loss", BLUE, "val")
    run = r.get("overfit_200")
    best = int(run["metrics.best_val_epoch"])
    a.axvline(best, color=INK, linestyle=":", linewidth=1)
    a.text(best + 2, a.get_ylim()[1] * 0.9, f"lowest val loss: epoch {best}", color=INK, fontsize=8)
    epochs, train_acc = r.curve("overfit_200", "train_accuracy")
    _, val_acc = r.curve("overfit_200", "val_accuracy")
    b.plot(epochs, [100 * (t - v) for t, v in zip(train_acc, val_acc)], color=BLUE, linewidth=1.8)
    a.set(xlabel="epoch", ylabel="loss", title="Symptom: val loss climbs back")
    b.set(xlabel="epoch", ylabel="train - val accuracy (points)", title="Evidence: the generalisation gap")
    a.legend(frameon=False)
    save(fig, out / "5_overfitting.png")
    _, val_loss = r.curve("overfit_200", "val_loss")
    return {"failure": "No regularisation, 200 epochs",
            "symptom": f"val loss {run['metrics.best_val_loss']:.2f} at epoch {best} -> {val_loss[-1]:.2f} at "
                       f"epoch {len(val_loss)}",
            "evidence": f"train {train_acc[-1] * 100:.0f} % vs val {val_acc[-1] * 100:.0f} % at the end",
            "fix": "early stopping (keep the best epoch) + regularisation (v2-v6)"}


@torch.no_grad()
def eval_mode_spread(subset_cfg: Path, repeats: int = 10) -> tuple[list[float], list[float]]:
    """Val accuracy of the same trained model evaluated repeatedly with and without model.eval()."""
    cfg = load_config(subset_cfg)
    device = get_device()
    loaders = make_loaders(cfg, seed_everything(0))
    model = make_model(cfg, len(loaders.classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    accs = {}
    for eval_mode in (True, False):
        model.load_state_dict(torch.load(cfg.checkpoint_path)["state_dict"])  # train mode drifts BatchNorm stats
        accs[eval_mode] = []
        for _ in range(repeats):
            model.train()  # the state train_one_epoch leaves the model in
            _, y, p = evaluate(model, loaders.val, loss_fn, device, eval_mode=eval_mode)
            accs[eval_mode].append((y == p).float().mean().item() * 100)
    return accs[True], accs[False]


def forget_eval(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("6. model.eval() forgotten at validation (BatchNorm + dropout model)")
    plot(a, r, "reference_bn_dropout", "val_accuracy", REF, "with model.eval()")
    plot(a, r, "forget_eval", "val_accuracy", BLUE, "without model.eval()")
    with_eval, without = eval_mode_spread(CONFIGS / "reference_bn_dropout.yaml")
    for x, values, color in [(0, with_eval, REF), (1, without, BLUE)]:
        jitter = [x + 0.03 * (i - len(values) / 2) for i in range(len(values))]
        b.scatter(jitter, values, color=color, s=40, alpha=0.8, edgecolors="white", linewidths=1)
    b.annotate(f"10 identical values\n({with_eval[0]:.1f} %)", (0.15, with_eval[0]), xytext=(0, -28),
               textcoords="offset points", color=INK, fontsize=8)
    b.set_xticks([0, 1], ["with model.eval()", "without"])
    b.set_xlim(-0.6, 1.6)
    a.set(xlabel="epoch", ylabel="val accuracy", title="Symptom: lower validation, changing with the batch")
    b.set(ylabel="val accuracy (%)", title="Evidence: same weights, 10 evaluations")
    a.legend(frameon=False)
    save(fig, out / "6_forget_eval.png")
    lo, hi = min(without), max(without)
    return {"failure": "model.eval() forgotten", "symptom": "noisy, lower validation scores",
            "evidence": f"same checkpoint evaluated 10 times: {with_eval[0]:.1f} % every time with eval(), "
                        f"{lo:.1f}-{hi:.1f} % without",
            "fix": "model.eval() in evaluate(), model.train() back in train_one_epoch()"}


def shuffled_labels(r: Runs, out: Path) -> dict:
    fig, (a, b) = two_panels("7. Training labels shuffled at random: memorisation without understanding")
    plot(a, r, "shuffled_labels", "train_accuracy", BLUE, "train (random labels)", linestyle="--")
    plot(a, r, "shuffled_labels", "val_accuracy", BLUE, "val (true labels)")
    a.axhline(0.2, color=INK, linestyle=":", linewidth=1)
    a.text(1, 0.22, "chance (20 %)", color=INK, fontsize=8)
    plot(b, r, "reference", "train_loss", REF, "true labels")
    plot(b, r, "shuffled_labels", "train_loss", BLUE, "random labels")
    a.set(xlabel="epoch", ylabel="accuracy", title="Symptom")
    b.set(xlabel="epoch", ylabel="training loss", title="Evidence: the training loss reaches 0 with random labels too")
    a.legend(frameon=False)
    b.legend(frameon=False)
    save(fig, out / "7_shuffled_labels.png")
    _, train_acc = r.curve("shuffled_labels", "train_accuracy")
    _, val_acc = r.curve("shuffled_labels", "val_accuracy")
    return {"failure": "Shuffled labels",
            "symptom": f"train {train_acc[-1] * 100:.0f} %, val {val_acc[-1] * 100:.0f} %",
            "evidence": "a training loss near 0 proves memorisation, not learning: 6.3 M parameters store 1 056 "
                        "arbitrary labels (Zhang et al., 2017)",
            "fix": "a val score near chance with a high train score = no signal learned: check the labels"}


def ungrouped(r: Runs, out: Path) -> dict:
    stats = {}
    for name in ("reference", "ungrouped_split"):
        found = latest_per_name(r.df, [f"{name}-seed{s}" for s in (0, 1, 2)])
        stats[name] = mean_std(found["metrics.best_val_accuracy"] * 100)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    style(ax)
    labels = ["grouped by artist\n(our split)", "not grouped"]
    for x, (name, color) in enumerate([("reference", REF), ("ungrouped_split", BLUE)]):
        mean, std = stats[name]
        ax.bar(x, mean, yerr=std, color=color, width=0.55, error_kw={"ecolor": INK, "capsize": 3})
        ax.text(x, mean + std + 1, f"{mean:.1f} ± {std:.1f}", ha="center", color=INK, fontsize=9)
    ax.set_xticks([0, 1], labels)
    ax.set(ylabel="val accuracy (%), 3 seeds", title="8. Split not grouped by artist")
    ax.grid(axis="x", visible=False)
    save(fig, out / "8_ungrouped_split.png")
    diff = stats["ungrouped_split"][0] - stats["reference"][0]
    return {"failure": "Split not grouped by artist",
            "symptom": f"val {stats['ungrouped_split'][0]:.1f} % vs {stats['reference'][0]:.1f} % ({diff:+.1f} pts)",
            "evidence": "only 20 of 225 val images share a painter with train: 93 % of the paintings are "
                        "'Unknown Artist'",
            "fix": "keep the grouped split; the leakage cannot be fully measured on this subset"}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    args = p.parse_args()
    r = Runs(args.subset)
    out = Path("reports/figures") / args.subset / "failures"
    rows = [f(r, out) for f in (lr_too_high, lr_too_low, zero_init, deep_sigmoid, overfitting, forget_eval,
                                shuffled_labels, ungrouped)]
    lines = ["| # | Provoked failure | Symptom | Evidence (monitors) | Fix |", "|---|---|---|---|---|"]
    lines += [f"| {i} | {row['failure']} | {row['symptom']} | {row['evidence']} | {row['fix']} |"
              for i, row in enumerate(rows, start=1)]
    text = "\n".join(lines) + "\n\nOne seed (0) per failure, except failure 8 (3 seeds). Figures: " \
                              f"reports/figures/{args.subset}/failures/.\n"
    table = Path("reports/tables") / f"{args.subset}_failures.md"
    table.parent.mkdir(parents=True, exist_ok=True)
    table.write_text(f"# Provoked failures ({args.subset})\n\n{text}", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
