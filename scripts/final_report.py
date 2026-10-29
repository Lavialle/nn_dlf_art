"""Final table accuracy x latency x size, paired comparisons and figures, from the saved test-pass results.

    uv run python scripts/final_report.py --subset wikiart_style5

Reads reports/tables/<subset>_test.csv and _test_predictions.csv (scripts/evaluate.py); the test set is not
used again. Writes reports/tables/<subset>_final.md and figures in reports/figures/<subset>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from artguide.evaluation.metrics import confusion, paired_bootstrap
from artguide.reporting import BLUE, INK, INK_MUTED, ORANGE, style

COLORS = {"baseline": INK_MUTED, "mlp": BLUE, "resnet": ORANGE}


def summary(test: pd.DataFrame) -> pd.DataFrame:
    """One row per model: seed-0 scores with their interval, mean ± std over seeds, latency and size."""
    rows = []
    for model, group in test.groupby("model", sort=False):
        first = group.iloc[0]
        rows.append({**first.to_dict(), "seeds": len(group), "acc_mean": group["accuracy"].mean(),
                     "acc_std": group["accuracy"].std() if len(group) > 1 else np.nan})
    return pd.DataFrame(rows)


def find(table: pd.DataFrame, prefix: str) -> str:
    return next(m for m in table["model"] if m.startswith(prefix))


def markdown(table: pd.DataFrame, pairs: list[tuple[str, str, tuple[float, float, float]]]) -> str:
    lines = ["| Model | Test accuracy, seed 0 [95 % CI] | Mean ± std over 3 seeds | Test macro-F1 [95 % CI] "
             "| CPU latency, 1 image, 2 threads (median / p95) | Size |", "|---|---|---|---|---|---|"]
    for r in table.itertuples():
        spread = f"{r.acc_mean * 100:.1f} ± {r.acc_std * 100:.1f}" if r.seeds > 1 else "deterministic"
        lines.append(f"| {r.model} | {r.accuracy * 100:.1f} [{r.accuracy_lo * 100:.1f}, {r.accuracy_hi * 100:.1f}] "
                     f"| {spread} | {r.macro_f1 * 100:.1f} [{r.macro_f1_lo * 100:.1f}, {r.macro_f1_hi * 100:.1f}] "
                     f"| {r.median_ms:.1f} / {r.p95_ms:.1f} ms | {r.size_mb:.2f} MB |")
    lines += ["", "Paired bootstrap on the same test images (seed 0): accuracy difference, 95 % interval. "
              "If the interval contains 0, the test set cannot separate the two models.", "",
              "| Comparison | Difference (points) | 95 % CI |", "|---|---|---|"]
    lines += [f"| {a} - {b} | {d * 100:+.1f} | [{lo * 100:+.1f}, {hi * 100:+.1f}] |" for a, b, (d, lo, hi) in pairs]
    return "\n".join(lines) + "\n"


def tradeoff_figure(table: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    style(ax)
    for r in table.itertuples():
        color = COLORS[r.family]
        ax.errorbar(r.median_ms, r.accuracy * 100, yerr=[[(r.accuracy - r.accuracy_lo) * 100],
                                                         [(r.accuracy_hi - r.accuracy) * 100]],
                    fmt="o", color=color, markersize=8, ecolor=color, elinewidth=1.2, capsize=3,
                    markeredgecolor="white", markeredgewidth=1.5)
        ax.annotate(f"{r.model}\n{r.size_mb:.1f} MB", (r.median_ms, r.accuracy * 100), xytext=(8, -4),
                    textcoords="offset points", fontsize=8, color=INK)
    for family, color in COLORS.items():
        ax.scatter([], [], color=color, label=family)
    ax.set(xscale="log", xlabel="CPU latency per image, 2 threads (ms, log scale)",
           ylabel="test accuracy (%), seed 0, 95 % CI", ylim=(0, 100))
    ax.set_title("Kiosk trade-off: accuracy vs latency", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_figure(preds: pd.DataFrame, classes: list[str], columns: list[str], titles: list[str],
                     path: Path) -> None:
    fig, axes = plt.subplots(1, len(columns), figsize=(4.6 * len(columns), 4.4))
    for ax, column, title in zip(axes, columns, titles):
        m = confusion(preds["label"], preds[column], len(classes))
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.sum(axis=1).max())
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, m[i, j], ha="center", va="center", fontsize=9,
                        color="white" if m[i, j] > 0.6 * m.sum(axis=1).max() else INK)
        ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)), classes if ax is axes[0] else [], fontsize=8)
        acc = (preds["label"] == preds[column]).mean() * 100
        ax.set_title(f"{title}\n{acc:.1f} %", fontsize=10, color=INK)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("true style")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    args = p.parse_args()
    tables, figures = Path("reports/tables"), Path("reports/figures") / args.subset
    test = pd.read_csv(tables / f"{args.subset}_test.csv")
    preds = pd.read_csv(tables / f"{args.subset}_test_predictions.csv")
    table = summary(test)

    hsv, mlp = find(table, "Logistic regression (hsv)"), find(table, "MLP")
    probe, tuned = find(table, "ResNet18 linear probe"), find(table, "ResNet18 fine-tuned")
    embed = find(table, "ResNet18 embeddings")
    col = lambda model: next(c for c in preds.columns if c.startswith(f"{model} | seed"))  # noqa: E731
    pairs = [(a, b, paired_bootstrap(preds["label"], preds[col(a)], preds[col(b)]))
             for a, b in [(mlp, hsv), (tuned, hsv), (tuned, mlp), (tuned, probe), (tuned, embed)]]

    text = markdown(table, pairs)
    (tables / f"{args.subset}_final.md").write_text(f"# Final test results ({args.subset})\n\n{text}",
                                                     encoding="utf-8")
    figures.mkdir(parents=True, exist_ok=True)
    tradeoff_figure(table, figures / "final_tradeoff.png")
    split = pd.read_csv(Path("splits") / f"{args.subset}.csv")
    classes = split.drop_duplicates("label_idx").sort_values("label_idx")["label"].tolist()
    confusion_figure(preds, classes, [col(hsv), col(mlp), col(tuned)], ["HSV + logistic regression", mlp, tuned],
                     figures / "final_confusion.png")
    print(text)


if __name__ == "__main__":
    main()
