"""Ablation tables of the MLP versions and of the ResNet fine-tuning levels, read from MLflow: mean ± std over
seeds, validation set only.

    uv run python scripts/ablation_table.py --subset wikiart_style5

Every number is the model kept by early stopping (lowest val loss), not the best epoch picked afterwards.
Writes reports/tables/<subset>_ablation.md / .csv, <subset>_resnet.md / .csv and the matching figures in
reports/figures/<subset>/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from artguide.reporting import BLUE, INK, INK_MUTED, ORANGE, history, latest_per_name, mean_std, runs, style

SEEDS = (0, 1, 2)
VERSIONS = [  # config name, version, short label for the figure, what changed (one change per version)
    ("mlp_v0", "v0", "v0 naive", "naive: 1 hidden layer (512), SGD, guessed LR 0.01, 30 epochs"),
    ("mlp_v1_sgd", "v1", "v1 SGD", "early stopping + LR from the range test: SGD"),
    ("mlp_v1_nesterov", "v1", "v1 Nesterov", "early stopping + LR from the range test: SGD + Nesterov momentum"),
    ("mlp_v1_adamw", "v1", "v1 AdamW", "early stopping + LR from the range test: AdamW"),
    ("mlp_v2", "v2", "v2 + BatchNorm, 2 layers", "v1 Nesterov + BatchNorm + 2nd hidden layer (256), LR re-measured"),
    ("mlp_v3", "v3", "v3 + dropout", "+ dropout 0.5"),
    ("mlp_v4", "v4", "v4 + weight decay", "+ weight decay 5e-4"),
    ("mlp_v5", "v5", "v5 + augmentation", "+ augmentation (random crop 60-100 %, horizontal flip)"),
    ("mlp_v6", "v6", "v6 + cosine, smoothing", "+ cosine LR schedule + label smoothing 0.1"),
]
RESNET_VERSIONS = [  # same pipeline (224 px, ImageNet statistics, crop + flip, AdamW); only what learns changes
    ("resnet_probe", "probe", "linear probe", "backbone frozen, only the head (512 -> 5) learns; LR from range test"),
    ("resnet_partial", "partial", "partial (layer4 + head)", "layer4 + head learn; backbone LR 1e-4, head 1e-3"),
    ("resnet_full", "full", "full fine-tuning", "every layer learns; backbone LR 1e-4, head 1e-3"),
]
BASELINE_LABELS = {"majority class": "majority", "logistic regression, pixels": "pixels",
                   "logistic regression, HSV histogram": "HSV", "best MLP": "MLP",
                   "logistic regression on frozen ResNet embeddings": "embeddings"}


def baselines(subset: str) -> list[tuple[str, float]]:
    """Val accuracy of the classical baselines; for each logistic regression, the best C on val."""
    df = runs(f"{subset}-baselines")
    if df.empty:
        return []
    out = []
    for label, mask in [("majority class", df["params.model"] == "majority"),
                        ("logistic regression, pixels", df["params.features"] == "pixels"),
                        ("logistic regression, HSV histogram", df["params.features"] == "hsv")]:
        if mask.any():
            out.append((label, df.loc[mask, "metrics.val_accuracy"].max()))
    return out


def version_row(df: pd.DataFrame, name: str, version: str, label: str, change: str) -> dict | None:
    found = latest_per_name(df, [f"{name}-seed{s}" for s in SEEDS])
    if found.empty:
        return None
    gaps = []  # train accuracy - val accuracy, both at the kept epoch
    for _, run in found.iterrows():
        train_acc = history(run.run_id, "train_accuracy")[int(run["metrics.best_val_epoch"])]
        gaps.append(train_acc - run["metrics.best_val_accuracy"])
    first = found.iloc[0]
    trainable = first.get("params.n_trainable")  # not logged by the older runs: then everything was trainable
    row = {"version": version, "config": name, "label": label, "change": change, "seeds": len(found),
           "params": int(first["params.n_parameters"]),
           "trainable": int(trainable if pd.notna(trainable) else first["params.n_parameters"])}
    for col, metric, scale in [("val_acc", "best_val_accuracy", 100), ("val_f1", "best_val_macro_f1", 100),
                               ("val_loss", "best_val_loss", 1), ("best_epoch", "best_val_epoch", 1)]:
        row[col], row[f"{col}_std"] = (v * scale for v in mean_std(found[f"metrics.{metric}"]))
    row["gap"], row["gap_std"] = (v * 100 for v in mean_std(gaps))
    return row


def parameters(r) -> str:
    if r.trainable == r.params:
        return f"{r.params / 1e6:.2f} M"
    trainable = f"{r.trainable / 1e6:.2f} M" if r.trainable >= 1e5 else f"{r.trainable:,}"
    return f"{trainable} trainable of {r.params / 1e6:.1f} M"


def markdown(table: pd.DataFrame, base: list[tuple[str, float]]) -> str:
    lines = ["| Version | Change | Val accuracy (%) | Val macro-F1 (%) | Val loss | Train - val gap (pts) "
             "| Kept epoch | Parameters |", "|---|---|---|---|---|---|---|---|"]
    for label, acc in base:
        lines.append(f"| reference | {label} | {acc * 100:.1f} | | | | | |")
    for r in table.itertuples():
        lines.append(f"| {r.version} | {r.change} | {r.val_acc:.1f} ± {r.val_acc_std:.1f} "
                     f"| {r.val_f1:.1f} ± {r.val_f1_std:.1f} | {r.val_loss:.3f} ± {r.val_loss_std:.3f} "
                     f"| {r.gap:.0f} ± {r.gap_std:.0f} | {r.best_epoch:.0f} | {parameters(r)} |")
    return "\n".join(lines) + "\n\nMean ± std over seeds 0, 1, 2; validation set (222 images: 1 image = 0.45 pts).\n"


def figure(table: pd.DataFrame, base: list[tuple[str, float]], path: Path, title: str, color: str = BLUE) -> None:
    y = range(len(table))[::-1]  # v0 at the top
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(table) + 1.4))
    ax.barh(list(y), table["val_acc"], xerr=table["val_acc_std"], height=0.6, color=color,
            error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    for yi, r in zip(y, table.itertuples()):
        ax.text(r.val_acc + r.val_acc_std + 0.6, yi, f"{r.val_acc:.1f}", va="center", color=INK, fontsize=9)
    for label, acc in base:
        ax.axvline(acc * 100, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.text(acc * 100, 1.01, f"{BASELINE_LABELS.get(label, label)}\n{acc * 100:.0f} %", ha="center",
                va="bottom", color=INK_MUTED, fontsize=8, transform=ax.get_xaxis_transform())
    ax.set_yticks(list(y), table["label"])
    ax.set_xlim(0, max(60, table["val_acc"].max() + 10))
    ax.set_xlabel("validation accuracy (%), mean ± std over 3 seeds")
    ax.set_title(title, loc="left", color=INK, pad=28)
    style(ax)
    ax.grid(axis="y", visible=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    args = p.parse_args()

    tables, figures = Path("reports/tables"), Path("reports/figures") / args.subset
    tables.mkdir(parents=True, exist_ok=True)
    base = baselines(args.subset)

    mlp_runs = runs(f"{args.subset}-mlp")
    mlp = pd.DataFrame([r for v in VERSIONS if (r := version_row(mlp_runs, *v)) is not None])
    mlp.round(4).to_csv(tables / f"{args.subset}_ablation.csv", index=False)
    text = markdown(mlp, base)
    (tables / f"{args.subset}_ablation.md").write_text(f"# MLP ablation ({args.subset})\n\n{text}", encoding="utf-8")
    figure(mlp, base, figures / "ablation.png", "MLP ablation: one change per version (dashed: classical baselines)")
    print(text)

    resnet_runs = runs(f"{args.subset}-resnet")
    resnet = pd.DataFrame([r for v in RESNET_VERSIONS if (r := version_row(resnet_runs, *v)) is not None])
    if resnet.empty:
        return
    best_mlp = mlp.loc[mlp["val_loss"].idxmin()]  # the selection rule of the final evaluation
    references = [base[-1], ("best MLP", best_mlp["val_acc"] / 100)]
    embed = resnet_runs[resnet_runs["tags.mlflow.runName"].str.startswith("resnet_embed_logreg-C")]
    if not embed.empty:
        references.append(("logistic regression on frozen ResNet embeddings", embed["metrics.val_accuracy"].max()))
    resnet.round(4).to_csv(tables / f"{args.subset}_resnet.csv", index=False)
    text = markdown(resnet, references)
    (tables / f"{args.subset}_resnet.md").write_text(f"# Pre-trained ResNet18 ({args.subset})\n\n{text}",
                                                       encoding="utf-8")
    figure(resnet, references, figures / "resnet.png",
           "Pre-trained ResNet18: what learns (dashed: HSV baseline, best MLP, embeddings)", color=ORANGE)
    print(text)


if __name__ == "__main__":
    main()
