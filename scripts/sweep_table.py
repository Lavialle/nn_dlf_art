"""Sensitivity of the standard hyperparameters, read from MLflow: one table, validation set only.

    uv run python scripts/sweep_table.py --subset wikiart_genre5

Each sweep changes ONE hyperparameter of a reference config (MLP v5, MLP v6 or the CNN); the batch-size sweep
re-measures the LR for every size (the two are coupled). Runs come from the sweep script (run names below).
Writes reports/tables/<subset>_sweeps.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from artguide.reporting import latest_per_name, mean_std, runs

SEEDS = (0, 1, 2)
SWEEPS = [  # hyperparameter, model, group (MLflow experiment), [(run name, value shown)]; reference marked *
    ("batch size (LR re-measured)", "MLP v5", "mlp",
     [("mlp_v5-bs16", "16"), ("mlp_v5-bs32", "32"), ("mlp_v5", "64 *"), ("mlp_v5-bs128", "128")]),
    ("batch size (LR re-measured)", "CNN", "cnn",
     [("cnn-bs16", "16"), ("cnn-bs32", "32"), ("cnn", "64 *"), ("cnn-bs128", "128")]),
    ("epochs of the cosine schedule", "MLP v6", "mlp",
     [("mlp_v6", "40 *"), ("mlp_v6-ep60", "60"), ("mlp_v6-ep80", "80")]),
    ("crop scale (smallest area kept)", "MLP v5", "mlp",
     [("mlp_v5-crop0.4", "0.4"), ("mlp_v5", "0.6 *"), ("mlp_v5-crop0.8", "0.8"), ("mlp_v5-crop1.0", "1.0 (flip only)")]),
    ("crop scale (smallest area kept)", "CNN", "cnn",
     [("cnn-crop0.4", "0.4"), ("cnn", "0.6 *"), ("cnn-crop0.8", "0.8"), ("cnn-crop1.0", "1.0 (flip only)")]),
    ("input size (LR re-measured)", "CNN", "cnn", [("cnn", "64 px *"), ("cnn-128px", "128 px")]),
]


def sweep_rows(subset: str) -> list[dict]:
    out, cache = [], {}
    for hyperparameter, model, group, variants in SWEEPS:
        df = cache.setdefault(group, runs(f"{subset}-{group}"))
        for name, value in variants:
            found = latest_per_name(df, [f"{name}-seed{s}" for s in SEEDS]) if not df.empty else df
            if len(found) != len(SEEDS):
                continue
            row = {"hyperparameter": hyperparameter, "model": model, "value": value,
                   "lr": float(found.iloc[0]["params.lr"])}
            for col, metric, scale in [("acc", "best_val_accuracy", 100), ("f1", "best_val_macro_f1", 100),
                                       ("loss", "best_val_loss", 1), ("epoch", "best_val_epoch", 1)]:
                row[col], row[f"{col}_std"] = (v * scale for v in mean_std(found[f"metrics.{metric}"]))
            out.append(row)
    return out


def markdown(rows: list[dict]) -> str:
    lines = ["| Hyperparameter | Model | Value | LR | Val accuracy (%) | Val macro-F1 (%) | Val loss | Kept epoch |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['hyperparameter']} | {r['model']} | {r['value']} | {r['lr']:.1e} "
                     f"| {r['acc']:.1f} ± {r['acc_std']:.1f} | {r['f1']:.1f} ± {r['f1_std']:.1f} "
                     f"| {r['loss']:.3f} ± {r['loss_std']:.3f} | {r['epoch']:.0f} |")
    return ("\n".join(lines) + "\n\n* = value of the reference config. Mean ± std over seeds 0, 1, 2; validation set "
            "(1 image = 0.45 pts): differences below ~3 points are within noise.\n")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_genre5")
    args = p.parse_args()
    text = markdown(sweep_rows(args.subset))
    out = Path("reports/tables") / f"{args.subset}_sweeps.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"# Hyperparameter sweeps ({args.subset})\n\n{text}", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
