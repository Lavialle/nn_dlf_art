"""THE single pass on the test set, at the very end: test scores with 95 % bootstrap intervals, CPU latency
(kiosk: 2 threads, one image) and size of every final candidate. It refuses to run twice (--force to override):
looking at the test set again to change a choice would turn it into a second validation set.

    uv run python scripts/evaluate.py --subset wikiart_style5

Writes reports/tables/<subset>_test.csv (one row per candidate and seed) and <subset>_test_predictions.csv;
logs one run per candidate in the MLflow experiment <subset>-test. Then: scripts/final_report.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from artguide.data.dataset import WikiArtSplit
from artguide.evaluation.candidates import build_candidates
from artguide.evaluation.latency import KIOSK_THREADS, measure_latency
from artguide.evaluation.metrics import bootstrap_ci
from artguide.tracking import log_scores, start_run


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    p.add_argument("--force", action="store_true", help="run again although the test set was already used")
    args = p.parse_args()
    tables = Path("reports/tables")
    out, out_preds = tables / f"{args.subset}_test.csv", tables / f"{args.subset}_test_predictions.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists: the test set was already used once. Read the results instead.")

    candidates = build_candidates(args.subset)  # every choice made on validation scores
    split_file = Path("splits") / f"{args.subset}.csv"
    tests = {d: WikiArtSplit(d, split_file, "test") for d in sorted({c.data_dir for c in candidates})}
    first = next(iter(tests.values()))
    predictions = pd.DataFrame({"id": first.ids, "label": first.labels.numpy()})
    rows = []
    for c in candidates:
        test = tests[c.data_dir]
        assert test.ids == first.ids  # both image sizes hold the same paintings, in the same order
        y, pred = test.labels.numpy(), c.predict(test.images)
        acc, acc_lo, acc_hi = bootstrap_ci(y, pred, "accuracy")
        f1, f1_lo, f1_hi = bootstrap_ci(y, pred, "macro_f1")
        row = {"model": c.name, "family": c.family, "seed": c.seed, "accuracy": acc, "accuracy_lo": acc_lo,
               "accuracy_hi": acc_hi, "macro_f1": f1, "macro_f1_lo": f1_lo, "macro_f1_hi": f1_hi,
               "size_mb": c.size_mb()}
        if c.seed in (None, 0):  # latency depends on the architecture, not on the seed
            row.update(measure_latency(c.predict, test.images[:1]))
        rows.append(row)
        predictions[f"{c.name} | seed {c.seed}"] = pred

        params = {"model": c.name, "family": c.family, "seed": c.seed, "data_dir": c.data_dir,
                  "latency_threads": KIOSK_THREADS}
        with start_run(f"{args.subset}-test", f"{c.name}-seed{c.seed}", params, data_dir=c.data_dir):
            log_scores({k: v for k, v in row.items() if isinstance(v, float)}, "test")
        print(f"{c.name:45s} seed {c.seed!s:4s} acc {acc:.3f} [{acc_lo:.3f}, {acc_hi:.3f}]  F1 {f1:.3f}"
              f"  {row.get('median_ms', float('nan')):7.2f} ms  {row['size_mb']:6.1f} MB")

    tables.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    predictions.to_csv(out_preds, index=False)
    print(f"\n-> {out}, {out_preds}. Next: uv run python scripts/final_report.py --subset {args.subset}")


if __name__ == "__main__":
    main()
