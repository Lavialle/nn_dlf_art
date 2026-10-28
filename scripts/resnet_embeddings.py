"""Bonus: logistic regression on the embeddings of the frozen pre-trained ResNet18 (512 numbers per image).

    uv run python scripts/resnet_embeddings.py --subset wikiart_style5

The ResNet is used as a fixed feature extractor, like the HSV histogram of the baselines: no gradient, no epoch.
One MLflow run per C in <subset>-resnet; the test set is not touched.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from artguide.data.dataset import WikiArtSplit
from artguide.data.transforms import image_transform
from artguide.evaluation.metrics import scores
from artguide.models.baselines import logistic_regression
from artguide.models.resnet import embedding_model, imagenet_stats
from artguide.tracking import log_scores, start_run
from artguide.train.builders import get_device

C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]


@torch.no_grad()
def embeddings(split: WikiArtSplit, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    transform = image_transform(*imagenet_stats(), size=224)
    x = torch.stack([transform(image) for image in split.images])
    return torch.cat([model(batch.to(device)).cpu() for batch in x.split(256)])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subset", default="wikiart_style5")
    args = p.parse_args()
    data_dir, split_file = f"data/{args.subset}_224", Path("splits") / f"{args.subset}.csv"
    device = get_device()
    model = embedding_model().to(device)

    train, val = (WikiArtSplit(data_dir, split_file, split) for split in ("train", "val"))
    t0 = time.perf_counter()
    X_train, X_val = embeddings(train, model, device).numpy(), embeddings(val, model, device).numpy()
    print(f"embeddings {X_train.shape} + {X_val.shape} in {time.perf_counter() - t0:.1f} s")
    y_train, y_val = train.labels.numpy(), val.labels.numpy()

    print(f"\n{'C':>7s} {'val acc':>8s} {'val F1':>8s}")
    for C in C_GRID:
        params = {"model": "logreg", "features": "resnet18_embedding", "n_features": X_train.shape[1], "C": C}
        with start_run(f"{args.subset}-resnet", f"resnet_embed_logreg-C{C}", params, data_dir=data_dir):
            clf = logistic_regression(C).fit(X_train, y_train)
            log_scores(scores(y_train, clf.predict(X_train)), "train")
            val_scores = scores(y_val, clf.predict(X_val))
            log_scores(val_scores, "val")
        print(f"{C:7g} {val_scores['accuracy']:8.3f} {val_scores['macro_f1']:8.3f}")


if __name__ == "__main__":
    main()
