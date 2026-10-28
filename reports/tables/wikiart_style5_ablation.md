# MLP ablation (wikiart_style5)

| Version | Change | Val accuracy (%) | Val macro-F1 (%) | Val loss | Train - val gap (pts) | Kept epoch | Parameters |
|---|---|---|---|---|---|---|---|
| baseline | majority class | 18.9 | | | | | |
| baseline | logistic regression, pixels | 33.3 | | | | | |
| baseline | logistic regression, HSV histogram | 50.9 | | | | | |
| v0 | naive: 1 hidden layer (512), SGD, guessed LR 0.01, 30 epochs | 40.7 ± 1.4 | 39.3 ± 1.4 | 1.407 ± 0.012 | 26 ± 2 | 11 | 6.29 M |
| v1 | early stopping + LR from the range test: SGD | 39.6 ± 0.9 | 38.3 ± 1.0 | 1.407 ± 0.014 | 24 ± 8 | 16 | 6.29 M |
| v1 | early stopping + LR from the range test: SGD + Nesterov momentum | 41.0 ± 1.4 | 39.8 ± 1.5 | 1.419 ± 0.012 | 24 ± 2 | 5 | 6.29 M |
| v1 | early stopping + LR from the range test: AdamW | 39.0 ± 1.6 | 37.1 ± 0.7 | 1.429 ± 0.010 | 25 ± 3 | 6 | 6.29 M |
| v2 | v1 Nesterov + BatchNorm + 2nd hidden layer (256), LR re-measured | 42.8 ± 1.2 | 40.9 ± 1.3 | 1.388 ± 0.017 | 23 ± 2 | 2 | 6.43 M |
| v3 | + dropout 0.5 | 41.9 ± 0.8 | 39.6 ± 1.6 | 1.395 ± 0.004 | 18 ± 2 | 5 | 6.43 M |
| v4 | + weight decay 5e-4 | 41.0 ± 1.6 | 38.7 ± 1.3 | 1.396 ± 0.011 | 18 ± 4 | 5 | 6.43 M |
| v5 | + augmentation (random crop 60-100 %, horizontal flip) | 41.0 ± 3.2 | 38.8 ± 3.7 | 1.370 ± 0.026 | 10 ± 4 | 17 | 6.43 M |
| v6 | + cosine LR schedule + label smoothing 0.1 | 41.7 ± 1.6 | 38.9 ± 1.1 | 1.364 ± 0.001 | 11 ± 2 | 28 | 6.43 M |

Mean ± std over seeds 0, 1, 2; validation set (222 images: 1 image = 0.45 pts).
