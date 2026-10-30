# MLP ablation (wikiart_genre5)

| Version | Change | Val accuracy (%) | Val macro-F1 (%) | Val loss | Train - val gap (pts) | Kept epoch | Parameters |
|---|---|---|---|---|---|---|---|
| reference | majority class | 19.4 | | | | | |
| reference | logistic regression, pixels | 39.6 | | | | | |
| reference | logistic regression, HSV histogram | 32.0 | | | | | |
| v0 | naive: 1 hidden layer (512), SGD, guessed LR 0.01, 30 epochs | 43.8 ± 5.0 | 42.9 ± 4.4 | 1.280 ± 0.008 | 26 ± 11 | 11 | 6.29 M |
| v1 | early stopping + LR from the range test: SGD | 44.6 ± 1.8 | 43.7 ± 2.1 | 1.276 ± 0.011 | 25 ± 5 | 22 | 6.29 M |
| v1 | early stopping + LR from the range test: SGD + Nesterov momentum | 44.0 ± 1.4 | 42.9 ± 2.2 | 1.283 ± 0.018 | 25 ± 4 | 6 | 6.29 M |
| v1 | early stopping + LR from the range test: AdamW | 44.3 ± 3.0 | 43.3 ± 3.0 | 1.268 ± 0.002 | 21 ± 5 | 8 | 6.29 M |
| v2 | v1 Nesterov + BatchNorm + 2nd hidden layer (256), LR re-measured | 38.7 ± 4.0 | 35.4 ± 6.0 | 1.329 ± 0.041 | 11 ± 10 | 1 | 6.43 M |
| v3 | + dropout 0.5 | 41.1 ± 3.6 | 40.6 ± 3.9 | 1.294 ± 0.010 | 24 ± 2 | 6 | 6.43 M |
| v4 | + weight decay 5e-4 | 42.0 ± 2.9 | 40.9 ± 2.5 | 1.293 ± 0.015 | 22 ± 4 | 6 | 6.43 M |
| v5 | + augmentation (random crop 60-100 %, horizontal flip) | 45.0 ± 1.4 | 43.7 ± 2.6 | 1.250 ± 0.006 | 5 ± 2 | 16 | 6.43 M |
| v6 | + cosine LR schedule + label smoothing 0.1 | 42.9 ± 1.4 | 41.9 ± 1.8 | 1.265 ± 0.006 | 12 ± 0 | 34 | 6.43 M |

Mean ± std over seeds 0, 1, 2; validation set (222 images: 1 image = 0.45 pts).
