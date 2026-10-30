# Hyperparameter sweeps (wikiart_genre5)

| Hyperparameter | Model | Value | LR | Val accuracy (%) | Val macro-F1 (%) | Val loss | Kept epoch |
|---|---|---|---|---|---|---|---|
| batch size (LR re-measured) | MLP v5 | 16 | 1.4e-03 | 43.2 ± 0.5 | 42.2 ± 0.6 | 1.265 ± 0.008 | 21 |
| batch size (LR re-measured) | MLP v5 | 32 | 2.9e-03 | 44.6 ± 0.5 | 43.1 ± 0.4 | 1.242 ± 0.009 | 38 |
| batch size (LR re-measured) | MLP v5 | 64 * | 9.9e-03 | 45.0 ± 1.4 | 43.7 ± 2.6 | 1.250 ± 0.006 | 16 |
| batch size (LR re-measured) | MLP v5 | 128 | 6.1e-03 | 43.5 ± 1.7 | 43.0 ± 1.8 | 1.258 ± 0.016 | 25 |
| batch size (LR re-measured) | CNN | 16 | 8.0e-04 | 57.4 ± 1.1 | 56.8 ± 1.1 | 1.047 ± 0.046 | 43 |
| batch size (LR re-measured) | CNN | 32 | 2.7e-03 | 57.8 ± 5.0 | 57.5 ± 5.0 | 1.061 ± 0.068 | 29 |
| batch size (LR re-measured) | CNN | 64 * | 2.9e-03 | 55.3 ± 4.1 | 55.0 ± 4.7 | 1.095 ± 0.026 | 25 |
| batch size (LR re-measured) | CNN | 128 | 3.2e-03 | 55.9 ± 2.0 | 54.9 ± 1.9 | 1.121 ± 0.057 | 35 |
| epochs of the cosine schedule | MLP v6 | 40 * | 9.9e-03 | 42.9 ± 1.4 | 41.9 ± 1.8 | 1.265 ± 0.006 | 34 |
| epochs of the cosine schedule | MLP v6 | 60 | 9.9e-03 | 47.9 ± 1.4 | 47.0 ± 1.8 | 1.249 ± 0.005 | 42 |
| epochs of the cosine schedule | MLP v6 | 80 | 9.9e-03 | 44.1 ± 0.5 | 43.2 ± 0.2 | 1.247 ± 0.005 | 55 |
| crop scale (smallest area kept) | MLP v5 | 0.4 | 9.9e-03 | 46.4 ± 2.4 | 45.2 ± 2.4 | 1.262 ± 0.008 | 18 |
| crop scale (smallest area kept) | MLP v5 | 0.6 * | 9.9e-03 | 45.0 ± 1.4 | 43.7 ± 2.6 | 1.250 ± 0.006 | 16 |
| crop scale (smallest area kept) | MLP v5 | 0.8 | 9.9e-03 | 44.4 ± 1.8 | 44.1 ± 1.6 | 1.253 ± 0.008 | 17 |
| crop scale (smallest area kept) | MLP v5 | 1.0 (flip only) | 9.9e-03 | 44.0 ± 2.1 | 43.1 ± 2.6 | 1.273 ± 0.012 | 8 |
| crop scale (smallest area kept) | CNN | 0.4 | 2.9e-03 | 55.7 ± 1.9 | 55.4 ± 1.9 | 1.118 ± 0.024 | 21 |
| crop scale (smallest area kept) | CNN | 0.6 * | 2.9e-03 | 55.3 ± 4.1 | 55.0 ± 4.7 | 1.095 ± 0.026 | 25 |
| crop scale (smallest area kept) | CNN | 0.8 | 2.9e-03 | 56.0 ± 0.9 | 55.7 ± 1.0 | 1.077 ± 0.010 | 31 |
| crop scale (smallest area kept) | CNN | 1.0 (flip only) | 2.9e-03 | 55.4 ± 1.6 | 54.8 ± 2.0 | 1.100 ± 0.040 | 22 |
| input size (LR re-measured) | CNN | 64 px * | 2.9e-03 | 55.3 ± 4.1 | 55.0 ± 4.7 | 1.095 ± 0.026 | 25 |
| input size (LR re-measured) | CNN | 128 px | 3.2e-03 | 60.1 ± 3.4 | 60.0 ± 3.4 | 1.083 ± 0.017 | 25 |

* = value of the reference config. Mean ± std over seeds 0, 1, 2; validation set (1 image = 0.45 pts): differences below ~3 points are within noise.
