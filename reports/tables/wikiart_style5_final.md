# Final test results (wikiart_style5)

| Model | Test accuracy, seed 0 [95 % CI] | Mean ± std over 3 seeds | Test macro-F1 [95 % CI] | CPU latency, 1 image, 2 threads (median / p95) | Size |
|---|---|---|---|---|---|
| Majority class | 18.9 [14.0, 24.3] | deterministic | 6.4 [4.9, 7.8] | 0.0 / 0.1 ms | 0.00 MB |
| Logistic regression (pixels) | 37.8 [31.5, 44.1] | deterministic | 36.6 [29.8, 42.5] | 1.7 / 2.4 ms | 0.54 MB |
| Logistic regression (hsv) | 47.7 [41.4, 54.1] | deterministic | 46.9 [40.3, 53.0] | 6.5 / 10.5 ms | 0.01 MB |
| MLP (mlp_v6) | 41.9 [35.6, 48.6] | 40.7 ± 1.1 | 40.1 [33.4, 46.2] | 3.7 / 4.6 ms | 25.72 MB |
| ResNet18 embeddings + logistic regression | 81.1 [75.7, 86.0] | deterministic | 81.1 [75.5, 85.8] | 111.0 / 131.9 ms | 44.81 MB |
| ResNet18 linear probe | 80.6 [75.2, 85.6] | 79.4 ± 2.5 | 80.6 [75.3, 85.5] | 113.6 / 170.0 ms | 44.79 MB |
| ResNet18 fine-tuned (resnet_partial) | 82.4 [77.5, 86.9] | 84.1 ± 2.5 | 82.3 [77.1, 86.9] | 116.1 / 136.6 ms | 44.79 MB |

Paired bootstrap on the same test images (seed 0): accuracy difference, 95 % interval. If the interval contains 0, the test set cannot separate the two models.

| Comparison | Difference (points) | 95 % CI |
|---|---|---|
| MLP (mlp_v6) - Logistic regression (hsv) | -5.9 | [-13.1, +0.9] |
| ResNet18 fine-tuned (resnet_partial) - Logistic regression (hsv) | +34.7 | [+27.5, +41.9] |
| ResNet18 fine-tuned (resnet_partial) - MLP (mlp_v6) | +40.5 | [+33.3, +47.7] |
| ResNet18 fine-tuned (resnet_partial) - ResNet18 linear probe | +1.8 | [-2.7, +5.9] |
| ResNet18 fine-tuned (resnet_partial) - ResNet18 embeddings + logistic regression | +1.4 | [-3.2, +5.9] |
