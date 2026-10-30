# Final test results (wikiart_genre5)

| Model | Test accuracy, seed 0 [95 % CI] | Mean ± std over 3 seeds | Test macro-F1 [95 % CI] | CPU latency, 1 image, 2 threads (median / p95) | Size |
|---|---|---|---|---|---|
| Majority class | 18.9 [14.1, 24.2] | deterministic | 6.4 [4.9, 7.8] | 0.0 / 0.0 ms | 0.00 MB |
| Logistic regression (pixels) | 39.6 [33.5, 46.3] | deterministic | 38.9 [32.8, 45.0] | 0.4 / 0.5 ms | 0.54 MB |
| Logistic regression (hsv) | 39.6 [33.5, 45.8] | deterministic | 39.8 [33.5, 46.0] | 2.2 / 3.0 ms | 0.01 MB |
| MLP (mlp_v5-bs32) | 46.3 [40.1, 52.9] | 45.2 ± 2.2 | 46.0 [39.6, 52.4] | 1.1 / 1.7 ms | 25.72 MB |
| CNN from scratch (cnn-bs16) | 62.6 [56.4, 68.7] | 60.6 ± 2.3 | 62.3 [55.8, 68.3] | 1.8 / 2.5 ms | 1.57 MB |
| ResNet18 embeddings + logistic regression | 78.9 [73.6, 84.1] | deterministic | 78.5 [72.8, 83.8] | 29.5 / 44.7 ms | 44.81 MB |
| ResNet18 linear probe | 78.0 [72.7, 83.3] | 77.5 ± 0.8 | 77.2 [71.3, 82.5] | 29.1 / 43.0 ms | 44.79 MB |
| ResNet18 fine-tuned (resnet_partial) | 80.2 [74.9, 85.5] | 79.6 ± 2.3 | 79.8 [74.1, 84.8] | 27.6 / 42.4 ms | 44.79 MB |

Paired bootstrap on the same test images (seed 0): accuracy difference, 95 % interval. If the interval contains 0, the test set cannot separate the two models.

| Comparison | Difference (points) | 95 % CI |
|---|---|---|
| MLP (mlp_v5-bs32) - Logistic regression (hsv) | +6.6 | [-1.8, +15.0] |
| ResNet18 fine-tuned (resnet_partial) - Logistic regression (hsv) | +40.5 | [+32.2, +48.5] |
| ResNet18 fine-tuned (resnet_partial) - MLP (mlp_v5-bs32) | +33.9 | [+26.0, +41.9] |
| ResNet18 fine-tuned (resnet_partial) - ResNet18 linear probe | +2.2 | [-1.8, +5.7] |
| ResNet18 fine-tuned (resnet_partial) - ResNet18 embeddings + logistic regression | +1.3 | [-2.6, +5.3] |
| CNN from scratch (cnn-bs16) - MLP (mlp_v5-bs32) | +16.3 | [+8.8, +23.3] |
| ResNet18 fine-tuned (resnet_partial) - CNN from scratch (cnn-bs16) | +17.6 | [+10.1, +24.7] |
