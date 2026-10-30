# Pre-trained ResNet18 (wikiart_genre5)

| Version | Change | Val accuracy (%) | Val macro-F1 (%) | Val loss | Train - val gap (pts) | Kept epoch | Parameters |
|---|---|---|---|---|---|---|---|
| reference | logistic regression, HSV histogram | 32.0 | | | | | |
| reference | best MLP | 45.0 | | | | | |
| reference | logistic regression on frozen ResNet embeddings | 83.3 | | | | | |
| probe | backbone frozen, only the head (512 -> 5) learns; LR from range test | 82.4 ± 0.9 | 82.4 ± 0.9 | 0.433 ± 0.003 | 6 ± 1 | 12 | 2,565 trainable of 11.2 M |
| partial | layer4 + head learn; backbone LR 1e-4, head 1e-3 | 88.1 ± 0.9 | 88.2 ± 1.0 | 0.361 ± 0.006 | 10 ± 3 | 5 | 8.40 M trainable of 11.2 M |
| full | every layer learns; backbone LR 1e-4, head 1e-3 | 85.9 ± 0.3 | 86.0 ± 0.2 | 0.413 ± 0.002 | 8 ± 2 | 3 | 11.18 M |

Mean ± std over seeds 0, 1, 2; validation set (222 images: 1 image = 0.45 pts).
