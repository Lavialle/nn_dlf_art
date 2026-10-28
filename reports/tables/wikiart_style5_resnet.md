# Pre-trained ResNet18 (wikiart_style5)

| Version | Change | Val accuracy (%) | Val macro-F1 (%) | Val loss | Train - val gap (pts) | Kept epoch | Parameters |
|---|---|---|---|---|---|---|---|
| reference | logistic regression, HSV histogram | 50.9 | | | | | |
| reference | best MLP | 41.7 | | | | | |
| reference | logistic regression on frozen ResNet embeddings | 81.1 | | | | | |
| probe | backbone frozen, only the head (512 -> 5) learns; LR from range test | 80.6 ± 1.2 | 80.5 ± 1.1 | 0.506 ± 0.009 | 8 ± 1 | 6 | 2,565 trainable of 11.2 M |
| partial | layer4 + head learn; backbone LR 1e-4, head 1e-3 | 84.5 ± 1.4 | 84.2 ± 1.5 | 0.403 ± 0.014 | 12 ± 3 | 4 | 8.40 M trainable of 11.2 M |
| full | every layer learns; backbone LR 1e-4, head 1e-3 | 85.1 ± 0.5 | 84.8 ± 0.3 | 0.425 ± 0.029 | 10 ± 2 | 3 | 11.18 M |

Mean ± std over seeds 0, 1, 2; validation set (222 images: 1 image = 0.45 pts).
