# Provoked failures (wikiart_style5)

| # | Provoked failure | Symptom | Evidence (monitors) | Fix |
|---|---|---|---|---|
| 1 | LR x 100 | loss explodes to NaN | 1st-layer gradient norm 1.03e+13 at epoch 1 vs 0.581 | lower the LR (range test); clipping alone: val acc 30 % |
| 2 | LR / 1000 | training loss 1.610 -> 1.562 in 40 epochs | update ratio 8.3e-08 vs 1.1e-04 | LR from the range test |
| 3 | Zero initialisation | val acc stuck at 19 %, macro-F1 6 % (one class predicted) | activation std 0.0e+00, 1st-layer gradient 0.0e+00 | random init (Kaiming for ReLU) breaks the symmetry |
| 4 | 8 sigmoid layers, no BN | training loss flat at ln 5 | gradient norm 0.98 at the output, 2e-06 at layer 1 (x4e+05) | ReLU + Kaiming + BatchNorm: val acc 41 % |
| 5 | No regularisation, 200 epochs | val loss 1.39 at epoch 4 -> 2.89 at epoch 200 | train 100 % vs val 37 % at the end | early stopping (keep the best epoch) + regularisation (v2-v6) |
| 6 | model.eval() forgotten | noisy, lower validation scores | same checkpoint evaluated 10 times: 41.0 % every time with eval(), 36.9-41.0 % without | model.eval() in evaluate(), model.train() back in train_one_epoch() |
| 7 | Shuffled labels | train 100 %, val 21 % | a training loss near 0 proves memorisation, not learning: 6.3 M parameters store 1 056 arbitrary labels (Zhang et al., 2017) | a val score near chance with a high train score = no signal learned: check the labels |
| 8 | Split not grouped by artist | val 41.2 % vs 39.5 % (+1.7 pts) | only 20 of 225 val images share a painter with train: 93 % of the paintings are 'Unknown Artist' | keep the grouped split; the leakage cannot be fully measured on this subset |

One seed (0) per failure, except failure 8 (3 seeds). Figures: reports/figures/wikiart_style5/failures/.
