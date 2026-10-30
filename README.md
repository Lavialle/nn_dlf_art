# nn_dlf_art

M2 Deep Learning Project: the neural-network components of a (fictional) museum guide that recognises the
**style** or the **genre** of a painting from a photo, and a recommendation to the museum director: is a neural
network worth it on a modest kiosk CPU, or does a classical model do the job?

Deliverables, all built from PyTorch primitives (hand-written training loop, no Lightning) and measured on
WikiArt subsets:

1. a **feed-forward network (MLP)** for artwork classification, improved step by step (ablation v0 → v6);
2. **training monitoring and failure diagnosis** (gradient norms, update ratios, dead ReLUs, 8 provoked
   failures, hyperparameter sweeps);
3. a **pre-trained CNN (ResNet18)**: linear probe, partial and full fine-tuning, and frozen embeddings + logistic
   regression; plus a small CNN trained from scratch, to separate the effect of convolutions from pre-training.

Results: [`reports/wikiart_style5_summary.md`](reports/wikiart_style5_summary.md) (styles) and
[`reports/wikiart_genre5_summary.md`](reports/wikiart_genre5_summary.md) (genres).

## Setup

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Clone and install the exact locked environment (Python + PyTorch, CUDA 13.0 build on Linux/Windows)
git clone https://github.com/Lavialle/nn_dlf_art.git
cd nn_dlf_art
uv sync

# 3. Check which device PyTorch will use (True = NVIDIA GPU, False = CPU)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# 4. Run the tests
uv run pytest -q
```

### Without an NVIDIA GPU

Nothing to change in the code: `get_device()` (`src/artguide/train/builders.py`) uses the GPU if PyTorch sees one,
the CPU otherwise.

- **Linux / Windows without NVIDIA GPU**: `uv sync` installs the same CUDA build of PyTorch (a few GB of CUDA
  libraries downloaded for nothing); it runs on the CPU.
- **macOS**: PyTorch comes from PyPI (the CUDA index has no macOS build), so the locked version may differ
  slightly from Linux/Windows. It runs on the CPU.

Time of one epoch on the genre subset (1 051 training images), measured on a laptop (8 CPU threads vs RTX 3060
Laptop). A run usually stops after 25-40 epochs (early stopping); every config is trained on 3 seeds.

| Model | GPU | CPU only |
|---|---|---|
| MLP, 64 px | 0.8 s | 1.3 s |
| Small CNN, 64 px | 0.8 s | 4.3 s |
| Small CNN, 128 px | 1.2 s | 17 s |
| ResNet18 linear probe, 224 px | 2.2 s | 24 s |
| ResNet18 partial fine-tuning, 224 px | 2.3 s | 31 s |

On a CPU, the MLP and the 64 px CNN stay quick (under a minute and a few minutes per run). The ResNet fine-tuning
runs take 10-20 minutes each; the frozen ResNet + logistic regression (`scripts/resnet_embeddings.py`) needs a
single forward pass over the images and remains cheap.

## Data

`make_wikiart_subset.py` (provided, not modified) streams only the needed shards of `huggan/wikiart` from the
Hugging Face Hub (pinned revision) and writes an ImageFolder tree + `metadata.csv` + `subset_info.json` under
`data/` (ignored by git: the images are not ours to publish). Each subset is built twice, with the same paintings:
128 px (MLP, CNN, baselines) and 224 px (ResNet). Always `--dry-run` first to see the plan.

```bash
# Styles: 5 styles x 300 paintings (9 shards, ~4.2 GB streamed per size)
for size in 128 224; do
  out=data/wikiart_style5; [ $size = 224 ] && out=data/wikiart_style5_224
  uv run python make_wikiart_subset.py --label style \
      --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art --per-class 300 --size $size --out $out
done

# Genres: 5 genres x 300 paintings, at most 30 per named painter (5 shards, ~2.3 GB streamed per size).
# With the default cap of 50, Claude Monet alone gave 50 of the 300 cityscapes, one indivisible block for the
# artist-grouped split, which broke the stratification (cityscape: 61 % in train instead of 70 %).
for size in 128 224; do
  out=data/wikiart_genre5; [ $size = 224 ] && out=data/wikiart_genre5_224
  uv run python make_wikiart_subset.py --label genre \
      --classes portrait landscape cityscape still_life religious_painting --per-class 300 \
      --max-per-artist 30 --size $size --out $out
done
```

The train / val / test split (70 / 15 / 15, stratified by class, grouped by artist) is frozen in
`splits/<subset>.csv` and committed; `tests/test_splits_no_leak.py` checks it. To rebuild it:
`uv run python scripts/make_splits.py --data-dir data/wikiart_genre5 --out splits/wikiart_genre5.csv`.

## Reproduce

Every run is logged in MLflow (`mlflow.db`, local, ignored by git) with its config, metrics per epoch, git commit
and subset description. The tables in `reports/tables/` are generated from MLflow by the scripts below. Commands
for the genres (replace `genre` / `wikiart_genre5` by `style` / `wikiart_style5` for the styles).

```bash
# Classical baselines: majority class, logistic regression on pixels, HSV histogram, painting orientation
uv run python scripts/run_baselines.py --data-dir data/wikiart_genre5

# Sanity checks (initial loss = ln K, memorise one batch) and LR range test of a config
uv run python scripts/sanity_checks.py --config configs/genre/mlp_v0.yaml
uv run python scripts/lr_range_test.py --config configs/genre/mlp_v2.yaml

# MLP ablation v0 -> v6, small CNN, ResNet18 (3 seeds each)
uv run python scripts/train.py --config configs/genre/mlp_v5.yaml --seeds 0 1 2   # any config in configs/genre/
uv run python scripts/train.py --config configs/genre/cnn.yaml --seeds 0 1 2
uv run python scripts/train.py --config configs/genre/resnet_partial.yaml --seeds 0 1 2
uv run python scripts/resnet_embeddings.py --subset wikiart_genre5

# One hyperparameter changed at a run (here the batch size, with its LR measured again)
uv run python scripts/lr_range_test.py --config configs/genre/cnn.yaml --set name=cnn-bs32 batch_size=32
uv run python scripts/train.py --config configs/genre/cnn.yaml --seeds 0 1 2 --set name=cnn-bs32 batch_size=32 lr=2.7e-3

# Provoked failures (styles only): configs/style/failures/, then
uv run python scripts/failure_figures.py --subset wikiart_style5

# Tables (validation only)
uv run python scripts/ablation_table.py --subset wikiart_genre5
uv run python scripts/sweep_table.py --subset wikiart_genre5

# THE single test-set pass (refuses to run twice), then the final table and figures
uv run python scripts/evaluate.py --subset wikiart_genre5
uv run python scripts/final_report.py --subset wikiart_genre5

# Browse every run
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000   # then http://127.0.0.1:5000
```

## Repository layout

```
make_wikiart_subset.py   provided data script (not modified)
src/artguide/            data (split, in-memory dataset, transforms), models (baselines, MLP, CNN, ResNet),
                         training (loop, early stopping, optimizers), diagnostics (monitors, sanity checks),
                         evaluation (metrics, bootstrap, latency, final candidates), MLflow tracking
configs/{style,genre}/   one YAML per experiment (the file is the record of the experiment)
scripts/                 command-line entry points (see above)
splits/                  frozen train/val/test splits (ids only, no image)
reports/                 summaries, tables and figures generated from MLflow
tests/                   pytest suite (split leakage, shapes, loop, monitors, metrics, ...)
```
