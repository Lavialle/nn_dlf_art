# nn_dlf_art

M2 Deep Learning Project

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

## Build the WikiArt subset

```bash
# List the available classes
uv run python make_wikiart_subset.py --list-classes --label style

# Check the plan without downloading images
uv run python make_wikiart_subset.py --label style \
    --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
    --per-class 300 --size 128 --out data/wikiart_style5 --dry-run

# Build it (same command without --dry-run)
uv run python make_wikiart_subset.py --label style \
    --classes Impressionism Baroque Cubism Ukiyo_e Pop_Art \
    --per-class 300 --size 128 --out data/wikiart_style5
```
