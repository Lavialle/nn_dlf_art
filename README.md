# nn_dlf_art

M2 Deep Learning Project

## Setup

```bash
# 1. Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Clone and install the exact locked environment (Python + PyTorch CUDA 13.0)
git clone https://github.com/Lavialle/nn_dlf_art.git
cd nn_dlf_art
uv sync

# 3. Check the GPU is visible (expects: True)
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

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
