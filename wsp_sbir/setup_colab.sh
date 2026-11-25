#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
dataset="${1:-/content/Sketchy}"
weights="$PWD/ViT-B-32.pt"
gpu=clip_zs_sbir/.venv/bin/python
cpu=wsp_sbir/.venv_sandbox/bin/python

if [[ ! -d "$dataset/sketch" || ! -d "$dataset/photo" ]]; then
    printf 'Run download_sketchy.py before setup. Missing dataset at %s\n' "$dataset" >&2
    exit 1
fi
if ! command -v bwrap >/dev/null; then
    apt-get update -qq
    apt-get install -y -qq bubblewrap
fi
bwrap --unshare-all --ro-bind /usr /usr --symlink usr/lib /lib --symlink usr/lib64 /lib64 /usr/bin/true

if [[ ! -x "$gpu" ]]; then
    python3 -m venv --system-site-packages clip_zs_sbir/.venv
fi
if ! "$gpu" -c 'import ftfy, regex, torchmetrics' 2>/dev/null; then
    "$gpu" -m pip install --no-cache-dir ftfy regex torchmetrics
fi
"$gpu" -c 'import torch, torchvision; assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name())'

if [[ ! -x "$cpu" ]]; then
    python3 -m venv wsp_sbir/.venv_sandbox
fi
if ! "$cpu" -c 'import torch, numpy, scipy, pandas, sklearn, PIL, sympy, networkx, safetensors; assert torch.version.cuda is None' 2>/dev/null; then
    "$cpu" -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
    "$cpu" -m pip install --no-cache-dir numpy scipy pandas scikit-learn pillow sympy networkx safetensors
fi

if [[ ! -x .colab-tools/node_modules/.bin/codex ]]; then
    npm install --prefix "$PWD/.colab-tools" --no-fund --no-audit @openai/codex
fi
.colab-tools/node_modules/.bin/codex --version

PYTHONPATH="$PWD/clip_zs_sbir" "$gpu" -c 'import os; from clip.clip import _download, _MODELS; _download(_MODELS["ViT-B/32"], os.getcwd())'
for split in sketchy_1 sketchy_2; do
    OMP_NUM_THREADS=3 "$gpu" wsp_sbir/prepare.py --root "$dataset" --dataset "$split" --weights "$weights"
done
