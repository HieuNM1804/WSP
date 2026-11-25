#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
dataset="${1:-/mnt/d/Sketchy}"
weights="${2:-/mnt/c/Users/Admin/.cache/clip/ViT-B-32.pt}"
gpu=clip_zs_sbir/.venv/bin/python
cpu="${WSP_SANDBOX_VENV:-$HOME/.venvs/wsp-clip-cpu}/bin/python"

if [[ ! -d "$dataset/sketch" || ! -d "$dataset/photo" || ! -f "$weights" ]]; then
    printf '%s\n' "Missing Sketchy directories or checkpoint: $dataset, $weights" >&2
    exit 1
fi
scratch=$(mktemp -d "$PWD/.setup-XXXXXXXX")
trap 'rmdir -- "$scratch" 2>/dev/null || true' EXIT
export TMPDIR="$scratch" PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1

install() {
    local python="$1"
    shift
    if command -v "${WSP_UV_BIN:-uv}" >/dev/null; then
        "${WSP_UV_BIN:-uv}" pip install --python "$python" "$@"
    else
        "$python" -m pip install "$@"
    fi
}

setup_cpu() {
    local check='import networkx, numpy, pandas, PIL, safetensors, scipy, sklearn, sympy, torch; assert torch.version.cuda is None; print("CPU sandbox:", torch.__version__)'
    if [[ -x "$cpu" ]] && "$cpu" -c "$check"; then
        printf 'Reusing CPU environment %s\n' "$cpu"
        return
    fi
    local cpu=wsp_sbir/.venv_sandbox/bin/python
    if [[ ! -x "$cpu" ]]; then
        python3 -m venv wsp_sbir/.venv_sandbox
    fi
    install "$cpu" 'torch @ https://download.pytorch.org/whl/cpu/torch-2.6.0%2Bcpu-cp312-cp312-linux_x86_64.whl' \
        numpy scipy scikit-learn pandas sympy networkx safetensors pillow
    "$cpu" -c "$check"
}

setup_gpu() {
    local check='import ftfy, regex, torch, torchmetrics, torchvision; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
    if [[ ! -x "$gpu" ]]; then
        python3 -m venv clip_zs_sbir/.venv
    fi
    if ! "$gpu" -c "$check" 2>/dev/null; then
        install "$gpu" \
            'torch @ https://download.pytorch.org/whl/cu124/torch-2.6.0%2Bcu124-cp312-cp312-linux_x86_64.whl' \
            'torchvision @ https://download.pytorch.org/whl/cu124/torchvision-0.21.0%2Bcu124-cp312-cp312-linux_x86_64.whl' \
            -r clip_zs_sbir/requirements.txt
        "$gpu" -c "$check"
    fi
    for split in sketchy_1 sketchy_2; do
        OMP_NUM_THREADS=3 "$gpu" wsp_sbir/prepare.py --root "$dataset" --dataset "$split" --weights "$weights"
    done
}

setup_cpu &
cpu_pid=$!
setup_gpu &
gpu_pid=$!
status=0
wait "$cpu_pid" || status=$?
wait "$gpu_pid" || status=$?
exit "$status"
