#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# PyTorch workers use Unix sockets in TMPDIR, which must be on the Linux filesystem.
export TMPDIR=/tmp
if [[ -z "${WSP_SANDBOX_VENV:-}" && -x "$HOME/.venvs/wsp-clip-cpu/bin/python" ]]; then
    export WSP_SANDBOX_VENV="$HOME/.venvs/wsp-clip-cpu"
fi
exec ../clip_zs_sbir/.venv/bin/python harness/driver.py \
    --backend codex --model gpt-6.2-astra --effort max \
    --mem-gb 5 --min-free-ram-gb 1.5 --forward-ram-gb 3 --forward-gpu-gb 4 \
    --gpu-job-threads 3 --forward-workers 1 --eval-workers 1 --eval-batch-size 64 \
    --min-disk-gb 12 --work-quota-gb 24 --pass-gb 6 \
    "$@"
