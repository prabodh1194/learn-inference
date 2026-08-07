#!/usr/bin/env bash
# Provision a freshly-rented NVIDIA box to a working state.
#   bash scripts/setup_gpu.sh
set -euo pipefail

echo "== GPU =="
nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.12
uv sync --extra kernels --extra serve

echo "== verify =="
uv run python -c "
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda)
print('device:', torch.cuda.get_device_name(0))
props = torch.cuda.get_device_properties(0)
print(f'  SMs {props.multi_processor_count}  mem {props.total_memory/1024**3:.1f} GiB')
import triton; print('triton', triton.__version__)
"

uv run python scripts/fetch_model.py
echo "== ready. remember to STOP THE POD when done. =="
