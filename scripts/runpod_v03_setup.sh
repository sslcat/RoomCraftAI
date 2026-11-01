#!/usr/bin/env bash
set -euo pipefail
# Run manually INSIDE an already-approved Pod. Never creates paid resources.
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.local/share/uv/python}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$UV_PYTHON_INSTALL_DIR"
# Install pinned uv into a small venv, leaving system Python/PyTorch untouched.
python3 -m venv /workspace/roomcraft-bootstrap
/workspace/roomcraft-bootstrap/bin/python -m pip install 'uv==0.12.23'
UV=/workspace/roomcraft-bootstrap/bin/uv
"$UV" python install 3.12.15
"$UV" sync --project environments/qwen --frozen --python 3.12.15
PY=environments/qwen/.venv/bin/python
"$PY" -m pytest -q
"$PY" -m roomcraft edit-preflight --output experiments/v03_runpod_preflight.json
"$PY" -c 'import json; assert json.load(open("experiments/v03_runpod_preflight.json"))["can_attempt"], "Resource preflight failed; no weights downloaded"'
printf '%s\n' 'Setup checked. No model weights or generation started. Run the documented smoke test explicitly.'
