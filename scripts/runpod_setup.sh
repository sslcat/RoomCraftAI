#!/usr/bin/env bash
set -euo pipefail
# Run inside an already-approved Linux GPU Pod. Does not create resources.
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_PYTHON_INSTALL_DIR="${UV_PYTHON_INSTALL_DIR:-/workspace/.local/share/uv/python}"
export TMPDIR="${TMPDIR:-/workspace/tmp}"
mkdir -p "$UV_CACHE_DIR" "$HF_HOME" "$UV_PYTHON_INSTALL_DIR" "$TMPDIR"
python3 -m pip install --user 'uv==0.12.23'
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
uv run python scripts/make_fixture.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())'
printf '%s\n' 'Setup passed. Run scripts/gpu_smoke.py explicitly to download weights and generate.'
