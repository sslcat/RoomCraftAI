#!/usr/bin/env bash
# Execute only within an owner-approved Pod. No infrastructure/API commands.
set -euo pipefail
ROOMCRAFT_ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
if [ "$ROOMCRAFT_ROOT" != /workspace/RoomCraftAI ]; then
  echo 'Place this standalone project at /workspace/RoomCraftAI on verified regional/POSIX storage.' >&2
  exit 1
fi
ROOMCRAFT_FS="$(findmnt -n -T "$ROOMCRAFT_ROOT" -o FSTYPE)"
case "$ROOMCRAFT_FS" in
  ext4|xfs|btrfs|zfs|nfs|nfs4) ;;
  fuse.mfs|fuse.moosefs) python3 scripts/verify_posix_storage.py --root "$ROOMCRAFT_ROOT" ;;
  fuse) case "$(findmnt -n -T "$ROOMCRAFT_ROOT" -o SOURCE)" in mfs#*) python3 scripts/verify_posix_storage.py --root "$ROOMCRAFT_ROOT" ;; *) exit 1 ;; esac ;;
  *) echo "Unverified filesystem $ROOMCRAFT_FS; do not install on an object/FUSE Global Volume." >&2; exit 1 ;;
esac
export HF_HOME="$ROOMCRAFT_ROOT/.cache/huggingface"
export UV_CACHE_DIR="$ROOMCRAFT_ROOT/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$ROOMCRAFT_ROOT/.cache/python"
export TORCH_HOME="$ROOMCRAFT_ROOT/.cache/torch"
export XDG_CACHE_HOME="$ROOMCRAFT_ROOT/.cache/xdg"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$ROOMCRAFT_ROOT"
python3 -m venv .cache/bootstrap
.cache/bootstrap/bin/python -m pip install uv==0.12.23
.cache/bootstrap/bin/uv python install 3.12.15
.cache/bootstrap/bin/uv sync --project environments/qwen --frozen
environments/qwen/.venv/bin/python -m pytest -q
environments/qwen/.venv/bin/ruff check .
environments/qwen/.venv/bin/mypy --cache-dir .cache/mypy-qwen
environments/qwen/.venv/bin/python -m compileall -q src scripts tests
echo 'Environment ready. No pretrained weights downloaded or training launched. Export the same cache variables in each new shell.'
