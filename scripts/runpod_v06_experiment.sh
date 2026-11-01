#!/usr/bin/env bash
# Only in an owner-approved RoomCraft Pod under an independent deadline controller.
set -euo pipefail
test "${ROOMCRAFT_CLOUD_SUPERVISED:-${ROOMCRAFT_INDEPENDENT_GUARD:-}}" = yes || { echo 'Independent deadline controller required'; exit 1; }
test "${ROOMCRAFT_V06_APPROVED:-}" = yes || { echo 'Explicit approval flag required'; exit 1; }
test "$(pwd -P)" = /workspace/RoomCraftAI || { echo 'Wrong isolated project root'; exit 1; }
test "$#" -eq 1 && { test "$1" = smoke || test "$1" = pilot; } || exit 1
case "$(findmnt -n -T "$PWD" -o FSTYPE)" in
  ext4|xfs|btrfs|zfs|nfs|nfs4|fuse.mfs|fuse.moosefs) python3 scripts/verify_posix_storage.py --root "$PWD" ;;
  fuse) case "$(findmnt -n -T "$PWD" -o SOURCE)" in mfs#*) python3 scripts/verify_posix_storage.py --root "$PWD" ;; *) exit 1 ;; esac ;;
  *) echo 'Refusing active training on unverified/object storage'; exit 1 ;;
esac
for path in .cache/huggingface .cache/uv .cache/python .cache/torch .cache/xdg .cache/bootstrap runs data environments/qwen/.venv; do
  case "$(realpath -m "$PWD/$path")" in
    "$PWD"/*) ;;
    *) echo "Path escapes isolated RoomCraft root: $path"; exit 1 ;;
  esac
done
export HF_HOME="$PWD/.cache/huggingface"
export UV_CACHE_DIR="$PWD/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$PWD/.cache/python"
export TORCH_HOME="$PWD/.cache/torch"
export XDG_CACHE_HOME="$PWD/.cache/xdg"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONUNBUFFERED=1
mkdir -p runs
exec 9>runs/.v06-writer.lock
flock -n 9 || { echo 'Another RoomCraft V0.6 writer is active'; exit 1; }
test ! -e runs/v06-first || { echo 'Run exists; inspect before recovery'; exit 1; }
mkdir runs/v06-first
exec > >(tee runs/v06-first/console.log) 2>&1
bash scripts/runpod_v04_inspect.sh
bash scripts/runpod_v04_setup.sh
PY=environments/qwen/.venv/bin/python
if $PY -c 'import torch; raise SystemExit(torch.cuda.get_device_capability()[0] != 12)'; then
  $PY scripts/check_blackwell_stack.py --exercise --output runs/v06-first/blackwell-kernels.json
fi
$PY -m roomcraft validate-pairs data/v06_ready/manifest.json
$PY scripts/v06_gpu_probe.py --output runs/v06-first/model-loading.json
$PY -m roomcraft qwen-cache --manifest data/v06_ready/manifest.json --output runs/v06-first/cache --size 512
$PY -m roomcraft train-qwen --cache runs/v06-first/cache/cache.json --config configs/qwen_train_v06_first.json --output runs/v06-first/smoke-first
$PY -m roomcraft train-qwen --cache runs/v06-first/cache/cache.json --config configs/qwen_train_v06_smoke.json --output runs/v06-first/smoke-resumed --resume runs/v06-first/smoke-first/step-000001
$PY scripts/verify_v06_training.py --root runs/v06-first --manifest data/v06_ready/manifest.json
if test "$1" = pilot; then
  test -n "${ROOMCRAFT_DEADLINE_UNIX:-}" || { echo 'Independent deadline is required for pilot'; exit 1; }
  set +e
  $PY scripts/v06_pilot_budget.py --root runs/v06-first --deadline "$ROOMCRAFT_DEADLINE_UNIX"
  ROOMCRAFT_BUDGET_STATUS=$?
  set -e
  if test "$ROOMCRAFT_BUDGET_STATUS" -eq 3; then
    echo 'Verified real smoke retained; pilot skipped because measured time budget is insufficient.'
    exit 0
  fi
  test "$ROOMCRAFT_BUDGET_STATUS" -eq 0 || exit "$ROOMCRAFT_BUDGET_STATUS"
  $PY -m roomcraft train-qwen --cache runs/v06-first/cache/cache.json --config configs/qwen_train_v06_pilot.json --output runs/v06-first/pilot
  $PY scripts/verify_v06_training.py --root runs/v06-first --manifest data/v06_ready/manifest.json --pilot
  $PY scripts/v04_compare.py --manifest data/v06_ready/manifest.json --checkpoint runs/v06-first/pilot/step-000100 --output runs/v06-first/evaluation --size 512 --seed 42 --ids family_05_v00 family_18_v00 family_19_v00 --run
fi
echo 'Workflow completed. Independent RoomCraft controller must terminate the exact Pod and verify 404.'
