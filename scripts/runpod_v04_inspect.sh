#!/usr/bin/env bash
# Read-only. Run in an already approved Pod; never provisions or mounts anything.
set -euo pipefail
for path in /workspace /workspace-global /workspace/RoomCraftAI; do
  if [ -e "$path" ]; then
    printf '\nMount covering %s\n' "$path"
    findmnt -T "$path" -o TARGET,SOURCE,FSTYPE,OPTIONS
    stat -f -c 'filesystem=%T' "$path"
    df -h "$path"
  fi
done
printf '\nRoomCraft processes on THIS Pod only (not other Global Volume clients):\n'
pgrep -af 'roomcraft|train-qwen|qwen-cache' || true
printf '\nGPU inventory\n'
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv
printf '\nThis cannot establish account-wide attachments or cross-Pod write safety. Check the Runpod console. Do not write shared logs, caches or active checkpoints to a Global Volume.\n'
