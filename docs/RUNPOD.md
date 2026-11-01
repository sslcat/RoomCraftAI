# Bounded on-demand Runpod worker

The local studio never provisions paid resources. Obtain fresh owner approval, inspect a live quote, explicitly start one dedicated worker, then stop it. Previous research budgets are not new authorization.

The helper reuses the verified credential resolver, exact Pod name/ID/volume checks, detached deadline guard, SSH and DELETE followed by fresh GET 404 verification. It creates no volumes and never mounts Global Volumes. Use an existing dedicated STANDARD regional volume named `roomcraft-training`; GPUs are selected only beside that volume.

## Operator configuration

Set `RUNPOD_API_KEY` securely in the process, or use an existing Windows Credential Manager entry selected by `ROOMCRAFT_RUNPOD_CREDENTIAL_TARGET` (default `Runpod/RoomCraftAI`). `ROOMCRAFT_SSH_KEY` points to an existing authorized key. Never copy secrets into the repository. The ignored `.cache/runpod-owner.json` may store nonsecret credential-target, SSH-path and volume-ID bindings.

```bash
uv run --project environments/app python scripts/app_runpod.py quote --volume YOUR_ROOMCRAFT_VOLUME_ID
```

`configs/app_worker.json` pins the proven CUDA 12.8 image, compatible 80 GB GPUs, 30 GB container disk, isolated `/workspace/RoomCraftAI` root and five-minute idle shutdown. Refresh live availability/pricing before deployment. Billing reporting can lag; no current-hour records does not mean free usage.

## Explicit approved startup

After fresh approval, copy `configs/app_approval.example.json` to an ignored `.cache` file and record the actual approval, UTC timestamp, budget and maximum minutes. This receipt cannot manufacture permission. It expires after 24 hours; bounds are 10-60 minutes and at most $10. Compute for the entire lifetime, storage and reserve must fit the approved budget.

```bash
uv run --project environments/app python scripts/app_runpod.py start --volume YOUR_ROOMCRAFT_VOLUME_ID --approval .cache/app-approved.json --state .cache/runpod-mvp-FRESH
uv run --project environments/app python scripts/launch_app.py --runpod-session .cache/runpod-mvp-FRESH
```

The guard arms before provisioning. Startup verifies identity, CUDA, price and SSH, then uploads an isolated code snapshot. Cached Qwen weights and matching LoRA/RGA checkpoints must already exist on dedicated RoomCraft storage. Remote inference is offline; missing caches fail instead of starting long downloads. Model/checkpoint files are not overwritten.

No generation starts with under eleven minutes remaining. Active requests heartbeat; idle shutdown is five minutes. The absolute deadline is measured from Pod creation, with two minutes reserved for deletion. The controller is independent of the inference process but depends on Windows/network availability. This is supervised operation, not an unattended cloud-independent guarantee. Startup refuses when Windows sleep prevention is unavailable.

## Stop and verify

```bash
uv run --project environments/app python scripts/app_runpod.py status --state .cache/runpod-mvp-FRESH
uv run --project environments/app python scripts/app_runpod.py stop --state .cache/runpod-mvp-FRESH
```

The exact session's `app-termination.json` must contain `fresh_404: true`. Process exit alone is insufficient. Dedicated network storage remains retained and billable; no retention extension or unrelated-resource cleanup is implied.

Historical allocation wrappers/receipts remain private. The public candidate retains the common lifecycle library and fresh-approval MVP entrypoint, excluding old account-bound allocation entrypoints. Never reuse a historical mission budget as approval.

The remote-control helper uses the UI Python environment and standard-library authentication/SSH code. A small local computer does not need to install local Torch just to control an approved remote worker. Graceful UI closure requests exact-worker termination; forced exits rely on the independent idle/deadline guard. Inspect the termination receipt. Each remote job also records whether its temporary directory removal was confirmed.
