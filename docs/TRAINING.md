# Training and mathematics

No RoomCraft dataset-trained weights are included. Tiny random-model tests establish plumbing and derivatives, not room-furnishing capability. The GPU trainer is project code; it does not launch an external example training script.

## Input contract and objective

For one example: source `x_s` and furnished target `x_0` are `[1,3,H,W]` in `[-1,1]`; edit mask `m` is `[1,1,H,W]`, one inside editable regions. Source and target need matching camera pose and architecture. The source outside the mask should agree with the target; use a mask that includes furniture, contact shadows and relevant reflections. Do not use arbitrary unpaired room photographs.

The frozen VAE yields target latent `z_0 = s E(x_0)` and masked-source latent `z_s = s E(x_s (1-m))`, each `[1,4,H/8,W/8]`, using its configured scale `s`. Training samples from its posterior with a seeded generator. The mask uses nearest-neighbor downsampling. For timestep `t` and Gaussian noise `epsilon`:

```math
z_t = \sqrt{\bar\alpha_t}z_0 + \sqrt{1-\bar\alpha_t}\epsilon.
```

UNet input is `concat(z_t, m_down, z_s)` = `[1,9,H/8,W/8]`. SDXL text conditioning is `[1,77,2048]`, pooled text `[1,1280]`, and size/crop conditioning `[1,6]` = `[H,W,0,0,H,W]` for the prepared working image. The trainer records actual shapes rather than silently assuming them. Prediction is `[1,4,H/8,W/8]`.

Default epsilon prediction uses a region-weighted mean square error. For channel/pixel index `i`, weight `w_i = m_i + lambda_out (1-m_i)`, with `lambda_out=0.25`:

```math
L = \frac{\sum_i w_i (\hat\epsilon_i-\epsilon_i)^2}{\sum_i w_i},
\qquad \frac{\partial L}{\partial\hat\epsilon_i} = \frac{2w_i(\hat\epsilon_i-\epsilon_i)}{\sum_jw_j}.
```

Weights are broadcast over four channels and the denominator includes that multiplicity. A scheduler configured for velocity prediction instead uses `v = sqrt(alpha_bar) epsilon - sqrt(1-alpha_bar) z_0`. Other prediction modes fail. This loss emphasizes editable content; it is **not** a novel geometry-preservation guarantee. A future layout loss needs validated semantic labels and ablation evidence.

## LoRA and gradients

For each attention projection, frozen `W_0` has shape `[d_out,d_in]`. Train `A:[r,d_in]` and `B:[d_out,r]`:

```math
W = W_0 + (\alpha/r)BA, \qquad y = W_0x + (\alpha/r)BAx.
```

If `G = dL/dW`, then `dL/dB = (alpha/r) G A^T` and `dL/dA = (alpha/r) B^T G`. `B` starts at zero and `A` is Kaiming initialized, so the initial forward pass exactly matches the base. The first `A` gradient is zero while `B` can learn; subsequent steps update both. Tests check this behavior and analytic weighted-loss gradients.

Only `to_q`, `to_k`, `to_v`, and `to_out.0` linear projections are adapted. Trainable parameter count per projection is `r(d_in+d_out)`; full shape/count reports go to `parameters.json`. Adapter parameters remain FP32; frozen models use configured precision and VAE encoding remains FP32. FP16 uses GradScaler; BF16 requires hardware support. Accumulation divides each microbatch loss by the accumulation count. Gradients are unscaled, checked for finite norm, logged, and clipped to norm 1 before AdamW.

## Commands

```bash
uv run roomcraft validate-manifest data/rooms/manifest.jsonl
uv run roomcraft train --manifest data/rooms/manifest.jsonl --config configs/train.json --output runs/first

# Increase steps in a copy of the same config; all other fields must match.
uv run roomcraft train --manifest data/rooms/manifest.jsonl --config configs/train_longer.json --resume runs/first/step-001000 --output runs/resumed

# Plumbing only on an approved GPU, never report as dataset training:
uv run roomcraft train --manifest tests/fixtures/manifest.jsonl --config configs/train_smoke.json --output runs/toy-smoke
```

Checkpoints contain project-format `adapter.safetensors`, `config.json`, and restricted-load `state.pt` (optimizer, scaler, completed step). They are atomically renamed from `.partial` directories. They do not contain base weights. Inference uses `--adapter CHECKPOINT_DIRECTORY`; they are not claimed to be PEFT/ComfyUI adapters. Do not load arbitrary untrusted training-state files.

Resume uses identical configuration and manifest hash, except increased total steps. Validation and model/data RNG are derived from step/example seeds, making the next update independent of prior loader consumption. Full GPU determinism is still kernel-dependent. The CPU test verifies resumed AdamW exactly matches uninterrupted updates. A new output directory is required to prevent log/checkpoint overwrites.

Validation averages a fixed, configurable leading subset of the explicit validation split using fixed noise seeds. It reports denoising loss only; use generated-image human review for actual quality. Metrics are JSONL with loss, validation loss, gradient norm/max, duration and peak memory. Checkpoint every 100 steps by default; keep the original config and environment alongside results.
