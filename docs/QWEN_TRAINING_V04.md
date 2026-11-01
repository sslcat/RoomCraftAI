# Qwen 2511 paired LoRA training — V0.4

**Release context:** This is the original V0.4 implementation/mathematics record. Full-model LoRA training subsequently completed in V0.6; see [current results](RESEARCH_RESULTS.md) and [model card](MODEL_CARD.md). Future-tense execution statements below describe that historical milestone.

Executable preparation, not a released model. No pretrained Qwen training run has completed. The owner reports convincing Replicate staging results; those images, settings and provider revision have not been independently reproduced here. Keep SDXL as the existing reference and use Qwen-Image-Edit-2511 for the first adapter experiment.

## Architecture and dependencies

Use `environments/qwen/uv.lock`: Python 3.12.15, PyTorch 2.8.0+cu128, torchvision 0.23.0, Diffusers 0.37.1, Transformers 4.57.6, Accelerate 1.10.1, PEFT 0.17.1, safetensors 0.6.2. The SDXL environment and lock remain separate. Model: `Qwen/Qwen-Image-Edit-2511`, revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`.

The 60-block dual-stream `QwenImageTransformer2DModel` predicts flow velocity. `QwenImageEditPlusPipeline` supplies the official visual-language instruction template, Qwen2.5-VL encoder, Qwen VAE normalization and latent packing. Frozen VAE/VLM outputs are cached once; the training process loads only the transformer. No encoder gradients, SDXL scheduler, inpainting mask channels, external training subprocess, quantization or CPU offloading are used.

The [maintainer's 2511 recipe](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh) confirms paired source/target conditioning, LoRA and `zero_cond_t`. RoomCraft implements its own loop and attention-only adapter selection rather than copying that recipe's MLP/modulation targets. The [pinned Diffusers editing implementation](https://github.com/huggingface/diffusers/blob/v0.37.1/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit_plus.py) defines preprocessing. Its T2I DreamBooth example is not used as an editing trainer.

## Objective and gradients

For a reviewed empty/furnished pair `(x,y)` and instruction `c`, let `z0 = pack(normalize(VAE(y)))`, `zx = pack(normalize(VAE(x)))`, and `h = VLM(x,c)`. VAE encoding uses the posterior mode. Per-channel means and standard deviations come from this checkpoint, not SDXL's scaling factor.

Sample `sigma ~ Uniform[0,1)` and independent `epsilon ~ N(0,I)` in FP32:

```text
z_sigma = (1-sigma) z0 + sigma epsilon
u = d(z_sigma)/d(sigma) = epsilon - z0
p = Qwen(concat(z_sigma,zx), sigma, h, image_grids, text_padding_mask)
L = mean_over_target_tokens_and_channels((p_target - u)^2)
dL/dp_target = 2 (p_target-u) / number_of_target_elements
dL/dp_reference = 0  (no direct reference reconstruction loss)
```

Reference tokens still influence predictions through attention; they are clean conditioning, not independent denoising targets. `zero_cond_t=true` makes upstream Qwen concatenate time `sigma` with zero and route image modulation according to target/reference grid positions. The transformer receives `sigma`, matching inference's scheduler timestep divided by 1000. The inference scheduler integrates from high to low noise. Uniform sigma and unit loss weighting are explicit RoomCraft starting choices, not a claim to reproduce Qwen's undisclosed pretraining distribution or its dynamically shifted inference schedule. Classifier-free prompt dropout is not implemented; guidance behavior must be checked after tuning.

Typical shapes (batch size is currently fixed at one):

| Tensor | Shape |
|---|---|
| Target pixels | `[1,3,1,H,W]` |
| Normalized VAE latent | `[1,16,1,H/8,W/8]` |
| Packed target | `[1,(H/16)*(W/16),64]` |
| Clean source | `[1,N_source,64]`, reference area approximately 1024² |
| VLM features / mask | `[1,L,3584]` / `[1,L]` |
| Transformer input | `[1,N_target+N_source,64]` |
| Grid metadata | `[[[1,H/16,W/16],[1,H_source/16,W_source/16]]]` |

For a 512×384 target, `N_target=768`. At 1024×768 it is 3072. Source VAE area stays at the official approximately 1024² even during a 512 pilot. Visual-language input uses the official approximately 384² area. These two source representations have different purposes. Images are resized without crop; no random augmentation is hidden in the cache.

For a projection `W0 ∈ R^(out×in)`:

```text
W = W0 + s B A, s = alpha/r
A ∈ R^(r×in), B ∈ R^(out×r)
G = dL/dW
dL/dB = s G A^T; dL/dA = s B^T G
```

`W0` is frozen. A uses Kaiming initialization; B starts at zero, so initial inference equals the foundation model. A's first gradient is zero; B learns first. FP32 master adapter parameters and AdamW state are retained under BF16 autocast. For accumulation count K, each microbatch backpropagates `L/K`; clipping and AdamW occur once per optimizer step.

RoomCraft selects `to_q`, `to_k`, `to_v`, `to_out.0`, `add_q_proj`, `add_k_proj`, `add_v_proj`, and `to_add_out`, excluding **the final block's `to_add_out`**. That last text output is discarded and has no gradient path to image loss. This was found by a real transformer backward test, not inferred from parameter names alone. There are **479 projections**, **23,543,808 trainable parameters at rank 8**, or **47,087,616 at rank 16**. See `experiments/v04_model_audit.json`. V0.3's 480-projection estimate predates this correction.

## Data and commands

Follow [paired-data preparation](DATASETS_V04.md). The empty template intentionally fails validation. Never train on the schematic fixtures or turn the three V0.2 unpaired photos into invented training pairs.

```bash
# CPU: seal a genuinely reviewed ledger, then split by property/scene family.
uv run --project environments/qwen python scripts/prepare_qwen_pairs.py data/qwen/reviewed.json --output data/qwen/sealed.json
uv run --project environments/qwen roomcraft split-pairs data/qwen/sealed.json --output data/qwen/pairs.json --seed 42
uv run --project environments/qwen roomcraft validate-pairs data/qwen/pairs.json

# Only on an approved sufficiently large GPU; first stage loads frozen encoders only.
uv run --project environments/qwen roomcraft qwen-cache --manifest data/qwen/pairs.json --output data/qwen/cache-512 --size 512
uv run --project environments/qwen roomcraft train-qwen --cache data/qwen/cache-512/cache.json --config configs/qwen_train_smoke.json --output runs/qwen-smoke
```

The cache stores normalized/packed latents, text embeddings, masks, grids, source-manifest identity, provenance and per-file SHA256 checksums. All three splits are encoded with frozen encoders; **test examples never enter optimization or validation loss**. Caches are private derivative data, not public repository content. A partial cache lacks `cache.json` and cannot be trained. Changed images/instructions require a new manifest/cache. Keep the whole immutable cache for resume.

`parameters.json` reports every trainable tensor and count. `loss.jsonl` records training/validation loss, pre-clipping gradient norms, elapsed time and CUDA allocated/reserved peaks. Validation uses all validation rows with fixed noise/time seeds; loss is diagnostic, not an image-quality metric. Step selection, noise and torch/random seeds are derived from `(seed,optimizer_step,microbatch)` so no hidden dataloader position needs restoring. Sampling is with replacement; `steps` means optimizer updates, not epochs.

Checkpoints contain adapter safetensors, exact config, training-source hashes, AdamW state, disabled-BF16-scaler state, and optimizer step. Writes use the existing atomic-directory checkpoint utility on POSIX storage. Only `steps` may change on resume; data hash, code hashes, model, environment and other settings must match. Resume into a **new output directory**. Replaying a tiny CPU continuation is bit-identical in tests; different GPU/software kernels are not promised bit-identical. Deterministic algorithms are requested with warnings for unsupported operations. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting CUDA.

```bash
python -c 'import json; p=json.load(open("configs/qwen_train_smoke.json")); p["steps"]=3; json.dump(p,open("configs/qwen_train_resume_smoke.json","w"),indent=2)'
uv run --project environments/qwen roomcraft train-qwen --cache data/qwen/cache-512/cache.json --config configs/qwen_train_resume_smoke.json --output runs/qwen-resume-smoke --resume runs/qwen-smoke/step-000002
# After reviewing smoke loss, gradients, memory and resumption; a fresh, bounded pilot:
uv run --project environments/qwen roomcraft train-qwen --cache data/qwen/cache-512/cache.json --config configs/qwen_train_pilot.json --output runs/qwen-pilot
```

No scheduler or distributed trainer is added for a 100-update pilot: learning rate is constant, microbatch is one and effective batch defaults to four. Save frequency and accumulation are configurable. Lowering target resolution reduces activations but not the 38.05 GiB BF16 transformer. Do not interpret the tiny GPU test's memory usage as an estimate for the pretrained model.

## Adapter inference and evaluation

```bash
uv run --project environments/qwen roomcraft edit --image data/authorized-room.jpg --instruction "Furnish with a sofa and coffee table; preserve windows, walls and camera viewpoint." --adapter runs/qwen-pilot/step-000100 --output outputs/qwen-adapter --seed 42
uv run --project environments/qwen python scripts/v04_compare.py --manifest data/qwen/pairs.json --checkpoint runs/qwen-pilot/step-000100 --output outputs/v04/seed42 --size 512 --seed 42 --run
```

The comparison script requires the exact manifest used in training and selects only its test split. Without `--run` it writes a protocol only. It uses the same original photograph, instruction, seed, size, 40 steps and true CFG 4 for baseline and adapter. Metadata comparison rejects setting/environment/model changes. Review `comparison.png`, blind `review.html`, and `human_review.csv` for realism, instruction adherence, walls/windows/doors, floor/perspective, lighting/shadows and artifacts. Leave absent features blank; document failures. Use multiple fixed seeds and reviewers; do not tune on the final test set. Optional manually annotated wall/window/door/floor masks allow regional edge/MAE proxies via `compare_qwen`; no semantic score is fabricated when annotations are missing.

Adapters use **RoomCraft's own `.a/.b` safetensors format**, not PEFT/Diffusers LoRA files. Load them through `roomcraft edit --adapter CHECKPOINT_DIRECTORY`; no PEFT interoperability/export is claimed. Upstream model definitions/attention/VAE/VLM are external; pair validation, cached-data contracts, target-only flow loss, LoRA linear updates/selection, gradient auditing, training orchestration and matched evaluation are project code. Original engineering does not imply a new foundation architecture or a demonstrated quality gain.
