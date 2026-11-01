# Qwen adapter feasibility — V0.3 historical design

V0.4 implements the paired trainer described in [the current training guide](QWEN_TRAINING_V04.md). The record below is the original V0.3 feasibility study. Its proposed 480-projection selection was corrected to 479 after backward testing found that the final text-output projection has no path to image loss. No pretrained adapter training has occurred.

`Qwen-Image-Edit-2511` is a conditional flow-matching transformer, not an SDXL noise-prediction UNet. The existing `roomcraft train` remains SDXL-only; it must not be pointed at Qwen. There is no RoomCraft Qwen training loop or trained adapter in V0.3.

## Verified support

The pinned Diffusers Edit Plus pipeline inherits `QwenImageLoraLoaderMixin`, and its transformer supports PEFT adapters. The full config has **20,430,401,088** transformer parameters and `zero_cond_t=true`. The local meta-device audit (`../experiments/v03_model_audit.json`, local research artifact excluded from the source candidate) counts parameters without allocating weights.

The maintainer implementation in [DiffSynth-Studio's 2511 LoRA example](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/lora/Qwen-Image-Edit-2511.sh) supplies a target image, an edit/source image and a prompt, and explicitly enables zero-condition time. Its [training module](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/qwen_image/model_training/train.py) uses supervised flow matching. Its recipe uses additional modulation/MLP LoRA targets and rank 32; it is evidence of support, not a recipe executed here. The [versioned Diffusers Qwen DreamBooth trainer](https://github.com/huggingface/diffusers/blob/v0.37.1/examples/dreambooth/train_dreambooth_lora_qwen_image.py) is a text-to-image example, not a drop-in paired editing trainer. Do not confuse those objectives.

## Proposed RoomCraft objective

For a rights-cleared aligned pair, let `x` be the empty source photograph, `y` the furnished target and `c` the furnishing instruction. Freeze the visual-language encoder, VAE and base transformer. Normalize VAE latents with the checkpoint's own per-channel means/stds, and pack 2×2 latent patches as the inference pipeline does.

Let `z0 = E(y)`, `zx = E(x)`, and `epsilon ~ N(0,I)`. With a specified sampling density for `sigma in (0,1)`, use

```text
z_sigma = (1 - sigma) z0 + sigma epsilon
target_velocity = epsilon - z0
prediction = v_(theta, A, B)(z_sigma, sigma; zx, visual_text_condition(x, c))
L = E[ w(sigma) mean((prediction - target_velocity)^2) ]
```

The initial implementation should make timestep sampling/weighting explicit (for example uniform sigma and unit weighting), then validate against the pinned upstream scheduler convention. Do not transfer SDXL's epsilon loss, mask concatenation or text encoders into this model. Reference image tokens remain clean and use the 2511 zero-time conditioning branch. Loss is computed over **target tokens**, not reference tokens. Denoising/flow time is distinct from a training update index.

At output 1024×768, target latents have shape `[B,16,1,96,128]`, packed to `[B,3072,64]`. Clean reference latents are packed separately; the official pipeline resizes reference images by area, so their token count can differ. Concatenate target/reference sequences along the token dimension, pass their `img_shapes`, and retain `zero_cond_t=true`. VLM hidden features have width 3584; padding masks and the official instruction template must be retained.

For a frozen linear weight `W0` of shape `[d_out,d_in]`, LoRA gives

```text
W = W0 + (alpha/r) B A
A: [r,d_in], B: [d_out,r]
dL/dB = (alpha/r) G A^T
dL/dA = (alpha/r) B^T G
G = dL/dW
```

One conservative target set is the eight attention projections per block: `to_q`, `to_k`, `to_v`, `to_out.0`, `add_q_proj`, `add_k_proj`, `add_v_proj`, `to_add_out`. Across 60 blocks, the audit finds **480 linear matrices**, each 3072×3072. Rank 16 therefore adds **47,185,920 parameters** (about 0.231% of the base transformer); rank 8 adds 23,592,960. These are design counts, not trained parameters. Start with zero-initialized B and a conventional random A; train only adapter weights. Expanding to MLP/modulation targets changes both memory and behavior and needs its own ablation.

## Data and verification required before training

Use source/target/instruction triples with the same camera pose and intact room geometry. Sources must be empty, targets plausibly furnished, and each pair must have documented provenance, commercial-use permissions and derivative restrictions. Keep room/house/capture identity disjoint between train and validation splits. Avoid furniture removal artifacts that leak the desired object into the source. Rendered pairs can help only if rendering rights and the real-image domain gap are tracked.

The three V0.2 photos are evaluation-only, have no paired target and are not a training dataset. Model-generated outputs are not automatically accurate ground truth. Start with a small independently reviewed pilot; determine dataset size by coverage and held-out learning curves rather than claiming a fabricated sample-count guarantee.

A future trainer needs mixed precision, frozen-base/trainable-parameter assertions, deterministic sampling, adapter-gradient diagnostics, finite-loss checks, validation generations, and optimizer/RNG/sampler checkpoint resumption. Precompute frozen source/target latents and instruction embeddings when augmentation permits. No training command or automatic dataset acquisition is included here.

## Memory planning, not measurements

BF16 transformer weights alone occupy 38.05 GiB. FP32 adapter weights, gradients and two Adam moments for rank 16 require about **0.70 GiB** before any mixed-precision copies. Activations through 60 blocks, attention, source/target tokens and temporary buffers dominate the remaining training memory.

Plan approximately **60–90 GiB** for unquantized, batch-one, ~1024-area LoRA with gradient checkpointing and cached frozen conditioning; start on **80–96 GB** hardware and measure before scaling. This is an engineering estimate, not a tested fit. Keeping the visual-language encoder resident adds substantial memory. A 48 GB training attempt would require lower resolution and/or significant offloading/quantization, with separate performance and quality validation. Multi-GPU sharding or quantized-base LoRA is future work, not implemented.

The prerequisite remains successful **untrained-model visual staging**. Do not use a large training run to conceal an unverified inference baseline or a compositing defect.
