# Historical foundation decision — 2026-10-27

The current V1.1 Studio uses pinned Qwen-Image-Edit-2511. The original comparison and SDXL-first decision below are historical research context, not current quick-start instructions. See [current model card](MODEL_CARD.md).

Recommend **SDXL Inpainting 0.1**, initially mask-only, with separate Canny/depth ablations. It offers an explicit nine-channel inpainting contract and a practical single-GPU LoRA path. This is an engineering choice for a verifiable baseline, not a claim that SDXL has the best image quality. No comparative image benchmark has been run.

| Approach / exact checkpoint | License and distribution | Expected strengths / preservation risks | LoRA and ecosystem | Planning VRAM at ~1024, batch 1 |
| --- | --- | --- | --- | --- |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`, optionally `diffusers/controlnet-canny-sdxl-1.0` or `diffusers/controlnet-depth-sdxl-1.0` | CreativeML Open RAIL++-M: commercial use permitted subject to restrictions, license/notice retention and downstream use restrictions. Hosted service counts as distribution. Not MIT weights. | Explicit editable mask; edges anchor straight lines; depth supports perspective. Older prompt following; changed openings, distorted furniture and seams remain possible. | Mature Diffusers SDXL pipeline and attention LoRA; lowest implementation complexity here. This repo owns its training loop. | Inference roughly 10–16 GB mask-only, 14–22 GB with one ControlNet; offloading lowers resident VRAM. LoRA roughly 24–40 GB at 1024 with checkpointing, less at 512. Target 48 GB. |
| `black-forest-labs/FLUX.1-Fill-dev` | Actual reviewed license is FLUX.1 dev Non-Commercial **v1.1.1**. Commercial/production model use and derivatives require separate BFL permission. Outputs have a commercial-use allowance, with restrictions including competitive model training. Output permission does not license commercial model deployment. Redistribution requires license, notices and derivative disclosure. | Purpose-built in/outpainting; strong reported visual blending. Masks help local edits but do not certify geometry. | Diffusers `FluxFillPipeline`; transformer LoRA is viable but Fill-specific conditioning and training need additional integration. Medium/high complexity. | 12B denoiser alone ~24 GB at 16-bit; full inference roughly 32–48+ GB, offload useful. LoRA ~40–64+ GB without base quantization; 48 GB requires memory tuning. |
| `Qwen/Qwen-Image-Edit-2511` | Model card declares Apache-2.0; actual Apache text reviewed in official Qwen-Image repository. Commercial use and redistribution allowed under notice/license/change requirements and patent provisions; trademarks excluded. Checkpoint repo has no standalone license file; preserve card plus upstream text. | Instruction-based editing and reported improved consistency/geometric reasoning. Whole-image editing can drift outside intended regions; not a direct SDXL masked replacement. | Actively maintained Qwen-Image family, Diffusers `QwenImageEditPlusPipeline`, community/maintainer LoRA training integrations. Highest integration/memory complexity here. | 20B family denoiser ~40 GB at 16-bit before text/vision encoder, VAE and activations. Budget ~48–80 GB inference, ~64–96+ GB LoRA without quantization/offload. 48 GB needs those optimizations. |

These memory ranges are engineering estimates, **not measured results or guarantees**. Resolution, attention kernels, text encoders, optimizer, precision and offloading materially change them. Full fine-tuning is out of scope. Qwen 2511 is a concrete open checkpoint selected for comparison, not a claim that it is the newest Qwen offering. The pinned SDXL environment does not promise Qwen compatibility; evaluate Qwen in its own lockfile/environment if selected later.

## Immutable model revisions

The packaged [models.json](../src/roomcraft/models.json) is the executable source of truth:

| Key | Revision |
| --- | --- |
| inpaint | `115134f363124c53c7d878647567d04daf26e41e` |
| canny | `eb115a19a10d14909256db740ed109532ab1483c` |
| depth | `17bb97973f29801224cd66f192c5ffacf82648b4` |
| flux | `358293da0354175698b67ec8299acf928313a78a` |
| qwen | `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` |

Direct dependencies: Python 3.12, PyTorch 2.8.0+cu128, Diffusers 0.35.1, Transformers 4.56.2, Accelerate 1.10.1, PEFT 0.17.1, safetensors 0.6.2, huggingface-hub 0.34.4, NumPy 2.2.6, Pillow 11.3.0, OpenCV headless 4.12.0.88. `uv.lock` pins transitive dependencies. This deliberately stable SDXL stack is not advertised as latest.

## Primary evidence

- [SDXL inpainting card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1), which explicitly links the [actual SDXL license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md). The inpaint and ControlNet repos declare Open RAIL++ rather than containing independent license texts.
- [Canny card](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0), [depth card](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0), [versioned ControlNet API](https://huggingface.co/docs/diffusers/v0.35.1/en/api/pipelines/controlnet_sdxl), [versioned LoRA guide](https://huggingface.co/docs/diffusers/v0.35.1/en/training/lora).
- [FLUX Fill card](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev), [actual Fill license](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md), [BFL Fill implementation](https://github.com/black-forest-labs/flux/blob/main/docs/fill.md).
- [Qwen edit card](https://huggingface.co/Qwen/Qwen-Image-Edit-2511), [official maintained repository](https://github.com/QwenLM/Qwen-Image), [actual Apache license](https://github.com/QwenLM/Qwen-Image/blob/main/LICENSE), [Diffusers Qwen API](https://huggingface.co/docs/diffusers/api/pipelines/qwenimage), [maintainer training integration](https://verl-omni.readthedocs.io/en/latest/examples/qwen_image_edit/flowgrpo_trainer_qwen_image_edit.html).

License texts are retained under `docs/sources/` for review. Review upstream terms again before publishing weights or a hosted service; this project has neither done that nor acquired a FLUX commercial license. No license here clears rights in private input photographs or third-party trademarks.
