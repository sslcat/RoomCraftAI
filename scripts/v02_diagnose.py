"""Replay w02 through stock Diffusers and inspect tensor contracts without changing them."""

import json
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

from roomcraft.common import seed_all, write_json

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/v02/w02-caption"
meta = json.loads((RUN / "metadata.json").read_text())
seed_all(meta["settings"]["seed"])
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    meta["base"]["repo"],
    revision=meta["base"]["revision"],
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True,
    add_watermarker=False,
)
pipe.enable_vae_tiling()
pipe.enable_model_cpu_offload()
diagnostics = {
    "reference": "w02-caption",
    "method": "stock Diffusers replay; no RoomCraft inference wrapper",
    "scheduler_class": type(pipe.scheduler).__name__,
    "scheduler_config": dict(pipe.scheduler.config),
    "steps_observed": 0,
}


def before_unet(module, args, kwargs):
    diagnostics["steps_observed"] += 1
    if diagnostics["steps_observed"] != 1:
        return
    sample = args[0]
    context = kwargs["encoder_hidden_states"]
    mask = sample[:, 4:5]
    diagnostics.update(
        {
            "unet_input_shape": list(sample.shape),
            "context_shape": list(context.shape),
            "positive_negative_context_rms_difference": float(
                (context[1].float() - context[0].float()).square().mean().sqrt()
            ),
            "mask_min": float(mask.min()),
            "mask_max": float(mask.max()),
            "mask_fraction": float(mask.mean()),
            "first_timestep": float(args[1]),
            "all_scheduler_timesteps": pipe.scheduler.timesteps.cpu().tolist(),
            "all_scheduler_sigmas": pipe.scheduler.sigmas.cpu().tolist(),
            "noisy_latent_std": float(sample[:, :4].float().std()),
            "masked_image_latent_std": float(sample[:, 5:].float().std()),
        }
    )


def after_unet(module, args, output):
    if diagnostics["steps_observed"] == 1:
        pred = output[0]
        diagnostics["prediction_shape"] = list(pred.shape)
        diagnostics["positive_negative_prediction_rms_difference"] = float(
            (pred[1].float() - pred[0].float()).square().mean().sqrt()
        )


before = pipe.unet.register_forward_pre_hook(before_unet, with_kwargs=True)
after = pipe.unet.register_forward_hook(after_unet)
with Image.open(RUN / "input.png") as image, Image.open(RUN / "mask.png") as mask:
    with torch.inference_mode():
        result = pipe(
            prompt=meta["prompt"],
            negative_prompt=meta["negative_prompt"],
            image=image,
            mask_image=mask,
            width=image.width,
            height=image.height,
            num_inference_steps=meta["settings"]["steps"],
            guidance_scale=meta["settings"]["guidance"],
            strength=meta["settings"]["strength"],
            generator=torch.Generator(device="cuda").manual_seed(meta["settings"]["seed"]),
        ).images[0]
before.remove()
after.remove()
out = ROOT / "outputs/v02/diagnostic"
out.mkdir(exist_ok=True)
result.save(out / "stock-replay.png")
with Image.open(RUN / "raw.png") as reference:
    delta = np.abs(np.asarray(result, dtype=np.float32) - np.asarray(reference, dtype=np.float32))
diagnostics["stock_replay_max_pixel_difference_255"] = float(delta.max())
diagnostics["stock_replay_mean_pixel_difference_255"] = float(delta.mean())
write_json(ROOT / "experiments/v02_diagnostics.json", diagnostics)
print(json.dumps(diagnostics, indent=2))
