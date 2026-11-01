"""Executable SDXL inpainting with optional frozen structural ControlNet."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from roomcraft.common import environment, model_spec, seed_all, sha256, write_json
from roomcraft.lora import inject_lora, load_adapter
from roomcraft.preprocess import canny, composite, prepare, save_prepared


@dataclass
class InferenceConfig:
    size: int = 1024
    steps: int = 30
    guidance: float = 7.5
    strength: float = 0.99
    seed: int = 42
    conditioning: str = "none"
    control_scale: float = 0.5
    offload: bool = False

    def validate(self) -> None:
        if self.steps < 2 or not 0 < self.strength <= 1 or int(self.steps * self.strength) < 1:
            raise ValueError("steps/strength produce no denoising steps")
        if not 0 <= self.seed < 2**63 or not 0 <= self.guidance <= 30:
            raise ValueError("invalid seed or guidance")
        if self.conditioning not in {"none", "canny", "depth"} or not 0 <= self.control_scale <= 2:
            raise ValueError("invalid structural conditioning")


def load_pipeline(config: InferenceConfig) -> Any:
    from diffusers import (
        ControlNetModel,
        StableDiffusionXLControlNetInpaintPipeline,
        StableDiffusionXLInpaintPipeline,
    )

    spec = model_spec("inpaint")
    kwargs: dict[str, Any] = {
        "revision": spec["revision"],
        "torch_dtype": torch.float16,
        "variant": "fp16",
        "use_safetensors": True,
        "add_watermarker": False,
    }
    if config.conditioning == "none":
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(spec["repo"], **kwargs)
    else:
        control = model_spec(config.conditioning)
        controlnet = ControlNetModel.from_pretrained(
            control["repo"],
            revision=control["revision"],
            torch_dtype=torch.float16,
            variant="fp16" if config.conditioning == "canny" else None,
            use_safetensors=True,
        )
        pipe = StableDiffusionXLControlNetInpaintPipeline.from_pretrained(
            spec["repo"], controlnet=controlnet, **kwargs
        )
    pipe.enable_vae_tiling()
    if config.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def infer(
    image_path: Path,
    mask_path: Path | None,
    style: str,
    output: Path,
    config: InferenceConfig,
    depth_path: Path | None = None,
    adapter: Path | None = None,
    prompt_override: str | None = None,
) -> dict[str, Any]:
    config.validate()
    if not style.strip():
        raise ValueError("style must not be empty")
    if prompt_override is not None and not prompt_override.strip():
        raise ValueError("prompt override must not be empty")
    if config.conditioning == "depth" and depth_path is None:
        raise ValueError("depth conditioning requires --depth: aligned near-white/far-black map")
    if depth_path is not None and config.conditioning != "depth":
        raise ValueError("--depth is only valid with --conditioning depth")
    image, mask = prepare(image_path, mask_path, config.size)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA required for inference. Use `roomcraft preprocess` for CPU checks."
        )
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    seed_all(config.seed)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    pipe = load_pipeline(config)
    if adapter:
        meta = json.loads((adapter / "config.json").read_text())
        if meta["base"] != model_spec("inpaint"):
            raise ValueError("adapter base checkpoint differs")
        inject_lora(pipe.unet, meta["rank"], meta["alpha"])
        load_adapter(pipe.unet, adapter / "adapter.safetensors")
    prompt = f"Photorealistic interior photograph, furnish this room in {style} style, realistic furniture scale, preserve existing walls, windows, doors and camera viewpoint"
    if prompt_override is not None:
        prompt = prompt_override
    negative = "distorted architecture, extra windows, extra doors, floating furniture, warped furniture, illustration, text, watermark"
    kwargs: dict[str, Any] = {}
    if config.conditioning == "canny":
        kwargs["control_image"] = canny(image)
    elif depth_path:
        with Image.open(depth_path) as depth:
            from roomcraft.preprocess import read_rgb

            if depth.size != read_rgb(image_path).size:
                raise ValueError("depth dimensions must match input")
            kwargs["control_image"] = depth.convert("RGB").resize(
                image.size, Image.Resampling.BILINEAR
            )
    if config.conditioning != "none":
        kwargs["controlnet_conditioning_scale"] = config.control_scale
    torch.cuda.synchronize()
    generation_start = time.perf_counter()
    with torch.inference_mode():
        raw = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=image,
            mask_image=mask,
            width=image.width,
            height=image.height,
            num_inference_steps=config.steps,
            guidance_scale=config.guidance,
            strength=config.strength,
            generator=torch.Generator(device="cuda").manual_seed(config.seed),
            **kwargs,
        ).images[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    save_prepared(image, mask, output)
    raw.save(output / "raw.png")
    composite(image, raw, mask).save(output / "output.png")
    if "control_image" in kwargs:
        kwargs["control_image"].save(output / "control.png")
    meta = {
        "roomcraft_version": "0.1.0",
        "base": model_spec("inpaint"),
        "control": model_spec(config.conditioning) if config.conditioning != "none" else None,
        "adapter_sha256": sha256(adapter / "adapter.safetensors") if adapter else None,
        "settings": asdict(config),
        "prompt": prompt,
        "negative_prompt": negative,
        "input_sha256": sha256(image_path),
        "mask_sha256": sha256(output / "mask.png"),
        "depth_sha256": sha256(depth_path) if depth_path else None,
        "mask_origin": "user" if mask_path else "heuristic rectangle; not segmentation",
        "working_size": list(image.size),
        "generation_seconds": generation_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "environment": environment(),
        "warning": "Architecture preservation is not guaranteed; raw and composited outputs differ.",
    }
    write_json(output / "metadata.json", meta)
    return meta
