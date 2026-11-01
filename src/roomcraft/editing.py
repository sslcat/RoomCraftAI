"""Separate, unquantized Qwen instruction-editing backend.

Use environments/qwen; SDXL dependencies and inference remain independent.
"""

import math
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import psutil
import torch
from PIL import Image

from roomcraft.common import environment, model_spec, seed_all, sha256, write_json
from roomcraft.preprocess import read_rgb


@dataclass
class EditConfig:
    size: int = 1024
    steps: int = 40
    true_cfg_scale: float = 4.0
    seed: int = 42
    negative_prompt: str = " "
    offload: bool = False

    def validate(self) -> None:
        if not 256 <= self.size <= 1536 or self.size % 64:
            raise ValueError("size must be a multiple of 64 between 256 and 1536")
        if not 1 <= self.steps <= 100 or not 0 <= self.seed < 2**63:
            raise ValueError("invalid steps or seed")
        if not math.isfinite(self.true_cfg_scale) or not 1 <= self.true_cfg_scale <= 10:
            raise ValueError("true_cfg_scale must be finite and between 1 and 10")


def edit_size(image: Image.Image, size: int) -> tuple[int, int]:
    """Match V0.2 output dimensions without cropping or creating an edit mask."""
    scale = size / max(image.size)
    width = max(64, round(image.width * scale / 64) * 64)
    height = max(64, round(image.height * scale / 64) * 64)
    return width, height


def memory_reasons(free_gpu_gib: float, free_ram_gib: float, offload: bool) -> list[str]:
    """Conservative planning gates, not measurements or guarantees of fitting."""
    reasons = []
    gpu_floor = 46 if offload else 60
    if free_gpu_gib < gpu_floor:
        reasons.append(f"Need at least {gpu_floor} GiB free GPU memory for this unquantized mode")
    if free_ram_gib < 64:
        reasons.append("Need at least 64 GiB available host RAM for conservative model loading")
    return reasons


def edit_preflight(offload: bool = False) -> dict[str, Any]:
    cuda = torch.cuda.is_available()
    free_gpu, total_gpu = torch.cuda.mem_get_info() if cuda else (0, 0)
    ram = psutil.virtual_memory()
    reasons = memory_reasons(free_gpu / 2**30, ram.available / 2**30, offload)
    versions = {name: metadata.version(name) for name in ("diffusers", "transformers", "torch")}
    if versions["diffusers"] != "0.37.1" or versions["transformers"] != "4.57.6":
        reasons.append(
            "Use the locked environments/qwen environment (Diffusers 0.37.1, Transformers 4.57.6)"
        )
    if not cuda:
        reasons.append("CUDA is unavailable")
    elif not torch.cuda.is_bf16_supported():
        reasons.append("This backend requires native BF16 support")
    return {
        "status": "ready_for_gpu_smoke_test" if not reasons else "blocked_before_weight_download",
        "can_attempt": not reasons,
        "reasons": reasons,
        "base": model_spec("qwen"),
        "gpu": torch.cuda.get_device_name() if cuda else None,
        "gpu_total_gib": total_gpu / 2**30,
        "gpu_free_gib": free_gpu / 2**30,
        "host_total_gib": ram.total / 2**30,
        "host_available_gib": ram.available / 2**30,
        "packages": versions,
        "offload": offload,
        "quantization": "none",
        "basis": "Transformer BF16 tensors alone are 38.05 GiB; total stored weights about 53.74 GiB. Gates include planning headroom, not measured peak memory.",
        "recommended": "80 GB A100/H100, 128 GB host RAM, BF16 resident inference; no cloud resource is created",
    }


def load_edit_pipeline(config: EditConfig) -> Any:
    # Optional newer backend: unavailable in the separately preserved SDXL environment.
    pipeline_class = import_module("diffusers").QwenImageEditPlusPipeline

    spec = model_spec("qwen")
    pipe = pipeline_class.from_pretrained(
        spec["repo"],
        revision=spec["revision"],
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    if not pipe.transformer.config.zero_cond_t:
        raise RuntimeError("2511 requires zero_cond_t=True; incompatible transformer/config")
    if config.offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")
    return pipe


def edit(
    image_path: Path,
    instruction: str,
    output: Path,
    config: EditConfig,
    adapter: Path | None = None,
    *,
    rga_checkpoint: Path | None = None,
    rga_strength: float = 1.0,
    rga_mask: Path | None = None,
    rga_depth: Path | None = None,
    rga_intervention: Path | None = None,
    rga_edge_contrast: bool = False,
) -> dict[str, Any]:
    config.validate()
    if not math.isfinite(rga_strength) or not 0 <= rga_strength <= 4:
        raise ValueError("RGA strength must be finite in [0,4]")
    if rga_checkpoint is None and (
        rga_mask is not None
        or rga_depth is not None
        or rga_intervention is not None
        or rga_edge_contrast
    ):
        raise ValueError("RGA cues require an RGA checkpoint")
    if rga_intervention is not None and (rga_mask is not None or rga_depth is not None):
        raise ValueError("explicit intervention cannot silently mix with other geometry cues")
    if rga_checkpoint is not None and adapter is None:
        raise ValueError("RGA requires its matching frozen RoomCraft LoRA")
    if not instruction.strip():
        raise ValueError("instruction must not be empty")
    if output.exists() and any(output.iterdir()):
        raise ValueError("output directory must be new or empty")
    original = read_rgb(image_path)
    target = edit_size(original, config.size)
    intervention = None
    if rga_intervention is not None:
        from roomcraft.causality import load_intervention

        intervention = load_intervention(rga_intervention, image_path, target)
    preflight = edit_preflight(config.offload)
    if not preflight["can_attempt"]:
        raise RuntimeError("Qwen preflight: " + "; ".join(preflight["reasons"]))
    seed_all(config.seed)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    pipe = load_edit_pipeline(config)
    adapter_info = None
    if adapter is not None:
        from roomcraft.qwen_training import load_training_adapter

        adapter_info = load_training_adapter(pipe.transformer, adapter)
    torch.cuda.synchronize()
    rga_setup_start = time.perf_counter()
    with torch.inference_mode():
        context: Any = nullcontext(None)
        if rga_checkpoint is not None:
            from roomcraft.rga_inference import rga_edit_context

            context = rga_edit_context(
                pipe.transformer,
                rga_checkpoint,
                adapter_info["sha256"] if adapter_info else None,
                original,
                target,
                rga_strength,
                rga_mask,
                rga_depth,
                intervention,
                rga_edge_contrast,
            )
        with context as rga_info:
            torch.cuda.synchronize()
            if rga_info is not None:
                rga_info["setup_seconds"] = time.perf_counter() - rga_setup_start
            generation_start = time.perf_counter()
            raw = pipe(
                image=[original],
                prompt=instruction,
                negative_prompt=config.negative_prompt,
                width=target[0],
                height=target[1],
                num_inference_steps=config.steps,
                true_cfg_scale=config.true_cfg_scale,
                num_images_per_prompt=1,
                generator=torch.Generator(device="cuda").manual_seed(config.seed),
            ).images[0]
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - generation_start
    if raw.size != target:
        raise RuntimeError(f"Unexpected output size {raw.size}, requested {target}")
    output.mkdir(parents=True, exist_ok=True)
    original.save(output / "source.png")
    original.resize(target, Image.Resampling.LANCZOS).save(output / "input.png")
    raw.save(output / "raw.png")
    raw.save(output / "output.png")
    meta = {
        "roomcraft_version": metadata.version("roomcraft-ai"),
        "backend": "qwen-image-edit-2511",
        "pipeline_class": type(pipe).__name__,
        "base": model_spec("qwen"),
        "adapter_sha256": adapter_info["sha256"] if adapter_info else None,
        "adapter": adapter_info,
        "settings": asdict(config),
        "prompt": instruction,
        "input_sha256": sha256(image_path),
        "output_sha256": sha256(output / "output.png"),
        "source_size": list(original.size),
        "working_size": list(target),
        "conditioning_preprocessing": "Original RGB photo passed to upstream processor; visual encoder uses ~384^2 area and VAE reference ~1024^2 area. Output dimensions match V0.2.",
        "dtype": "bfloat16",
        "quantization": "none",
        "compositing": "none",
        "scheduler_class": type(pipe.scheduler).__name__,
        "scheduler_config": dict(pipe.scheduler.config),
        "zero_cond_t": bool(pipe.transformer.config.zero_cond_t),
        "generation_seconds": generation_seconds,
        "total_seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
        "environment": environment(),
        "preflight": preflight,
        "warning": "Whole-image instruction editing; architecture, lighting and identity may drift. No protected pixels or quality guarantee.",
    }
    if rga_checkpoint is not None:
        meta["geometry_adapter"] = rga_info
    write_json(output / "metadata.json", meta)
    return meta
