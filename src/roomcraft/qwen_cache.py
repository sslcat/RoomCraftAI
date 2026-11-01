"""Frozen Qwen VAE/VLM preprocessing, pinned to the real Edit Plus implementation."""

import json
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file

from roomcraft.common import environment, model_spec, sha256, write_json
from roomcraft.data import local_path
from roomcraft.editing import EditConfig, edit_size
from roomcraft.preprocess import read_rgb
from roomcraft.qwen_data import validate_pairs
from roomcraft.qwen_objective import validate_tensors


def require_qwen_gpu(min_free_gib: float) -> None:
    if metadata.version("diffusers") != "0.37.1" or metadata.version("transformers") != "4.57.6":
        raise RuntimeError("use the locked environments/qwen environment")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA with BF16 is required")
    if torch.cuda.mem_get_info()[0] / 2**30 < min_free_gib:
        raise RuntimeError(f"need {min_free_gib} GiB free VRAM; planning gate, not a measured fit")


@torch.no_grad()
def encode_pair(
    pipe: Any, source: Any, target: Any, instruction: str, size: int, device: str = "cuda"
) -> dict[str, torch.Tensor]:
    upstream = import_module("diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus")
    cw, ch = upstream.calculate_dimensions(
        upstream.CONDITION_IMAGE_SIZE, source.width / source.height
    )
    rw, rh = upstream.calculate_dimensions(upstream.VAE_IMAGE_SIZE, source.width / source.height)
    tw, th = edit_size(target, size)
    condition = pipe.image_processor.resize(source, ch, cw)
    text, mask = pipe.encode_prompt(
        prompt=instruction, image=[condition], device=torch.device(device), num_images_per_prompt=1
    )
    if mask is None:
        mask = torch.ones(text.shape[:2], device=text.device, dtype=torch.bool)
    tensors = {"text": text, "text_mask": mask.bool()}
    shapes = []
    for key, image, width, height in (("target", target, tw, th), ("source", source, rw, rh)):
        pixels = (
            pipe.image_processor.preprocess(image, height, width)
            .unsqueeze(2)
            .to(device=device, dtype=pipe.vae.dtype)
        )
        latent = pipe._encode_vae_image(pixels, generator=None)
        batch, channels, frames, lh, lw = latent.shape
        if (batch, channels, frames) != (1, 16, 1):
            raise ValueError("unexpected Qwen VAE latent layout")
        tensors[key] = pipe._pack_latents(latent, batch, channels, lh, lw)
        shapes.append([1, lh // 2, lw // 2])
    tensors["shapes"] = torch.tensor(shapes, dtype=torch.int64)
    tensors = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
    validate_tensors(tensors)
    return tensors


def cache_pairs(manifest: Path, output: Path, size: int = 512) -> dict[str, Any]:
    EditConfig(size=size).validate()
    rows = validate_pairs(manifest)
    if output.exists():
        raise ValueError("cache output must be new; caches are immutable")
    require_qwen_gpu(24)
    spec = model_spec("qwen")
    cls = import_module("diffusers").QwenImageEditPlusPipeline
    pipe = cls.from_pretrained(
        spec["repo"],
        revision=spec["revision"],
        transformer=None,
        torch_dtype=torch.bfloat16,
        use_safetensors=True,
        low_cpu_mem_usage=True,
    )
    pipe.text_encoder.requires_grad_(False).eval().to("cuda")
    pipe.vae.requires_grad_(False).eval().to("cuda")
    output.mkdir(parents=True)
    cached = []
    for row in rows:
        tensors = encode_pair(
            pipe,
            read_rgb(local_path(manifest.parent, row["source"])),
            read_rgb(local_path(manifest.parent, row["target"])),
            row["instruction"],
            size,
        )
        filename = row["id"] + ".safetensors"
        save_file(tensors, str(output / filename))
        cached.append(
            {
                "id": row["id"],
                "group_id": row["group_id"],
                "split": row["split"],
                "file": filename,
                "sha256": sha256(output / filename),
            }
        )
    result = {
        "schema": "roomcraft-qwen-cache-v1",
        "base": spec,
        "manifest_sha256": sha256(manifest),
        "size": size,
        "reference_area": 1024**2,
        "environment": environment(),
        "rows": cached,
    }
    # Full source records make the rights/alignment decision auditable without copying private photos.
    write_json(output / "provenance.json", json.loads(manifest.read_text(encoding="utf-8")))
    result["provenance_sha256"] = sha256(output / "provenance.json")
    write_json(output / "cache.json", result)
    return result


def validate_cache(path: Path) -> dict[str, Any]:
    cache = json.loads(path.read_text(encoding="utf-8"))
    if cache.get("schema") != "roomcraft-qwen-cache-v1" or cache["base"] != model_spec("qwen"):
        raise ValueError("incompatible cached model or schema")
    if sha256(path.parent / "provenance.json") != cache["provenance_sha256"]:
        raise ValueError("changed cache provenance")
    provenance = json.loads((path.parent / "provenance.json").read_text(encoding="utf-8"))
    expected = {r["id"]: r for r in provenance["pairs"]}
    groups: dict[str, str] = {}
    ids: set[str] = set()
    for row in cache["rows"]:
        if row["id"] in ids or row["split"] not in ("train", "validation", "test"):
            raise ValueError("duplicate id or invalid split in cache")
        ids.add(row["id"])
        original = expected[row["id"]]
        if any(original[k] != row[k] for k in ("group_id", "split")):
            raise ValueError("cache disagrees with provenance")
        group = row["group_id"]
        if group in groups and groups[group] != row["split"]:
            raise ValueError("cache split leakage")
        groups[group] = row["split"]
        file = local_path(path.parent, row["file"])
        if sha256(file) != row["sha256"]:
            raise ValueError("cache tensor hash mismatch")
        validate_tensors(load_file(str(file)))
    if ids != set(expected) or set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("incomplete cache or missing splits")
    return dict(cache)
