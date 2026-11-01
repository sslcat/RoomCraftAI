"""Load only project RGA tensors; preserve the pinned foundation and frozen LoRA."""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file

from roomcraft.common import model_spec, sha256
from roomcraft.rga import GeometryEncoder, RGAConfig, geometry_conditioning, geometry_input


@contextmanager
def residual_diagnostics(model: Any, record: dict[str, Any]) -> Iterator[None]:
    """Observe first-call target residual arithmetic; never replace model outputs."""
    active: dict[str, Any] = {}
    handles = []

    def inputs(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        residuals = kwargs.get("controlnet_block_samples")
        if residuals is not None:
            active["residual"] = residuals[0]
            shape = kwargs["img_shapes"][0][0]
            active["target_tokens"] = int(shape[0] * shape[1] * shape[2])

    def block(label: str) -> Any:
        def observe(_module: Any, _args: Any, result: Any) -> None:
            if label in record or "residual" not in active:
                return
            n = active["target_tokens"]
            hidden = result[1][:, :n]
            residual = active["residual"][:, :n]
            added = hidden + residual
            record[label] = {
                "hidden_dtype": str(hidden.dtype),
                "residual_dtype": str(residual.dtype),
                "hidden_rms": float(hidden.float().square().mean().sqrt()),
                "residual_rms": float(residual.float().square().mean().sqrt()),
                "fraction_unchanged_after_add": float((added == hidden).float().mean()),
                "actual_addition_delta_rms": float(
                    (added.float() - hidden.float()).square().mean().sqrt()
                ),
            }

        return observe

    handles.append(model.register_forward_pre_hook(inputs, with_kwargs=True))
    handles.append(model.transformer_blocks[0].register_forward_hook(block("first_block")))
    handles.append(model.transformer_blocks[-1].register_forward_hook(block("last_block")))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def load_geometry_adapter(path: Path, model: Any, lora_sha256: str | None) -> GeometryEncoder:
    record = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if (
        record.get("format") != "roomcraft-rga-v1"
        or record.get("base") != model_spec("qwen")
        or record.get("frozen_lora_sha256") != lora_sha256
        or not lora_sha256
    ):
        raise ValueError("RGA checkpoint/base/frozen LoRA identity mismatch")
    config = RGAConfig(**record["geometry"])
    adapter = GeometryEncoder(config).to(device=next(model.parameters()).device).eval()
    state = load_file(str(path / "adapter.safetensors"))
    if set(state) != {"geometry." + k for k in adapter.state_dict()}:
        raise ValueError("RGA tensor keys do not match architecture")
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("nonfinite RGA checkpoint")
    adapter.load_state_dict({k.removeprefix("geometry."): v for k, v in state.items()}, strict=True)
    return adapter


@contextmanager
def rga_edit_context(
    model: Any,
    checkpoint: Path,
    lora_sha256: str | None,
    original: Image.Image,
    target_size: tuple[int, int],
    strength: float,
    mask_path: Path | None = None,
    depth_path: Path | None = None,
    intervention: tuple[torch.Tensor, dict[str, Any]] | None = None,
    edge_contrast: bool = False,
) -> Iterator[dict[str, Any]]:
    adapter = load_geometry_adapter(checkpoint, model, lora_sha256)
    mask = Image.open(mask_path) if mask_path else None
    try:
        depth = np.load(depth_path, allow_pickle=False) if depth_path else None
        g = (
            intervention[0]
            if intervention is not None
            else geometry_input(original, target_size, mask, depth)
        ).to(next(model.parameters()).device)
        if tuple(g.shape) != (1, 7, target_size[1], target_size[0]):
            raise ValueError("intervention geometry dimensions mismatch")
    finally:
        if mask is not None:
            mask.close()
    diagnostics: dict[str, Any] = {}
    handle = None
    if edge_contrast:
        from roomcraft.causality import validate_cues

        validate_cues(g)
        zero = g.clone()
        zero[:, 0] = 0
        with torch.inference_mode():
            zero_features = adapter(zero)
        handle = adapter.register_forward_hook(lambda _m, _a, result: result - zero_features)
    try:
        with (
            geometry_conditioning(model, adapter, g, strength),
            residual_diagnostics(model, diagnostics),
        ):
            yield {
                "checkpoint_sha256": sha256(checkpoint / "adapter.safetensors"),
                "config": json.loads((checkpoint / "config.json").read_text(encoding="utf-8")),
                "strength": strength,
                "geometry_shape": list(g.shape),
                "mask_sha256": sha256(mask_path) if mask_path else None,
                "depth_sha256": sha256(depth_path) if depth_path else None,
                "intervention": intervention[1] if intervention is not None else None,
                "edge_contrast_ablation": edge_contrast,
                "first_conditional_forward_residual_diagnostics": diagnostics,
            }
    finally:
        if handle is not None:
            handle.remove()
