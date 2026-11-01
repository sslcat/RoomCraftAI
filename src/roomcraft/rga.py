"""RoomCraft geometry encoder and target-only Qwen block-residual conditioning.

Original research implementation; not a trained or quality-validated adapter.
"""

import inspect
import math
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from roomcraft.qwen_objective import flow_path, target_flow_loss, validate_tensors

CHANNELS = ("canny", "protected_mask", "mask_available", "depth", "depth_available", "x", "y")


@dataclass(frozen=True)
class RGAConfig:
    width: int = 32
    hidden_dim: int = 3072
    strength: float = 1.0
    residual_gain: float = 1.0

    def validate(self) -> None:
        if self.width < 8 or self.width % 8 or self.hidden_dim < 1:
            raise ValueError("RGA width must be a positive multiple of 8; hidden_dim positive")
        if not math.isfinite(self.strength) or not 0 <= self.strength <= 4:
            raise ValueError("RGA strength must be finite in [0,4]")
        if not math.isfinite(self.residual_gain) or not 0 < self.residual_gain <= 1e9:
            raise ValueError("RGA residual gain must be finite in (0,1e9]")


def geometry_input(
    image: Image.Image,
    size: tuple[int, int],
    mask: Image.Image | None = None,
    depth: np.ndarray | None = None,
) -> torch.Tensor:
    """Source-only deterministic cues [1,7,H,W]; no depth/semantic model is inferred."""
    w, h = size
    if min(w, h) < 16 or w % 16 or h % 16:
        raise ValueError("geometry dimensions must be multiples of 16")
    pixels = np.asarray(image.convert("RGB").resize(size, Image.Resampling.LANCZOS))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    edge = cv2.Canny(gray, 100, 200, L2gradient=True).astype(np.float32) / 255
    protected = np.zeros((h, w), dtype=np.float32)
    if mask is not None:
        if mask.size != image.size:
            raise ValueError("mask must align with original source pixels")
        raw = np.asarray(mask.convert("L"))
        if not np.isin(raw, [0, 255]).all():
            raise ValueError("protected mask must be binary 0/255")
        protected[:] = np.asarray(mask.convert("L").resize(size, Image.Resampling.NEAREST)).astype(
            np.float32
        ) / np.float32(255)
    relative_depth = np.zeros((h, w), dtype=np.float32)
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        if depth.shape != (image.height, image.width) or not np.isfinite(depth).all():
            raise ValueError("depth must be finite and align with original source pixels")
        lo, hi = np.percentile(depth, [2, 98])
        normalized = np.clip((depth - lo) / max(float(hi - lo), 1e-6), 0, 1)
        relative_depth = cv2.resize(normalized, size, interpolation=cv2.INTER_LINEAR)
    x = np.broadcast_to(2 * (np.arange(w, dtype=np.float32) + 0.5) / w - 1, (h, w))
    y = np.broadcast_to((2 * (np.arange(h, dtype=np.float32) + 0.5) / h - 1)[:, None], (h, w))
    return torch.from_numpy(
        np.stack(
            [
                edge,
                protected,
                np.full((h, w), mask is not None, dtype=np.float32),
                relative_depth,
                np.full((h, w), depth is not None, dtype=np.float32),
                x,
                y,
            ]
        )
    ).unsqueeze(0)


def architecture_weights(geometry: torch.Tensor) -> torch.Tensor:
    """Average mask occupancy per 16x16 packed target token: [1,Nt,1]."""
    return F.avg_pool2d(geometry[:, 1:2], 16, 16).flatten(2).transpose(1, 2)


class GeometryEncoder(nn.Module):
    """Four stride-2 Conv4/GroupNorm/SiLU stages, then a zero output projection."""

    def __init__(self, config: RGAConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        layers: list[nn.Module] = []
        channels = 7
        for output in (config.width, 2 * config.width, 3 * config.width, 4 * config.width):
            layers.extend(
                [nn.Conv2d(channels, output, 4, 2, 1), nn.GroupNorm(8, output), nn.SiLU()]
            )
            channels = output
        self.encoder = nn.Sequential(*layers)
        self.projection = nn.Conv2d(channels, config.hidden_dim, 1)
        nn.init.zeros_(self.projection.weight)
        assert self.projection.bias is not None
        nn.init.zeros_(self.projection.bias)

    def forward(self, geometry: torch.Tensor) -> torch.Tensor:
        if (
            geometry.ndim != 4
            or geometry.shape[:2] != (1, 7)
            or min(geometry.shape[2:]) < 16
            or any(d % 16 for d in geometry.shape[2:])
            or not geometry.is_floating_point()
            or not torch.isfinite(geometry).all()
        ):
            raise ValueError("geometry must be finite [1,7,H,W], H/W multiples of 16")
        features = self.projection(self.encoder(geometry))
        # Spatial row-major order matches Qwen's packed latent token lattice.
        return features.flatten(2).transpose(1, 2)


def check_transformer(model: Any, config: RGAConfig) -> None:
    if (
        not model.config.zero_cond_t
        or model.config.in_channels != 64
        or model.inner_dim != config.hidden_dim
        or model.config.num_layers < 1
        or "controlnet_block_samples" not in inspect.signature(model.forward).parameters
    ):
        raise ValueError("incompatible Qwen transformer/residual interface")


def token_residual(
    features: torch.Tensor,
    hidden: torch.Tensor,
    shapes: Any,
    layers: int,
    strength: float,
    grid: tuple[int, int],
    gain: float = 1.0,
) -> torch.Tensor:
    if (
        hidden.ndim != 3
        or hidden.shape[0] != 1
        or hidden.shape[2] != 64
        or not isinstance(shapes, list)
        or len(shapes) != 1
        or len(shapes[0]) != 2
    ):
        raise ValueError("RGA v1 supports one target plus one reference, batch one")
    target, reference = shapes[0]
    if (
        len(target) != 3
        or len(reference) != 3
        or target[0] != 1
        or reference[0] != 1
        or min(*target, *reference) < 1
    ):
        raise ValueError("invalid Qwen image grids")
    nt, nr = math.prod(target), math.prod(reference)
    if tuple(target[1:]) != grid or features.shape[:2] != (1, nt) or hidden.shape[1] != nt + nr:
        raise ValueError("geometry target lattice or reference token count mismatch")
    # Upstream repeats a one-element residual list after EVERY block. Normalize by L.
    target_features = features.to(device=hidden.device, dtype=hidden.dtype) * (
        strength * gain / layers
    )
    return torch.cat([target_features, target_features.new_zeros(1, nr, features.shape[-1])], dim=1)


class GeometryConditionedQwen(nn.Module):
    """Train RGA only; pretrained Qwen and the existing LoRA are both frozen."""

    def __init__(self, transformer: Any, config: RGAConfig) -> None:
        super().__init__()
        check_transformer(transformer, config)
        self.transformer = transformer.requires_grad_(False).eval()
        self.geometry = GeometryEncoder(config)

    def train(self, mode: bool = True) -> "GeometryConditionedQwen":
        super().train(mode)
        self.transformer.eval()
        return self

    def forward(
        self, *, geometry: torch.Tensor, strength: float | None = None, **kwargs: Any
    ) -> Any:
        if kwargs.get("controlnet_block_samples") is not None:
            raise ValueError("external ControlNet residuals cannot be combined silently")
        scale = self.geometry.config.strength if strength is None else strength
        if not math.isfinite(scale) or not 0 <= scale <= 4:
            raise ValueError("invalid geometry strength")
        if scale:
            features = self.geometry(geometry)
            kwargs["controlnet_block_samples"] = [
                token_residual(
                    features,
                    kwargs["hidden_states"],
                    kwargs["img_shapes"],
                    self.transformer.config.num_layers,
                    scale,
                    (geometry.shape[2] // 16, geometry.shape[3] // 16),
                    self.geometry.config.residual_gain,
                )
            ]
        return self.transformer(**kwargs)


@contextmanager
def geometry_conditioning(
    model: Any, adapter: GeometryEncoder, geometry: torch.Tensor, strength: float = 1.0
) -> Iterator[None]:
    """Inference-only context; original transformer object/config/cache_context stay intact.

    Hook is on the outer transformer, not checkpoint-recomputed blocks. The same
    spatial residual is supplied to conditional and negative-prompt CFG calls.
    """
    check_transformer(model, adapter.config)
    if not math.isfinite(strength) or not 0 <= strength <= 4:
        raise ValueError("invalid geometry strength")
    if torch.is_grad_enabled():
        raise RuntimeError(
            "geometry_conditioning is inference-only; use GeometryConditionedQwen for training"
        )
    if getattr(model, "_roomcraft_rga_active", False):
        raise RuntimeError("nested/concurrent RGA contexts are unsupported")
    if not strength:
        yield
        return
    features = adapter(geometry)

    def inject(module: Any, args: Any, kwargs: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
        if args or kwargs.get("controlnet_block_samples") is not None:
            raise ValueError("RGA requires keyword model inputs and exclusive residual control")
        kwargs["controlnet_block_samples"] = [
            token_residual(
                features,
                kwargs["hidden_states"],
                kwargs["img_shapes"],
                model.config.num_layers,
                strength,
                (geometry.shape[2] // 16, geometry.shape[3] // 16),
                adapter.config.residual_gain,
            )
        ]
        return args, kwargs

    model._roomcraft_rga_active = True
    handle = model.register_forward_pre_hook(inject, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()
        del model._roomcraft_rga_active


def masked_velocity_loss(
    prediction: torch.Tensor, velocity: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """Paired-target weighted flow error, NOT an edge-image or decoded geometry loss."""
    if (
        weights.shape != (*velocity.shape[:2], 1)
        or not torch.isfinite(weights).all()
        or torch.any((weights < 0) | (weights > 1))
        or weights.sum() <= 0
    ):
        raise ValueError("architecture weights need nonempty finite target occupancy in [0,1]")
    error = prediction[:, : velocity.shape[1]].float() - velocity.float()
    return (weights.float() * error.square()).sum() / (weights.sum() * velocity.shape[-1])


def rga_loss(
    model: GeometryConditionedQwen,
    tensors: dict[str, torch.Tensor],
    geometry: torch.Tensor,
    seed: int,
    architecture_lambda: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    validate_tensors(tensors, text_width=model.transformer.config.joint_attention_dim)
    if not math.isfinite(architecture_lambda) or architecture_lambda < 0:
        raise ValueError("invalid architecture loss weight")
    target = tensors["target"]
    generator = torch.Generator(device=target.device).manual_seed(seed)
    sigma = torch.rand((1,), device=target.device, generator=generator)
    noise = torch.randn(
        target.shape, device=target.device, generator=generator, dtype=torch.float32
    )
    noisy, velocity = flow_path(target.float(), noise, sigma)
    prediction = model(
        geometry=geometry,
        hidden_states=torch.cat([noisy.to(target.dtype), tensors["source"]], dim=1),
        timestep=sigma,
        encoder_hidden_states=tensors["text"],
        encoder_hidden_states_mask=tensors["text_mask"],
        img_shapes=[[tuple(s) for s in tensors["shapes"].tolist()]],
        return_dict=False,
    )[0]
    flow = target_flow_loss(prediction, velocity)
    auxiliary = (
        masked_velocity_loss(prediction, velocity, architecture_weights(geometry))
        if architecture_lambda
        else flow.new_zeros(())
    )
    loss = flow + architecture_lambda * auxiliary
    return loss, {"flow": float(flow.detach()), "architecture": float(auxiliary.detach())}


def rga_record(config: RGAConfig) -> dict[str, Any]:
    return {
        "format": "roomcraft-rga-v1",
        "geometry": asdict(config),
        "channels": list(CHANNELS),
        "injection": "target-only-shared-block-residual-divide-L-v1",
    }
