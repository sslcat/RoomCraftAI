"""Project-owned paired rectified-flow objective and Qwen attention LoRA injection."""

from typing import Any

import torch
from torch import nn

from roomcraft.lora import LoRALinear

QWEN_TARGETS = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)


def inject_qwen_lora(model: Any, rank: int, alpha: float) -> list[str]:
    if not getattr(model.config, "zero_cond_t", False):
        raise ValueError("Qwen 2511 requires zero_cond_t=True")
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and ".attn." in name and name.endswith(QWEN_TARGETS)
    ]
    if len(targets) != 8 * model.config.num_layers:
        raise ValueError("expected eight attention projections in every Qwen block")
    # The last block's text output is discarded. Its projection has no path to image loss.
    targets.remove(f"transformer_blocks.{model.config.num_layers - 1}.attn.to_add_out")
    model.requires_grad_(False)
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha))
    return targets


def validate_tensors(tensors: dict[str, torch.Tensor], text_width: int = 3584) -> None:
    if set(tensors) != {"target", "source", "text", "text_mask", "shapes"}:
        raise ValueError("unexpected conditioning tensor keys")
    target, source, text = (tensors[k] for k in ("target", "source", "text"))
    for latent in (target, source):
        if latent.ndim != 3 or latent.shape[0] != 1 or latent.shape[2] != 64 or latent.shape[1] < 1:
            raise ValueError("packed latents must be [1,N,64]")
    if text.ndim != 3 or text.shape[0] != 1 or text.shape[2] != text_width or text.shape[1] < 1:
        raise ValueError("text must be [1,L,joint_attention_dim]")
    mask, shapes = tensors["text_mask"], tensors["shapes"]
    if mask.shape != text.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError("invalid text padding mask")
    if (
        shapes.shape != (2, 3)
        or shapes.dtype != torch.int64
        or not torch.all(shapes > 0)
        or not torch.all(shapes[:, 0] == 1)
    ):
        raise ValueError("shapes must contain two [1,H/16,W/16] entries")
    if int(shapes[0].prod()) != target.shape[1] or int(shapes[1].prod()) != source.shape[1]:
        raise ValueError("image grid and token count mismatch")
    for tensor in (target, source, text):
        if not tensor.is_floating_point() or not torch.isfinite(tensor).all():
            raise ValueError("nonfinite or nonfloating conditioning")


def flow_path(
    target: torch.Tensor, noise: torch.Tensor, sigma: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.shape != noise.shape or sigma.shape != (target.shape[0],):
        raise ValueError("flow shape mismatch")
    if not torch.isfinite(sigma).all() or torch.any((sigma < 0) | (sigma > 1)):
        raise ValueError("sigma outside [0,1]")
    s = sigma[:, None, None]
    return (1 - s) * target + s * noise, noise - target


def target_flow_loss(prediction: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
    if (
        prediction.ndim != 3
        or prediction.shape[0] != velocity.shape[0]
        or prediction.shape[2] != velocity.shape[2]
        or prediction.shape[1] < velocity.shape[1]
    ):
        raise ValueError("prediction cannot cover target tokens")
    # Reference predictions receive no direct reconstruction loss.
    return (prediction[:, : velocity.shape[1]].float() - velocity.float()).square().mean()


def paired_loss(model: Any, tensors: dict[str, torch.Tensor], seed: int) -> torch.Tensor:
    target = tensors["target"]
    generator = torch.Generator(device=target.device).manual_seed(seed)
    sigma = torch.rand((1,), device=target.device, generator=generator, dtype=torch.float32)
    noise = torch.randn(
        target.shape, device=target.device, generator=generator, dtype=torch.float32
    )
    noisy, velocity = flow_path(target.float(), noise, sigma)
    prediction = model(
        hidden_states=torch.cat([noisy.to(target.dtype), tensors["source"]], dim=1),
        timestep=sigma,
        encoder_hidden_states=tensors["text"],
        encoder_hidden_states_mask=tensors["text_mask"],
        img_shapes=[[tuple(s) for s in tensors["shapes"].tolist()]],
        return_dict=False,
    )[0]
    return target_flow_loss(prediction, velocity)


def gradient_report(model: nn.Module) -> dict[str, float | int]:
    squared = 0.0
    present = 0
    for name, param in model.named_parameters():
        if not param.requires_grad and param.grad is not None:
            raise RuntimeError(f"frozen base has a gradient: {name}")
        if param.requires_grad:
            if param.grad is None or not torch.isfinite(param.grad).all():
                raise RuntimeError(f"missing/nonfinite adapter gradient: {name}")
            squared += float(param.grad.detach().float().square().sum())
            present += 1
    if not present or squared == 0:
        raise RuntimeError("no learning signal reached the adapter")
    return {"gradient_norm": squared**0.5, "gradient_tensors": present}
