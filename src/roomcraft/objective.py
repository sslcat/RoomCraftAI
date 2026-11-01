"""Tensor contracts and a region-weighted denoising objective."""

import torch


def inpaint_input(
    noisy: torch.Tensor, mask: torch.Tensor, masked_latent: torch.Tensor
) -> torch.Tensor:
    if noisy.ndim != 4 or noisy.shape[1] != 4 or noisy.shape != masked_latent.shape:
        raise ValueError("latents must be matching [B,4,H/8,W/8] tensors")
    if mask.shape != (noisy.shape[0], 1, noisy.shape[2], noisy.shape[3]):
        raise ValueError("mask must be [B,1,H/8,W/8]")
    return torch.cat([noisy, mask, masked_latent], dim=1)


def region_loss(
    predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, outside_weight: float = 0.25
) -> torch.Tensor:
    if predicted.shape != target.shape or predicted.ndim != 4:
        raise ValueError("prediction/target shapes differ")
    if mask.shape != (predicted.shape[0], 1, *predicted.shape[2:]):
        raise ValueError("loss mask shape differs")
    if not 0 <= outside_weight <= 1 or bool(((mask < 0) | (mask > 1)).any()):
        raise ValueError("invalid region weight or mask")
    weights = mask.float() + outside_weight * (1 - mask.float())
    denominator = weights.sum() * predicted.shape[1]
    if denominator <= 0:
        raise ValueError("loss has zero total weight")
    return ((predicted.float() - target.float()).square() * weights).sum() / denominator
