"""Original small LoRA implementation for SDXL attention projections.

Adapters use a project format, deliberately not labeled PEFT-compatible.
"""

import math
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 8.0) -> None:
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("rank and alpha must be positive")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.a, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original = self.base(x)
        # FP32 master adapter parameters; autocast may choose lower precision matmuls.
        delta = (x.to(self.a.dtype) @ self.a.T) @ self.b.T
        return original + (self.scale * delta).to(original.dtype)


def inject_lora(model: nn.Module, rank: int, alpha: float) -> list[str]:
    model.requires_grad_(False)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and (name.endswith(("to_q", "to_k", "to_v", "to_out.0")))
    ]
    if not targets:
        raise ValueError("no SDXL attention projections found")
    for name in targets:
        parent_name, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha))
    return targets


def adapter_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: p.detach().cpu().contiguous()
        for name, p in model.named_parameters()
        if p.requires_grad
    }


def save_adapter(model: nn.Module, path: Path) -> None:
    save_file(adapter_state(model), str(path))


def load_adapter(model: nn.Module, path: Path) -> None:
    state = load_file(str(path))
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError("adapter keys do not match architecture")
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in state:
                if state[name].shape != parameter.shape:
                    raise ValueError(f"adapter shape mismatch: {name}")
                parameter.copy_(state[name])


def parameter_report(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "fraction": trainable / total,
        "tensors": {n: list(p.shape) for n, p in model.named_parameters() if p.requires_grad},
    }
