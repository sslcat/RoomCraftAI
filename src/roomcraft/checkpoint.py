"""Atomic checkpoint directories with restricted PyTorch state loading."""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from roomcraft.common import write_json
from roomcraft.lora import load_adapter, save_adapter


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    config: dict[str, Any],
) -> None:
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise ValueError("checkpoint already exists; choose a new run directory")
    temporary.mkdir(parents=True)
    save_adapter(model, temporary / "adapter.safetensors")
    write_json(temporary / "config.json", config)
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(), "step": step},
        temporary / "state.pt",
    )
    temporary.rename(path)


def resume_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    expected: dict[str, Any],
) -> int:
    saved = json.loads((path / "config.json").read_text())
    # Steps may increase; every other training/data parameter must stay fixed.
    for key, value in expected.items():
        if key != "steps" and saved.get(key) != value:
            raise ValueError(f"resume configuration mismatch: {key}")
    load_adapter(model, path / "adapter.safetensors")
    state = torch.load(path / "state.pt", map_location="cpu", weights_only=True)
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    return int(state["step"])
