"""Small reproducibility utilities shared by training and inference."""

import hashlib
import json
import random
from importlib import metadata, resources
from pathlib import Path
from typing import Any

import numpy as np
import torch


def model_spec(name: str) -> dict[str, str]:
    return json.loads(resources.files("roomcraft").joinpath("models.json").read_text())[name]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False


def environment() -> dict[str, Any]:
    packages = ["roomcraft-ai", "torch", "diffusers", "transformers", "peft", "numpy", "Pillow"]
    return {
        "packages": {p: metadata.version(p) for p in packages},
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
    }
