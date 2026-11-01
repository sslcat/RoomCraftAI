"""Run only on an already-approved GPU machine. Never provisions resources."""

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from roomcraft.inference import InferenceConfig, infer

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=Path("outputs/gpu-smoke"))
parser.add_argument("--size", type=int, default=512)
parser.add_argument("--steps", type=int, default=4)
parser.add_argument("--offload", action="store_true")
args = parser.parse_args()
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable: no inference attempted")
print(torch.cuda.get_device_name(), flush=True)
meta = infer(
    Path("tests/fixtures/room.png"),
    Path("tests/fixtures/mask.png"),
    "Scandinavian living room",
    args.output,
    InferenceConfig(size=args.size, steps=args.steps, offload=args.offload),
)
with Image.open(args.output / "raw.png") as image:
    pixels = np.asarray(image)
    assert pixels.std() > 1, "degenerate generated image"
assert meta["generation_seconds"] > 0
print("GPU plumbing smoke passed; a schematic fixture is not a quality benchmark.")
