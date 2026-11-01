"""Controlled cue interventions; these are not architectural-quality metrics."""

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path

ARMS = ("correct", "disabled", "zero_edges", "shifted", "unrelated", "removed", "random_removed")


def validate_cues(g: torch.Tensor) -> None:
    if g.ndim != 4 or g.shape[:2] != (1, 7) or any(x % 16 for x in g.shape[2:]):
        raise ValueError("expected [1,7,H,W] with a 16-pixel token lattice")
    if not g.is_floating_point() or not torch.isfinite(g).all() or min(g.shape[2:]) < 16:
        raise ValueError("invalid geometry values")
    if g[:, 1:5].count_nonzero():
        raise ValueError("V1.0 interventions are for the calibrated Canny/XY checkpoint only")
    if not ((g[:, 0] >= 0) & (g[:, 0] <= 1)).all():
        raise ValueError("edge channel outside [0,1]")


def intervene(
    geometry: torch.Tensor,
    arm: str,
    *,
    donor: torch.Tensor | None = None,
    region: torch.Tensor | None = None,
    shift: tuple[int, int] = (0, 32),
    seed: int = 42,
) -> torch.Tensor:
    """Change only Canny; retain source image, coordinates and absent-channel flags.

    Disabled supplies unchanged cues; caller must additionally set RGA strength zero.
    Translation uses zero padding, never wraparound. Random removal matches the
    actual count of removed edge pixels, outside the supplied structural region.
    """
    validate_cues(geometry)
    if arm not in ARMS:
        raise ValueError("unknown intervention")
    g = geometry.detach().clone()
    if arm == "zero_edges":
        g[:, 0] = 0
    elif arm == "shifted":
        dy, dx = shift
        h, w = g.shape[2:]
        if abs(dy) >= h or abs(dx) >= w:
            raise ValueError("shift outside image")
        g[:, 0] = 0
        g[:, 0, max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)] = geometry[
            :, 0, max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)
        ]
    elif arm == "unrelated":
        if donor is None or donor.shape != g.shape:
            raise ValueError("donor must use the same working lattice")
        validate_cues(donor)
        g[:, 0] = donor[:, 0]
    elif arm in ("removed", "random_removed"):
        if region is None or region.shape != g.shape[2:] or region.dtype != torch.bool:
            raise ValueError("explicit aligned boolean structural region required")
        region = region.to(g.device)
        edge = g[0, 0] > 0
        count = int((edge & region).sum())
        if count == 0:
            raise ValueError("structural region contains no source edges")
        if arm == "removed":
            g[0, 0, region] = 0
        else:
            candidates = (edge & ~region).flatten().nonzero().flatten().cpu()
            if len(candidates) < count:
                raise ValueError("insufficient nonstructural edges for matched-count control")
            chosen = candidates[
                torch.randperm(len(candidates), generator=torch.Generator().manual_seed(seed))[
                    :count
                ]
            ]
            g[0, 0].flatten()[chosen.to(g.device)] = 0
    return g


def save_intervention(path: Path, g: torch.Tensor, source: Path, record: dict[str, Any]) -> None:
    validate_cues(g)
    if path.exists() or path.with_suffix(".safetensors").exists():
        raise ValueError("intervention packages are immutable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_path = path.with_suffix(".safetensors")
    save_file({"geometry": g.detach().cpu().float().contiguous()}, str(tensor_path))
    write_json(
        path,
        {
            **record,
            "schema": "roomcraft-cue-intervention-v1",
            "source_sha256": sha256(source),
            "working_size": [g.shape[3], g.shape[2]],
            "tensor": tensor_path.name,
            "tensor_sha256": sha256(tensor_path),
        },
    )


def load_intervention(
    path: Path, source: Path, size: tuple[int, int]
) -> tuple[torch.Tensor, dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if (
        record.get("schema") != "roomcraft-cue-intervention-v1"
        or record["source_sha256"] != sha256(source)
        or record["working_size"] != list(size)
    ):
        raise ValueError("intervention source or working dimensions mismatch")
    tensor = local_path(path.parent, record["tensor"])
    if sha256(tensor) != record["tensor_sha256"]:
        raise ValueError("intervention tensor changed")
    tensors = load_file(str(tensor))
    if set(tensors) != {"geometry"}:
        raise ValueError("invalid intervention tensor keys")
    g = tensors["geometry"]
    validate_cues(g)
    if list(g.shape[2:]) != [size[1], size[0]]:
        raise ValueError("intervention lattice mismatch")
    return g, {**record, "package_sha256": sha256(path)}


def pixel_digest(image: Image.Image) -> str:
    return hashlib.sha256(str(image.size).encode() + image.convert("RGB").tobytes()).hexdigest()


def edge_contrast_features(encoder: torch.nn.Module, geometry: torch.Tensor) -> torch.Tensor:
    """Research candidate, NOT enabled in inference/training: F(edges,XY)-F(0,XY).

    Cancels the cue-independent branch and projection bias without changing saved
    weights. Both branches retain autograd. Does not imply better room geometry.
    """
    validate_cues(geometry)
    zero = geometry.clone()
    zero[:, 0] = 0
    return encoder(geometry) - encoder(zero)
