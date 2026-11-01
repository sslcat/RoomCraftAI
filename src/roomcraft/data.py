"""Local JSONL manifests with explicit provenance and leakage checks."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from roomcraft.common import sha256
from roomcraft.preprocess import prepare, read_rgb


def local_path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("manifest paths must remain inside the manifest directory")
    return path


def validate_manifest(path: Path, allow_fixture: bool = False) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not rows:
        raise ValueError("empty manifest")
    ids: set[str] = set()
    scenes: dict[str, str] = {}
    content_splits: dict[str, str] = {}
    for row in rows:
        required = {
            "id",
            "scene_id",
            "split",
            "input",
            "target",
            "mask",
            "prompt",
            "license",
            "source",
            "rights_evidence",
            "commercial_use",
            "kind",
            "sha256",
        }
        if required - row.keys():
            raise ValueError(f"missing fields: {sorted(required - row.keys())}")
        if row["id"] in ids or row["split"] not in {"train", "val", "test"}:
            raise ValueError("duplicate id or invalid split")
        ids.add(row["id"])
        if row["kind"] not in {"photograph_pair", "render_pair", "test_fixture"}:
            raise ValueError("unrecognized example kind")
        if row["kind"] == "test_fixture" and not allow_fixture:
            raise ValueError(
                "test fixtures are not real training data; explicitly allow only for smoke tests"
            )
        if row["commercial_use"] is not True:
            raise ValueError("commercial rights not affirmed; quarantine this data")
        if any(
            not isinstance(row[k], str) or not row[k].strip()
            for k in ("id", "scene_id", "prompt", "license", "source", "rights_evidence")
        ):
            raise ValueError("provenance, identifiers and prompt must be nonempty strings")
        prior = scenes.setdefault(row["scene_id"], row["split"])
        if prior != row["split"]:
            raise ValueError("scene leakage across splits")
        sizes = []
        for key in ("input", "target", "mask"):
            asset = local_path(path.parent, row[key])
            digest = sha256(asset)
            if row["sha256"].get(key) != digest:
                raise ValueError(f"checksum mismatch: {row['id']}/{key}")
            with Image.open(asset) as image:
                image.verify()
            sizes.append(read_rgb(asset).size)
            if key != "mask":
                previous = content_splits.setdefault(digest, row["split"])
                if previous != row["split"]:
                    raise ValueError("image content leakage across splits")
        if len(set(sizes)) != 1:
            raise ValueError("paired assets must have identical oriented dimensions")
        _, mask = prepare(
            local_path(path.parent, row["input"]), local_path(path.parent, row["mask"]), 512
        )
        if not np.asarray(mask).any():
            raise ValueError("empty mask")
    return rows


def image_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 127.5 - 1


class RoomDataset(Dataset[dict[str, Any]]):
    def __init__(self, manifest: Path, split: str, size: int, allow_fixture: bool = False) -> None:
        self.root = manifest.parent
        self.rows = [r for r in validate_manifest(manifest, allow_fixture) if r["split"] == split]
        self.size = size
        if not self.rows:
            raise ValueError(f"no {split} examples")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        source, mask = prepare(
            local_path(self.root, row["input"]), local_path(self.root, row["mask"]), self.size
        )
        target = read_rgb(local_path(self.root, row["target"])).resize(
            source.size, Image.Resampling.LANCZOS
        )
        return {
            "source": image_tensor(source),
            "target": image_tensor(target),
            "mask": torch.from_numpy(np.asarray(mask, dtype=np.float32).copy())[None] / 255,
            "prompt": row["prompt"],
            "id": row["id"],
        }
