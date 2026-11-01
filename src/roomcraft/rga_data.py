"""Immutable CPU geometry sidecars; existing V0.4 latent caches are unchanged."""

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path
from roomcraft.editing import EditConfig, edit_size
from roomcraft.preprocess import read_rgb
from roomcraft.qwen_data import validate_pairs
from roomcraft.rga import CHANNELS, geometry_input


def prepare_geometry(
    manifest: Path, output: Path, size: int = 512, annotations: Path | None = None
) -> dict[str, Any]:
    EditConfig(size=size).validate()
    rows = validate_pairs(manifest)
    if output.exists():
        raise ValueError("geometry output must be new; preserve existing data")
    records = json.loads(annotations.read_text(encoding="utf-8")) if annotations else {}
    if not set(records) <= {row["id"] for row in rows}:
        raise ValueError("unknown annotation ids")
    output.mkdir(parents=True)
    result = []
    for row in rows:
        source = local_path(manifest.parent, row["source"])
        original = read_rgb(source)
        record = records.get(row["id"], {})
        paths = {
            k: local_path(annotations.parent, record[k])
            for k in ("mask", "depth", "evidence")
            if k in record and annotations
        }
        if paths and "evidence" not in paths:
            raise ValueError("external masks/depth require provenance evidence")
        copied = {}
        for key, path in paths.items():
            destination = output / "provenance" / row["id"] / (key + path.suffix)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
            copied[key] = {
                "path": destination.relative_to(output).as_posix(),
                "sha256": sha256(destination),
            }
        mask = Image.open(paths["mask"]) if "mask" in paths else None
        depth = np.load(paths["depth"], allow_pickle=False) if "depth" in paths else None
        geometry = geometry_input(original, edit_size(original, size), mask, depth)
        if mask:
            mask.close()
        file = row["id"] + ".safetensors"
        save_file({"geometry": geometry.half().contiguous()}, str(output / file))
        result.append(
            {
                "id": row["id"],
                "split": row["split"],
                "group_id": row["group_id"],
                "source_sha256": sha256(source),
                "file": file,
                "sha256": sha256(output / file),
                "dimensions": [geometry.shape[3], geometry.shape[2]],
                "annotations": copied,
                "verified_visible_architecture": (
                    record.get("verified_visible_architecture") is True
                )
                and mask is not None,
            }
        )
    record = {
        "schema": "roomcraft-rga-geometry-v1",
        "manifest_sha256": sha256(manifest),
        "size": size,
        "channels": list(CHANNELS),
        "preprocessing": "Canny100-200-L2;target-resize;depth-p2-p98;pixel-center-xy-v1",
        "annotation_manifest_sha256": sha256(annotations) if annotations else None,
        "rows": result,
    }
    write_json(output / "geometry.json", record)
    return record


def validate_geometry(path: Path, latent_cache: dict[str, Any] | None = None) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != "roomcraft-rga-geometry-v1" or record["channels"] != list(CHANNELS):
        raise ValueError("incompatible geometry schema/channels")
    EditConfig(size=record["size"]).validate()
    ids: set[str] = set()
    groups: dict[str, str] = {}
    for row in record["rows"]:
        if row["id"] in ids or row["split"] not in ("train", "validation", "test"):
            raise ValueError("duplicate/invalid geometry split")
        ids.add(row["id"])
        if groups.setdefault(row["group_id"], row["split"]) != row["split"]:
            raise ValueError("geometry scene-family leakage")
        file = local_path(path.parent, row["file"])
        if sha256(file) != row["sha256"]:
            raise ValueError("geometry tensor hash mismatch")
        tensors = load_file(str(file))
        if set(tensors) != {"geometry"}:
            raise ValueError("unexpected geometry tensors")
        g = tensors["geometry"]
        w, h = row["dimensions"]
        if (
            g.shape != (1, 7, h, w)
            or w % 16
            or h % 16
            or min(w, h) < 16
            or not torch.isfinite(g).all()
        ):
            raise ValueError("invalid geometry tensor shape/values")
        if torch.any((g[:, :5] < 0) | (g[:, :5] > 1)) or torch.any(g[:, 5:].abs() > 1):
            raise ValueError("geometry cue range invalid")
        for cue, available in ((1, 2), (3, 4)):
            flag = g[:, available]
            if not (torch.all(flag == 0) or torch.all(flag == 1)):
                raise ValueError("geometry availability flags must be constant binary maps")
            if torch.all(flag == 0) and g[:, cue].count_nonzero():
                raise ValueError("missing cue contains data")
        for evidence in row["annotations"].values():
            if sha256(local_path(path.parent, evidence["path"])) != evidence["sha256"]:
                raise ValueError("changed geometry annotation evidence")
        if type(row["verified_visible_architecture"]) is not bool:
            raise ValueError("architecture visibility review must be boolean")
        if row["verified_visible_architecture"] and (
            not {"mask", "evidence"} <= set(row["annotations"])
            or not torch.all(g[:, 2] == 1)
            or g[:, 1].sum() <= 0
        ):
            raise ValueError("verified architecture needs a nonempty mask and evidence")
    if not ids or set(groups.values()) != {"train", "validation", "test"}:
        raise ValueError("geometry requires complete splits")
    if latent_cache is not None:
        if (
            latent_cache["manifest_sha256"] != record["manifest_sha256"]
            or latent_cache["size"] != record["size"]
        ):
            raise ValueError("geometry/latent manifest or resolution mismatch")
        expected = {r["id"]: r for r in latent_cache["rows"]}
        if set(expected) != ids or any(
            any(row[k] != expected[row["id"]][k] for k in ("split", "group_id"))
            for row in record["rows"]
        ):
            raise ValueError("geometry/latent rows mismatch")
    return record
