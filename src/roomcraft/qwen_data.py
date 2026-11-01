"""Rights-reviewed paired edit manifests. No pairing is inferred from filenames."""

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path
from roomcraft.preprocess import read_rgb


def validate_pairs(path: Path, require_splits: bool = True) -> list[dict[str, Any]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != "roomcraft-pairs-v1":
        raise ValueError("expected roomcraft-pairs-v1")
    rows = document["pairs"]
    if not rows:
        raise ValueError("no approved pairs; unpaired evaluation photos cannot train an editor")
    ids: set[str] = set()
    groups: dict[str, str] = {}
    content: dict[str, str] = {}
    scenes: dict[str, str] = {}
    for row in rows:
        identifier = row["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in ids:
            raise ValueError("invalid or duplicate pair id")
        ids.add(identifier)
        for key in (
            "scene_id",
            "group_id",
            "instruction",
            "creator",
            "source_url",
            "license",
            "license_url",
        ):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"missing provenance/instruction: {key}")
        if row["kind"] not in ("photograph_pair", "render_pair"):
            raise ValueError("only documented photographs or renders; no schematic fixtures")
        rights = row["rights"]
        for key in ("commercial_training", "derivatives", "adapter_distribution"):
            if rights.get(key) is not True:
                raise ValueError(f"rights not cleared: {key}")
        if not isinstance(rights.get("image_redistribution"), bool):
            raise ValueError("record image redistribution separately from model distribution")
        for key in ("evidence", "alignment_evidence"):
            evidence = local_path(path.parent, row[key])
            if not evidence.is_file() or sha256(evidence) != row["sha256"][key]:
                raise ValueError(f"missing or changed evidence: {key}")
        review = row["review"]
        if not review.get("reviewer") or not review.get("date"):
            raise ValueError("a named alignment/rights review is required")
        for key in ("same_camera", "architecture_unchanged", "source_empty", "target_furnished"):
            if review.get(key) is not True:
                raise ValueError(f"pair review not passed: {key}")
        split = row.get("split", "unassigned")
        if require_splits and split not in ("train", "validation", "test"):
            raise ValueError("assign train/validation/test before caching")
        group = row["group_id"]
        if group in groups and groups[group] != split:
            raise ValueError("property/scene-family leakage across splits")
        groups[group] = split
        scene = row["scene_id"]
        if scene in scenes and scenes[scene] != group:
            raise ValueError("one scene cannot belong to multiple groups")
        scenes[scene] = group
        sizes = []
        pixels = []
        for key in ("source", "target"):
            image_path = local_path(path.parent, row[key])
            if sha256(image_path) != row["sha256"][key]:
                raise ValueError(f"image hash mismatch: {key}")
            image = read_rgb(image_path)
            sizes.append(image.size)
            digest = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            pixels.append(digest)
            if digest in content and content[digest] != group:
                raise ValueError(
                    "duplicate decoded image across groups; merge groups before splitting"
                )
            content[digest] = group
        if sizes[0] != sizes[1] or pixels[0] == pixels[1]:
            raise ValueError("pair must have matching dimensions and distinct source/target")
    if require_splits and {r["split"] for r in rows} != {"train", "validation", "test"}:
        raise ValueError("all three nonempty splits are required")
    return list(rows)


def split_pairs(path: Path, output: Path, seed: int = 42) -> dict[str, Any]:
    """Stable property-level 80/10/10 allocation, at least one group per split."""
    if output.exists() or output.parent.resolve() != path.parent.resolve():
        raise ValueError("write a new manifest beside the input so relative paths stay valid")
    rows = validate_pairs(path, require_splits=False)
    groups = sorted({r["group_id"] for r in rows})
    if len(groups) < 3:
        raise ValueError("need at least three independent property/scene groups")
    random.Random(seed).shuffle(groups)
    held = max(1, round(len(groups) * 0.1))
    assignment = {
        g: "test" if i < held else "validation" if i < held * 2 else "train"
        for i, g in enumerate(groups)
    }
    for row in rows:
        row["split"] = assignment[row["group_id"]]
    result = {
        "schema": "roomcraft-pairs-v1",
        "split_seed": seed,
        "parent_sha256": sha256(path),
        "pairs": rows,
    }
    write_json(output, result)
    validate_pairs(output)
    return result
