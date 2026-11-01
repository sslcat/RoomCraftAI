"""Integrity checks and review artifacts for authored Blender pairs; never auto-approve quality."""

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path


def audit_render(folder: Path) -> dict[str, Any]:
    record = json.loads((folder / "render.json").read_text(encoding="utf-8"))
    if record["variant"]["id"] != folder.name or not record["paired_invariant_verified"]:
        raise ValueError("render identity or paired state verification failed")
    state = json.loads((folder / "invariant_state.json").read_text(encoding="utf-8"))
    import hashlib

    digest = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
    if digest != record["architecture_camera_lighting_sha256"]:
        raise ValueError("architectural state changed")
    images = []
    for key in ("source", "target"):
        path = folder / f"{key}.png"
        if sha256(path) != record["sha256"][key]:
            raise ValueError(f"changed {key} image")
        with Image.open(path) as im:
            if list(im.size) != record["settings"]["resolution"] or im.mode != "RGB":
                raise ValueError("incorrect render dimensions or color mode")
            images.append(np.asarray(im, dtype=np.float32) / 255)
    difference = float(np.abs(images[1] - images[0]).mean())
    if difference < 0.015:
        raise ValueError("pair is effectively unchanged")
    means = [float(im.mean()) for im in images]
    if any(mean < 0.07 or mean > 0.94 for mean in means):
        raise ValueError("unusable global exposure")
    return {
        "id": folder.name,
        "family": record["family"]["family"],
        "mean_absolute_rgb_change": difference,
        "mean_rgb": means,
        "sha256": record["sha256"],
        "generation_seconds": record["generation_seconds"],
        "automated_integrity": "passed",
        "visual_quality": "requires explicit review",
    }


def make_gallery(root: Path) -> list[dict[str, Any]]:
    """Create paired family sheets plus links to full-resolution images and render records."""
    gallery = root / "gallery"
    gallery.mkdir(exist_ok=True)
    audits = [audit_render(p.parent) for p in sorted((root / "pairs").glob("*/render.json"))]
    if not audits:
        raise ValueError("no completed renders")
    review_file = root / "quality_reviews.json"
    decisions = {}
    if review_file.exists():
        decisions = {
            row["id"]: row for row in json.loads(review_file.read_text(encoding="utf-8"))["pairs"]
        }
    body = [
        "<h1>RoomCraft paired-render review gallery</h1>",
        "<p>Synthetic Blender pairs. Left: empty input; right: furnished target. "
        "Integrity checks do not establish visual quality. See quality_reviews.json for decisions.</p>",
    ]
    for family in sorted({row["family"] for row in audits}):
        rows = [row for row in audits if row["family"] == family]
        sheet = Image.new("RGB", (1024, 410 * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            identifier = row["id"]
            decision = decisions.get(identifier)
            status = "pending visual review"
            if decision:
                if decision["sha256"] != row["sha256"]:
                    raise ValueError("gallery review hashes are stale")
                status = f"{decision['decision']}: {decision['notes']}"
            draw.text(
                (8, 410 * i + 5), f"{identifier} | EMPTY INPUT / FURNISHED TARGET", fill="black"
            )
            for j, key in enumerate(("source", "target")):
                with Image.open(root / "pairs" / identifier / f"{key}.png") as im:
                    sheet.paste(
                        im.resize((512, 384), Image.Resampling.LANCZOS), (512 * j, 410 * i + 24)
                    )
            body.append(
                f'<p>{html.escape(identifier)}: <a href="../pairs/{identifier}/source.png">input</a> '
                f'<a href="../pairs/{identifier}/target.png">target</a> '
                f'<a href="../pairs/{identifier}/render.json">settings</a> — {html.escape(status)}</p>'
            )
        sheet.save(gallery / f"{family}.jpg", quality=95)
        body.append(
            f'<img style="max-width:100%" src="{family}.jpg" alt="{family} paired contact sheet">'
        )
    (gallery / "index.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>RoomCraft paired review</title>'
        '<body style="font-family:system-ui;background:#eee;max-width:1050px;margin:auto">'
        + "\n".join(body)
        + "</body></html>",
        encoding="utf-8",
    )
    write_json(root / "integrity.json", {"rendered_count": len(audits), "pairs": audits})
    return audits


def reviewed_ledger(root: Path, reviews_path: Path) -> dict[str, Any]:
    """Require explicit, image-hash-bound decisions before passing to V0.4 sealing."""
    reviews = json.loads(reviews_path.read_text(encoding="utf-8"))
    if not reviews.get("reviewer") or not reviews.get("date") or not reviews.get("method"):
        raise ValueError("named dated visual review required")
    rows = []
    seen: set[str] = set()
    for review in reviews["pairs"]:
        identifier = review["id"]
        if identifier in seen:
            raise ValueError("duplicate review")
        seen.add(identifier)
        if review["decision"] not in ("accept", "reject"):
            raise ValueError("explicit accept/reject required")
        folder = local_path(root, f"pairs/{identifier}")
        audit = audit_render(folder)
        if audit["sha256"] != review["sha256"] or not review.get("notes"):
            raise ValueError("review does not bind to current images or lacks notes")
        if review["decision"] == "reject":
            continue
        record = json.loads((folder / "render.json").read_text(encoding="utf-8"))
        evidence = json.loads((root / "provenance/rights.json").read_text(encoding="utf-8"))
        for relative, expected_hash in evidence.get("evidence_files", {}).items():
            if sha256(local_path(root, relative)) != expected_hash:
                raise ValueError("rights evidence file changed")
        if evidence["asset_provenance_sha256"] != record["asset_provenance_sha256"]:
            raise ValueError("render asset provenance differs from rights evidence")
        rows.append(
            {
                "id": identifier,
                "scene_id": identifier,
                "group_id": audit["family"],
                "kind": "render_pair",
                "source": f"pairs/{identifier}/source.png",
                "target": f"pairs/{identifier}/target.png",
                "instruction": record["instruction"],
                "creator": evidence.get(
                    "creator",
                    "RoomCraft procedural scene generator; asset authors listed in provenance",
                ),
                "source_url": f"{evidence.get('source_namespace', 'urn:roomcraft:procedural:v05')}:{identifier}",
                "license": evidence.get(
                    "license", "MIT (original contribution); CC0-1.0 (listed assets)"
                ),
                "license_url": evidence.get("license_url", "https://opensource.org/license/mit"),
                "evidence": "provenance/rights.json",
                "alignment_evidence": f"pairs/{identifier}/render.json",
                "rights": evidence["rights"],
                "dimensions": record["settings"]["resolution"],
                "review": {
                    "reviewer": reviews["reviewer"],
                    "date": reviews["date"],
                    "method": reviews["method"],
                    "notes": review["notes"],
                    "same_camera": True,
                    "architecture_unchanged": True,
                    "source_empty": True,
                    "target_furnished": True,
                },
            }
        )
    if not rows:
        raise ValueError("no visually approved pairs")
    return {"schema": "roomcraft-pairs-v1", "pairs": rows, "review_sha256": sha256(reviews_path)}
