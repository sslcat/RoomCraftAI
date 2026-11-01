"""Architecture proxies, honest comparison artifacts and blinded human score sheets."""

import csv
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roomcraft.common import write_json
from roomcraft.preprocess import read_rgb


def structure_metrics(
    source: Image.Image, output: Image.Image, region: Image.Image
) -> dict[str, Any]:
    if source.size != output.size or region.size != source.size:
        raise ValueError("evaluation images and regions must have identical dimensions")
    a, b = np.asarray(source), np.asarray(output)
    roi = np.asarray(region.convert("L")) >= 128
    if not roi.any():
        return {"pixels": 0, "mae": None, "edge_f1": None}
    # Erode the region so furniture/mask boundary edges do not pollute the ROI.
    interior = cv2.erode(roi.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    edge_a = (cv2.Canny(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), 100, 200) > 0) & interior
    edge_b = (cv2.Canny(cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), 100, 200) > 0) & interior
    kernel = np.ones((5, 5), np.uint8)
    count_a, count_b = int(edge_a.sum()), int(edge_b.sum())
    if not count_a:
        edge_f1 = None  # Empty reference edges are not evidence of perfect preservation.
    elif not count_b:
        edge_f1 = 0.0
    else:
        precision = float(
            (edge_b & (cv2.dilate(edge_a.astype(np.uint8), kernel) > 0)).sum() / count_b
        )
        recall = float((edge_a & (cv2.dilate(edge_b.astype(np.uint8), kernel) > 0)).sum() / count_a)
        edge_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "pixels": int(roi.sum()),
        "mae": float(np.abs(a.astype(float) - b)[roi].mean() / 255),
        "edge_f1": edge_f1,
        "reference_edges": count_a,
        "output_edges": count_b,
    }


def compare_runs(
    baseline: Path, candidate: Path, output: Path, regions: Path | None = None
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("comparison output must be new or empty")
    base_meta = json.loads((baseline / "metadata.json").read_text())
    candidate_meta = json.loads((candidate / "metadata.json").read_text())
    for key in (
        "input_sha256",
        "mask_sha256",
        "depth_sha256",
        "prompt",
        "negative_prompt",
        "settings",
        "base",
        "control",
        "environment",
    ):
        if base_meta[key] != candidate_meta[key]:
            raise ValueError(f"unfair comparison: {key} differs")
    if base_meta["adapter_sha256"] is not None:
        raise ValueError("baseline must use unchanged foundation weights")
    image = read_rgb(baseline / "input.png")
    with Image.open(baseline / "mask.png") as mask:
        outside = Image.fromarray(255 - np.asarray(mask.convert("L")))
    rois = {"outside_edit_mask": outside}
    if regions:
        for name in ("walls", "windows", "doors"):
            path = regions / f"{name}.png"
            if path.exists():
                with Image.open(path) as roi:
                    if roi.size != image.size:
                        raise ValueError(
                            "semantic regions must be annotated at the working resolution"
                        )
                    rois[name] = roi.convert("L")
    result: dict[str, Any] = {
        "automated": {},
        "human": "pending; fill human_scores.csv",
        "limitations": "Edge/MAE proxies do not establish semantic geometry or visual quality.",
    }
    panels = [image]
    for label, folder, meta in (
        ("foundation", baseline, base_meta),
        ("candidate", candidate, candidate_meta),
    ):
        raw, composed = read_rgb(folder / "raw.png"), read_rgb(folder / "output.png")
        panels.extend([raw, composed])
        result["automated"][label] = {
            "raw": {name: structure_metrics(image, raw, roi) for name, roi in rois.items()},
            "composited": {
                name: structure_metrics(image, composed, roi) for name, roi in rois.items()
            },
            "generation_seconds": meta["generation_seconds"],
            "total_seconds": meta["total_seconds"],
            "peak_gpu_allocated_bytes": meta["peak_gpu_allocated_bytes"],
            "peak_gpu_reserved_bytes": meta["peak_gpu_reserved_bytes"],
            "adapter_sha256": meta["adapter_sha256"],
        }
    output.mkdir(parents=True, exist_ok=True)
    image.save(output / "input.png")
    w, h = image.size
    montage = Image.new("RGB", (w * len(panels), h + 30), "white")
    draw = ImageDraw.Draw(montage)
    labels = [
        "Input",
        "Foundation raw",
        "Foundation composited",
        "Candidate raw",
        "Candidate composited",
    ]
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        montage.paste(panel, (index * w, 30))
        draw.text((index * w + 8, 8), label, fill="black")
    montage.save(output / "comparison.png")
    # Stable blinding by input hash; mapping is separate from the reviewer sheet.
    labels_order = ["foundation", "candidate"]
    if int(base_meta["input_sha256"][:2], 16) % 2:
        labels_order.reverse()
    folders = {"foundation": baseline, "candidate": candidate}
    mapping = dict(zip(["A", "B"], labels_order, strict=True))
    for blind, label in mapping.items():
        read_rgb(folders[label] / "raw.png").save(output / f"{blind}.png")
    write_json(output / "blinding_key.json", mapping)
    with (output / "human_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reviewer",
                "image",
                "quality",
                "walls",
                "windows",
                "doors",
                "style",
                "furniture",
                "notes",
            ]
        )
        writer.writerows([["", name, "", "", "", "", "", "", ""] for name in mapping])
    escaped_prompt = html.escape(base_meta["prompt"])
    (output / "review.html").write_text(
        f'<!doctype html><meta charset="utf-8"><title>RoomCraft blind review</title><h1>Blind raw-output review</h1><p>{escaped_prompt}</p><p>Left: input. Center: A. Right: B. Score quality, walls, windows, doors, style and furniture 1–5; leave absent architecture blank.</p><img src="input.png" alt="Input" style="width:32%"><img src="A.png" alt="A" style="width:32%"><img src="B.png" alt="B" style="width:32%">',
        encoding="utf-8",
    )
    write_json(output / "metrics.json", result)
    return result


def summarize_human(path: Path) -> dict[str, Any]:
    scores: dict[str, dict[str, list[float]]] = {}
    seen: set[tuple[str, str]] = set()
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (row["reviewer"], row["image"])
            if not row["reviewer"] or key in seen:
                raise ValueError("reviewers required and reviewer/image rows must be unique")
            seen.add(key)
            for field in ("quality", "walls", "windows", "doors", "style", "furniture"):
                if row[field]:
                    value = float(row[field])
                    if not 1 <= value <= 5:
                        raise ValueError("human scores must be in [1,5]")
                    scores.setdefault(row["image"], {}).setdefault(field, []).append(value)
    return {
        image: {
            field: {"mean": sum(values) / len(values), "n": len(values)}
            for field, values in fields.items()
        }
        for image, fields in scores.items()
    }
