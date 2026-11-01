"""Matched Qwen baseline/adapter comparisons with explicit missing human judgments."""

import csv
import html
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from roomcraft.common import sha256, write_json
from roomcraft.evaluation import structure_metrics
from roomcraft.preprocess import read_rgb


def compare_qwen(
    baseline: Path, candidate: Path, output: Path, regions: Path | None = None
) -> dict[str, Any]:
    a, b = [
        json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        for folder in (baseline, candidate)
    ]
    for key in (
        "backend",
        "base",
        "input_sha256",
        "prompt",
        "settings",
        "working_size",
        "dtype",
        "quantization",
        "environment",
        "scheduler_config",
    ):
        if a[key] != b[key]:
            raise ValueError(f"unfair Qwen comparison: {key}")
    if a["adapter_sha256"] is not None or b["adapter_sha256"] is None:
        raise ValueError("need unchanged baseline and a trained candidate adapter")
    if output.exists():
        raise ValueError("comparison output must be new")
    source = read_rgb(baseline / "input.png")
    panels = [source]
    metrics: dict[str, Any] = {
        "human_review": "pending",
        "quality_improvement": None,
        "automated": {},
        "warning": "Edge agreement/MAE are regional pixel proxies, not realism or semantic architecture scores.",
    }
    for label, folder, meta in (("baseline", baseline, a), ("adapter", candidate, b)):
        if sha256(folder / "output.png") != meta["output_sha256"]:
            raise ValueError("changed generation artifact")
        panel = read_rgb(folder / "output.png")
        if panel.size != source.size:
            raise ValueError("comparison dimension mismatch")
        panels.append(panel)
        regional = {}
        if regions:
            for name in ("walls", "windows", "doors", "floor"):
                path = regions / f"{name}.png"
                if path.exists():
                    with Image.open(path) as mask:
                        regional[name] = structure_metrics(source, panel, mask)
        metrics["automated"][label] = {
            "regions": regional,
            **{
                k: meta[k]
                for k in (
                    "generation_seconds",
                    "peak_gpu_allocated_bytes",
                    "peak_gpu_reserved_bytes",
                    "adapter_sha256",
                )
            },
        }
    output.mkdir(parents=True)
    w, h = source.size
    montage = Image.new("RGB", (3 * w, h + 32), "white")
    draw = ImageDraw.Draw(montage)
    for i, (panel, label) in enumerate(
        zip(panels, ("Input", "Unchanged Qwen", "RoomCraft adapter"), strict=True)
    ):
        montage.paste(panel, (i * w, 32))
        draw.text((i * w + 5, 8), label, fill="black")
    montage.save(output / "comparison.png")
    order = (1, 2) if int(a["input_sha256"][:2], 16) % 2 == 0 else (2, 1)
    source.save(output / "input.png")
    for blind, i in zip(("A", "B"), order, strict=True):
        panels[i].save(output / f"{blind}.png")
    write_json(
        output / "blinding_key.json",
        {
            blind: "baseline" if i == 1 else "adapter"
            for blind, i in zip(("A", "B"), order, strict=True)
        },
    )
    with (output / "human_review.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reviewer",
                "image",
                "furniture_realism",
                "instruction",
                "walls",
                "windows",
                "doors",
                "floor_perspective",
                "lighting_shadows",
                "artifact_absence",
                "notes",
            ]
        )
        writer.writerows([["", label] + [""] * 9 for label in ("A", "B")])
    (output / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Qwen blind review</title><p>'
        + html.escape(a["prompt"])
        + "</p><p>Input / A / B. Score each criterion 1 (poor) to 5 (strong); leave absent features blank. Subjective judgments, not benchmark scores.</p>"
        + "".join(
            f'<img src="{name}.png" style="width:32%" alt="{name}">' for name in ("input", "A", "B")
        ),
        encoding="utf-8",
    )
    write_json(output / "metrics.json", metrics)
    return metrics
