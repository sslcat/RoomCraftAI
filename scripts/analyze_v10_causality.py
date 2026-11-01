"""Credited native-output galleries and explicitly nonsemantic edge/pixel proxies."""

import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roomcraft.common import sha256, write_json
from roomcraft.evaluation import structure_metrics

ROOT = Path(__file__).resolve().parents[1]
ARMS = (
    "correct",
    "disabled",
    "zero_edges",
    "shifted",
    "unrelated",
    "removed",
    "random_removed",
    "edge_contrast_untrained",
)


def window_translation(reference, target, box):
    """Translation of the visible upper-window edge pattern, not semantic landmarks."""

    def edges(image):
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))

    x0, y0, x1, y1 = box
    sx0, sy0 = max(0, x0 - 80), max(0, y0 - 24)
    sx1, sy1 = min(target.width, x1 + 80), min(target.height, y1 + 24)
    template = edges(reference)[y0:y1, x0:x1]
    search = edges(target)[sy0:sy1, sx0:sx1]
    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, location = cv2.minMaxLoc(scores)
    return {
        "dx_pixels": location[0] + sx0 - x0,
        "dy_pixels": location[1] + sy0 - y0,
        "edge_ncc": peak,
        "reference_box_xyxy": list(box),
        "search_radius_xy": [80, 24],
        "interpretation": "Translation-only upper-window edge registration to correct arm; not semantic landmark error. Review image visibility, shape changes and ambiguous matches.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "outputs/v10-causality-full")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v10-causality-review")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((ROOT / "configs/v07_evaluation.json").read_text())
    records = []
    body = [
        "<h1>V1.0 geometry causality: actual native outputs</h1><p>Matched source, prompt, seed 42, 40 steps, CFG 4, native long-side 1024. Edge/pixel measurements are proxies, not architectural causality or realism scores. Edge-contrast is an untrained inference ablation with unchanged weights and gain. Outputs are AI edits; source display may be resized. Click images for native pixels.</p><p><a href='measurements.json'>Measurements and residual diagnostics</a>; <a href='../../experiments/v10_cloud_visual_review.json'>Qualitative visual review</a>; <a href='../../ROOMCRAFT_V10_CAUSALITY_REPORT.md'>Execution report</a>.</p>"
    ]
    for case in protocol["cases"]:
        case_root = args.runs / case["id"]
        available = [arm for arm in ARMS if (case_root / arm / "metadata.json").exists()]
        if not available:
            continue
        folder = args.output / case["id"]
        folder.mkdir(exist_ok=True)
        first = json.loads((case_root / available[0] / "metadata.json").read_text())
        source = (
            Image.open(ROOT / case["source"])
            .convert("RGB")
            .resize(tuple(first["working_size"]), Image.Resampling.LANCZOS)
        )
        source.save(folder / "source.png")
        images = {"source": source}
        credits = case["provenance"]
        body.append(
            f"<h2>{html.escape(case['id'])} ({case['domain']})</h2><p>{html.escape(credits['author'])}; <a href='{html.escape(credits['license_url'], quote=True)}'>{html.escape(credits['license'])}</a>; <a href='{html.escape(credits['source_url'], quote=True)}'>source</a>. {html.escape(case['instruction'])}</p>"
        )
        region = None
        crop = None
        upper_window = None
        if case["domain"] == "synthetic":
            landmarks = json.loads(
                (ROOT / "outputs/v10-causality/landmarks" / (case["id"] + ".json")).read_text()
            )
            line = next(x for x in landmarks["polylines"] if x["name"] == "back_window")
            xy = [(x * source.width, y * source.height) for x, y in line["normalized_xy"]]
            region = Image.new("L", source.size)
            ImageDraw.Draw(region).line(xy, fill=255, width=24)
            region.save(folder / "window-boundary-region.png")
            crop = (
                max(0, int(min(x for x, y in xy)) - 40),
                max(0, int(min(y for x, y in xy)) - 35),
                min(source.width, int(max(x for x, y in xy)) + 40),
                min(source.height, int(max(y for x, y in xy)) + 35),
            )
            upper_window = (
                max(0, int(min(x for x, y in xy)) - 12),
                max(0, int(min(y for x, y in xy)) - 12),
                min(source.width, int(max(x for x, y in xy)) + 12),
                int((min(y for x, y in xy) + max(y for x, y in xy)) / 2),
            )
        for arm in available:
            location = case_root / arm
            meta = json.loads((location / "metadata.json").read_text())
            for key in (
                "input_sha256",
                "prompt",
                "settings",
                "base",
                "working_size",
                "adapter_sha256",
            ):
                if meta[key] != first[key]:
                    raise ValueError(f"Unmatched {case['id']}/{arm}: {key}")
            if sha256(location / "output.png") != meta["output_sha256"]:
                raise ValueError("output hash mismatch")
            image = Image.open(location / "output.png").convert("RGB")
            if image.size != source.size:
                raise ValueError("not native matched dimensions")
            images[arm] = image
            item = {
                "case": case["id"],
                "domain": case["domain"],
                "arm": arm,
                "output_sha256": meta["output_sha256"],
                "generation_seconds": meta["generation_seconds"],
                "total_seconds": meta["total_seconds"],
                "peak_allocated_gib": meta["peak_gpu_allocated_bytes"] / 2**30,
                "residual": meta["geometry_adapter"][
                    "first_conditional_forward_residual_diagnostics"
                ],
            }
            if region:
                item["back_window_edge_proxy_not_semantic_score"] = structure_metrics(
                    source, image, region
                )
            if upper_window and "correct" in images:
                item["upper_window_translation_proxy_vs_correct"] = window_translation(
                    images["correct"], image, upper_window
                )
            for ref in ("correct", "disabled"):
                if (case_root / ref / "output.png").exists():
                    other = np.asarray(
                        Image.open(case_root / ref / "output.png").convert("RGB"), dtype=np.int16
                    )
                    delta = np.asarray(image, dtype=np.int16) - other
                    item["vs_" + ref] = {
                        "pixels_exact": bool(not delta.any()),
                        "mae_8bit": float(np.abs(delta).mean()),
                    }
            records.append(item)
        names = list(images)
        sheet = Image.new("RGB", (1536, 412 * ((len(names) + 2) // 3)), "white")
        draw = ImageDraw.Draw(sheet)
        body.append("<div style='display:grid;grid-template-columns:repeat(3,1fr);gap:10px'>")
        for i, name in enumerate(names):
            im = images[name]
            thumb = im.copy()
            thumb.thumbnail((512, 384))
            x, y = i % 3 * 512, i // 3 * 412
            draw.text((x + 5, y + 5), name, fill="black")
            sheet.paste(thumb, (x, y + 24))
            link = (
                f"{case['id']}/source.png"
                if name == "source"
                else Path(os.path.relpath(case_root / name / "output.png", args.output)).as_posix()
            )
            body.append(
                f"<figure style='margin:0'><figcaption>{name}</figcaption><a href='{link}'><img style='width:100%' src='{link}'></a></figure>"
            )
        body.append("</div>")
        sheet.save(folder / "comparison.jpg", quality=96)
        if crop:
            width, height = crop[2] - crop[0], crop[3] - crop[1]
            strip = Image.new(
                "RGB",
                (width * min(3, len(names)), (height + 24) * ((len(names) + 2) // 3)),
                "white",
            )
            painter = ImageDraw.Draw(strip)
            for i, name in enumerate(names):
                x, y = i % 3 * width, i // 3 * (height + 24)
                painter.text((x + 4, y + 4), name, fill="black")
                strip.paste(images[name].crop(crop), (x, y + 24))
            strip.save(folder / "window-native-crops.png")
            body.append(
                f"<p><a href='{case['id']}/window-native-crops.png'>Native window crops, no resizing</a>; <a href='{case['id']}/window-boundary-region.png'>proxy region</a>. Review occlusion/visibility before interpretation.</p>"
            )
    write_json(
        args.output / "measurements.json",
        {
            "records": records,
            "scope": "Pixel/edge proxies only; semantic landmark and quality interpretation requires actual visual inspection.",
        },
    )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RoomCraft V1.0 causality</title><body style="font:16px system-ui;max-width:1600px;margin:auto">'
        + "\n".join(body),
        encoding="utf-8",
    )
    print(f"Verified {len(records)} actual output records")


if __name__ == "__main__":
    main()
