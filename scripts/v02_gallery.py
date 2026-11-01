"""Assemble actual local outputs and measured diagnostics; never generate quality scores."""

import html
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from roomcraft.evaluation import structure_metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/v02"
REGIONS = {
    "windows": {
        "windows": [(470, 0, 640, 495), (725, 125, 980, 385)],
        "walls": [(20, 20, 430, 650), (1000, 30, 1240, 400)],
    },
    "apartment": {"doors": [(460, 145, 700, 680)], "walls": [(720, 30, 1140, 430)]},
    "hotel": {"curtain_window_proxy": [(720, 30, 1210, 1190)], "walls": [(20, 20, 650, 680)]},
}


def main():
    sources = json.loads((ROOT / "experiments/v02_sources.json").read_text())
    sources_by_hash = {s["sha256"]: s for s in sources}
    rows = []
    sections = []
    notes_path = ROOT / "experiments/v02_visual_review.json"
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    for path in sorted(OUTPUT.glob("*/metadata.json")):
        meta = json.loads(path.read_text())
        run = path.parent
        source = sources_by_hash[meta["input_sha256"]]
        images = {
            name: Image.open(run / f"{name}.png").convert("RGB")
            for name in ("input", "raw", "output")
        }
        mask = Image.open(run / "mask.png").convert("L")
        edit = np.asarray(mask) > 0
        original = np.asarray(images["input"], dtype=np.float32)
        raw = np.asarray(images["raw"], dtype=np.float32)
        final = np.asarray(images["output"], dtype=np.float32)
        record = {
            "id": run.name,
            "source_id": source["id"],
            "metadata": str(path.relative_to(ROOT)).replace("\\", "/"),
            "settings": meta["settings"],
            "prompt": meta["prompt"],
            "generation_seconds": meta["generation_seconds"],
            "total_seconds": meta["total_seconds"],
            "peak_allocated_gib": meta["peak_gpu_allocated_bytes"] / 2**30,
            "peak_reserved_gib": meta["peak_gpu_reserved_bytes"] / 2**30,
            "editable_fraction": float(edit.mean()),
            "raw_protected_mae_255": float(np.abs(raw - original)[~edit].mean()),
            "composite_protected_max_error_255": float(np.abs(final - original)[~edit].max()),
            "edit_mae_255": float(np.abs(final - original)[edit].mean()),
            "visual_review": notes.get(run.name, {"status": "not yet reviewed"}),
        }
        rows.append(record)
        region_results = {}
        with Image.open(ROOT / f"data/v02/{source['id']}.jpg") as photo:
            for label, boxes in REGIONS[source["id"]].items():
                region = Image.new("L", photo.size)
                drawing = ImageDraw.Draw(region)
                for box in boxes:
                    drawing.rectangle(box, fill=255)
                region = region.resize(mask.size, Image.Resampling.NEAREST)
                region_results[label] = {
                    "raw": structure_metrics(images["input"], images["raw"], region),
                    "composite": structure_metrics(images["input"], images["output"], region),
                }
        record["manual_region_proxies"] = region_results
        overlay = Image.composite(
            Image.blend(images["input"], Image.new("RGB", mask.size, (255, 60, 60)), 0.4),
            images["input"],
            mask,
        )
        panels = [images["input"], overlay, images["raw"], images["output"]]
        sheet = Image.new("RGB", (1600, 480), "white")
        draw = ImageDraw.Draw(sheet)
        for i, (panel, label) in enumerate(
            zip(panels, ["Input", "Editable mask", "Raw generation", "Hard composite"], strict=True)
        ):
            fitted = ImageOps.contain(panel, (400, 400))
            sheet.paste(
                fitted, (i * 400 + (400 - fitted.width) // 2, 30 + (400 - fitted.height) // 2)
            )
            draw.text((i * 400 + 10, 10), label, fill="black")
        draw.text(
            (10, 435),
            f"{run.name} | {meta['generation_seconds']:.1f}s | "
            f"{record['peak_allocated_gib']:.2f} GiB allocated | "
            f"strength {meta['settings']['strength']} | CFG {meta['settings']['guidance']} | "
            f"steps {meta['settings']['steps']} | seed {meta['settings']['seed']}",
            fill="black",
        )
        draw.text(
            (10, 456),
            f"Source: {source['file_title']} | {source['license']} | resized and AI-edited; credits in index.html",
            fill="black",
        )
        sheet.save(run / "comparison.png")
        author = html.escape(html.unescape(re.sub(r"<[^>]+>", "", source["author_html"])))
        sections.append(
            f"<section><h2>{html.escape(run.name)}</h2>"
            f'<img src="{run.name}/comparison.png" alt="Input, mask, raw and composited output">'
            f"<p>{html.escape(json.dumps(record['visual_review']))}</p>"
            f'<p>Source: <a href="{html.escape(source["source_page"])}">{html.escape(source["file_title"])}</a>, '
            f'{author}, <a href="{html.escape(source["license_url"])}">{source["license"]}</a>. '
            "Resized and AI-edited; mask overlays added. BY-SA adaptations retain BY-SA. No author endorsement.</p>"
            f'<p><a href="{run.name}/metadata.json">Full settings and measurements</a> | '
            f'<a href="{run.name}/raw.png">Full raw image</a> | '
            f'<a href="{run.name}/output.png">Full composite</a></p></section>'
        )
        for image in images.values():
            image.close()
    (ROOT / "experiments/v02_results.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RoomCraft V0.2 local evaluation</title>'
        "<style>body{font:16px system-ui;max-width:1600px;margin:auto;padding:20px;background:#eee}img{width:100%}section{background:white;padding:16px;margin:20px 0}</style>"
        "<h1>RoomCraft V0.2: actual local GPU outputs</h1><p>Exploratory three-photo evaluation. "
        "Visual notes are an AI-assisted review, not a blinded human panel. Pixel errors measure changes, not furniture quality. "
        "Hard compositing guarantees protected pixels match the resized input; this does not demonstrate model geometry preservation.</p>"
        + "".join(sections),
        encoding="utf-8",
    )
    print(f"Recorded {len(rows)} completed runs; outputs/v02/index.html")
    (ROOT / "experiments/v02_regions.json").write_text(
        json.dumps(
            {
                "method": "Hand-selected boxes in downloaded-photo pixels; not semantic segmentation. "
                "Wall texture and Canny edge F1 are proxies, not geometry certification. "
                "Curtains occlude the hotel window, so that region is explicitly a proxy.",
                "boxes_xyxy": REGIONS,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
