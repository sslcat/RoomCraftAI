"""Free, repeatable native-pixel audit and hash-locked inference-only protocol."""

import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roomcraft.common import sha256, write_json
from roomcraft.preprocess import read_rgb

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/v07"


def image_record(path):
    image = read_rgb(path)
    with Image.open(path) as encoded:
        container = {"format": encoded.format, "encoded_mode": encoded.mode}
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "dimensions": list(image.size),
        **container,
        "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "mean_squared_sobel": float(
            np.mean(cv2.Sobel(gray, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(gray, cv2.CV_32F, 0, 1) ** 2)
        ),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "outputs/v06/trained/pilot/step-000100"
    config = json.loads((checkpoint / "config.json").read_text())
    manifest = ROOT / "data/v06_ready/manifest.json"
    rows = json.loads(manifest.read_text())["pairs"]
    results = {
        "warning": "Laplacian/Sobel are content-sensitive sharpness proxies, not perceptual quality or architecture scores. Compare matched-size images cautiously.",
        "adapter_sha256": sha256(checkpoint / "adapter.safetensors"),
        "checkpoint_files": {
            name: image_path.stat().st_size
            for name in ("adapter.safetensors", "config.json", "state.pt")
            if (image_path := checkpoint / name).exists()
        },
        "manifest_matches_training": sha256(manifest) == config["manifest_sha256"],
        "training_implementation_unchanged": {
            name: sha256(ROOT / "src/roomcraft" / name) == digest
            for name, digest in config["implementation_sha256"].items()
        },
        "gallery": {
            "native_comparison_width": 1536,
            "max_css_width": 1540,
            "max_css_scale_at_100_percent_zoom": 1540 / 1536,
            "browser_zoom_and_device_pixel_ratio": "not measured",
        },
        "cases": [],
    }
    cases = []
    pages = [
        '<!doctype html><meta charset="utf-8"><title>V0.7 native-pixel audit</title>',
        "<style>body{font:16px system-ui;margin:24px}.strip{overflow:auto}img{max-width:none}</style>",
        "<h1>Actual V0.6 pixels: free V0.7 audit</h1><p>All panels below are 1 image pixel per CSS pixel at 100% browser zoom. Scroll horizontally. No sharpening or upscaling. Synthetic renders only; no new generation.</p>",
    ]
    for case_id in ("family_05_v00", "family_18_v00", "family_19_v00"):
        row = next(r for r in rows if r["id"] == case_id)
        assert row["split"] == "test"
        base = ROOT / "outputs/v06/evaluation" / case_id
        paths = [
            ROOT / "data/v06_ready" / row["source"],
            ROOT / "data/v06_ready" / row["target"],
            base / "baseline/input.png",
            base / "baseline/output.png",
            base / "adapter/output.png",
        ]
        records = [image_record(p) for p in paths]
        checks = {}
        for label in ("baseline", "adapter"):
            folder = base / label
            meta = json.loads((folder / "metadata.json").read_text())
            checks[label] = {
                "output_hash_matches": sha256(folder / "output.png") == meta["output_sha256"],
                "raw_and_output_bytes_identical": sha256(folder / "raw.png")
                == sha256(folder / "output.png"),
                "source_hash_matches": sha256(paths[0]) == meta["input_sha256"],
                "native_dimensions_match_metadata": records[3 if label == "baseline" else 4][
                    "dimensions"
                ]
                == meta["working_size"],
                "generation_seconds": meta["generation_seconds"],
                "peak_gpu_allocated_bytes": meta["peak_gpu_allocated_bytes"],
            }
            if not all(value for key, value in checks[label].items() if isinstance(value, bool)):
                raise ValueError("V0.6 artifact integrity failure")
            destination = OUT / case_id / "512/seed-42" / label
            destination.mkdir(parents=True, exist_ok=True)
            for name in ("metadata.json", "source.png", "input.png", "raw.png", "output.png"):
                if (destination / name).exists():
                    if sha256(destination / name) != sha256(folder / name):
                        raise ValueError("refusing to overwrite changed reused artifact")
                else:
                    shutil.copyfile(folder / name, destination / name)
            write_json(
                destination / "v07-reuse.json",
                {
                    "status": "actual_v06_result_reused_no_new_inference",
                    "original": folder.relative_to(ROOT).as_posix(),
                    "metadata_sha256": sha256(folder / "metadata.json"),
                },
            )
        # Identical target content at both resolutions isolates detail lost by downsampling.
        reduced = read_rgb(paths[1]).resize((512, 384), Image.Resampling.LANCZOS)
        reduced_path = OUT / f"{case_id}-render-512.png"
        reduced.save(reduced_path)
        records.append(image_record(reduced_path))
        results["cases"].append({"id": case_id, "images": records, "checks": checks})
        panels = [read_rgb(paths[2]), reduced, read_rgb(paths[3]), read_rgb(paths[4])]
        strip = Image.new("RGB", (2048, 416), "white")
        draw = ImageDraw.Draw(strip)
        for i, (panel, label) in enumerate(
            zip(
                panels,
                (
                    "Empty render downsampled",
                    "Furnished render downsampled (not model)",
                    "Unchanged Qwen native 512",
                    "Adapter native 512",
                ),
                strict=True,
            )
        ):
            strip.paste(panel, (i * 512, 32))
            draw.text((i * 512 + 4, 8), label, fill="black")
        strip.save(OUT / f"{case_id}-native.png")
        # Same central coordinates at native 1024 and at 512; no interpolation of crops.
        crop_strip = Image.new("RGB", (5 * 256, 288), "white")
        draw = ImageDraw.Draw(crop_strip)
        images = [read_rgb(paths[1]), *panels]
        for i, image in enumerate(images):
            x, y = image.width // 2 - 128, image.height // 2 - 128
            crop_strip.paste(image.crop((x, y, x + 256, y + 256)), (i * 256, 32))
            draw.text(
                (i * 256 + 3, 8),
                ("Render 1024", "Empty 512", "Render 512", "Qwen 512", "Adapter 512")[i],
                fill="black",
            )
        crop_strip.save(OUT / f"{case_id}-crops.png")
        pages += [
            f'<h2>{html.escape(case_id)}</h2><div class="strip"><img src="{case_id}-native.png" alt="Native 512 panels"></div>',
            f'<p>Central 256x256 crops at native pixels. 1024 crop covers a smaller field of view.</p><div class="strip"><img src="{case_id}-crops.png" alt="Native crops"></div>',
        ]
        cases.append(
            {
                "id": case_id,
                "domain": "synthetic",
                "source": paths[0].relative_to(ROOT).as_posix(),
                "source_sha256": sha256(paths[0]),
                "instruction": row["instruction"],
                "provenance": {
                    "author": row["creator"],
                    "source_url": row["source_url"],
                    "license": row["license"],
                    "license_url": row["license_url"],
                },
            }
        )
    sources = json.loads((ROOT / "experiments/v02_sources.json").read_text())
    real_instructions = {
        "windows": "Furnish this room as a contemporary living room with a neutral fabric two-seat sofa against the right wall beneath the small window, a low wooden coffee table, a wool rug and a small potted plant. Preserve both windows, walls, baseboards, dark wood flooring, camera viewpoint and daylight. Do not add or move architectural features.",
        "apartment": "Furnish this room as a modest contemporary bedroom with a double bed, two compact bedside tables and a small dresser. Keep the open door and closet accessible. Preserve walls, doorways, closet, carpet, camera viewpoint and existing lighting. Do not add or move architectural features.",
        "hotel": "Furnish this small corner with one comfortable blue-gray upholstered armchair, a small wooden side table and a reading lamp. Keep circulation space open. Preserve the curtains, wallpaper, parquet floor, walls, camera viewpoint and existing lighting. Do not add or move architectural features.",
    }
    for source in sources:
        if source["status"] != "included":
            continue
        path = ROOT / "data/v02" / f"{source['id']}.jpg"
        if sha256(path) != source["sha256"]:
            raise ValueError("changed licensed real photo")
        cases.append(
            {
                "id": source["id"],
                "domain": "real",
                "source": path.relative_to(ROOT).as_posix(),
                "source_sha256": sha256(path),
                "instruction": real_instructions[source["id"]],
                "provenance": {
                    "author": "upyernoz" if source["id"] == "hotel" else "aismallard",
                    "source_url": source["source_page"],
                    "license": source["license"],
                    "license_url": source["license_url"],
                    "restrictions": source["restrictions"],
                    "rights_evidence": f"data/v02/{source['id']}-rights.json",
                },
                "image": image_record(path),
            }
        )
    # Execute each resolution across all rooms before progressing to the larger size.
    runs = [
        {"case": c["id"], "size": size, "seed": 42}
        for size in (512, 768, 1024)
        for c in sorted(cases, key=lambda c: c["domain"])
    ]
    runs += [{"case": c["id"], "size": 768, "seed": 17} for c in cases if c["domain"] == "real"]
    write_json(
        ROOT / "configs/v07_evaluation.json",
        {
            "status": "plan_only_no_paid_approval",
            "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
            "adapter_sha256": results["adapter_sha256"],
            "settings": {
                "steps": 40,
                "true_cfg_scale": 4.0,
                "negative_prompt": " ",
                "offload": False,
            },
            "comparison_warning": "Equal seeds across resolutions do not yield identical noise tensors. Each baseline/adapter pair has the same dimensions and settings.",
            "cases": cases,
            "runs": runs,
            "planned_image_count": len(runs) * 2,
            "reused_v06_images": 6,
            "new_inference_images": len(runs) * 2 - 6,
        },
    )
    write_json(ROOT / "experiments/v07_pixel_audit.json", results)
    (OUT / "pixel-audit.html").write_text("\n".join(pages), encoding="utf-8")
    print(
        json.dumps(
            {
                "images_audited": sum(len(c["images"]) for c in results["cases"]),
                "planned_images": len(runs) * 2,
                "checkpoint": results["checkpoint_files"],
            }
        )
    )


if __name__ == "__main__":
    main()
