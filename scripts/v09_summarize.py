"""Verify downloaded V0.9 images and build a local four-arm review gallery."""

import argparse
import html
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from roomcraft.common import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def summarize(output):
    protocol = json.loads((output / "protocol.json").read_text())
    results = json.loads((output / "results.json").read_text())
    parts = [
        '<!doctype html><meta charset="utf-8"><title>RoomCraft V0.9</title>',
        "<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}td{padding:8px;border:1px solid #aaa;vertical-align:top}img{width:280px;height:auto}.scroll{overflow:auto}</style>",
        "<h1>RGA four-arm evaluation</h1><p>A: unchanged Qwen. B: frozen V0.6 LoRA. C: LoRA + trained RGA. D: same RGA disabled. Native PNG links preserve actual resolution. Reused V0.7 controls are labeled. AI observations are subjective; pixel/sharpness proxies are not architecture or realism scores.</p>",
        f"<p>Trial: {html.escape(output.name)}. RGA SHA256: <code>{html.escape(protocol['rga_sha256'])}</code>. See each image's metadata for the recorded residual gain and settings.</p>",
    ]
    metrics = []
    for case in protocol["cases"]:
        credit = case["provenance"]
        source = ROOT / case["source"]
        if sha256(source) != case["source_sha256"]:
            raise ValueError("source changed")
        parts += [
            f"<h2>{case['id']} ({case['domain']})</h2>",
            f"<p>{html.escape(case['instruction'])}</p>",
            f'<p>{html.escape(credit["author"])}: <a href="{html.escape(credit["source_url"], quote=True)}">source</a>; <a href="{html.escape(credit["license_url"], quote=True)}">{html.escape(credit["license"])}</a>. AI edited derivatives; image rights remain separate from code licensing.</p>',
        ]
        for size in (1024, 768):
            for seed in (42, 17):
                parent = output / case["id"] / str(size) / f"seed-{seed}"
                if not parent.exists():
                    continue
                parts += [f'<h3>{size} long side, seed {seed}</h3><div class="scroll"><table><tr>']
                tiles = []
                row = {"case": case["id"], "domain": case["domain"], "size": size, "seed": seed}
                pixels = {}
                for label in ("source", "A_qwen", "B_lora", "C_lora_rga", "D_rga_disabled"):
                    path = source if label == "source" else parent / label / "output.png"
                    if not path.exists():
                        parts.append(f"<td>{label}: NOT EXECUTED</td>")
                        continue
                    with Image.open(path) as image:
                        image = image.convert("RGB")
                        if label != "source":
                            meta = json.loads((path.parent / "metadata.json").read_text())
                            if (
                                sha256(path) != meta["output_sha256"]
                                or list(image.size) != meta["working_size"]
                            ):
                                raise ValueError("output integrity mismatch")
                            if (
                                meta["input_sha256"] != case["source_sha256"]
                                or meta["prompt"] != case["instruction"]
                            ):
                                raise ValueError("comparison inputs differ")
                            pixels[label] = np.asarray(image).astype(np.float32)
                            row[label] = {
                                "sha256": meta["output_sha256"],
                                "dimensions": list(image.size),
                                "generation_seconds": meta["generation_seconds"],
                                "peak_gpu_allocated_bytes": meta["peak_gpu_allocated_bytes"],
                                "laplacian_variance_proxy": float(
                                    cv2.Laplacian(
                                        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY),
                                        cv2.CV_64F,
                                    ).var()
                                ),
                            }
                        image.thumbnail((384, 320), Image.Resampling.LANCZOS)
                        tile = Image.new("RGB", (384, 352), "white")
                        tile.paste(image, ((384 - image.width) // 2, 28))
                        ImageDraw.Draw(tile).text((8, 7), label, fill="black")
                        tiles.append(tile)
                    link = Path(os.path.relpath(path, output)).as_posix()
                    reused = label != "source" and (path.parent / "reuse.json").exists()
                    parts.append(
                        f'<td>{label}{" (V0.7 reused)" if reused else ""}<br><a href="{link}"><img src="{link}" alt="{label}"></a></td>'
                    )
                if {"B_lora", "C_lora_rga", "D_rga_disabled"} <= pixels.keys():
                    row["B_D_pixels_exact"] = bool(
                        np.array_equal(pixels["B_lora"], pixels["D_rga_disabled"])
                    )
                    row["B_C_mean_absolute_pixel_delta"] = float(
                        np.abs(pixels["B_lora"] - pixels["C_lora_rga"]).mean()
                    )
                metrics.append(row)
                strip = Image.new("RGB", (384 * len(tiles), 352), "white")
                for i, tile in enumerate(tiles):
                    strip.paste(tile, (384 * i, 0))
                strip.save(parent / "comparison.png")
                parts.append("</tr></table></div>")
                if "B_D_pixels_exact" in row:
                    parts.append(
                        f"<p>B/D pixels identical: {row['B_D_pixels_exact']}. B/C mean absolute RGB difference: {row['B_C_mean_absolute_pixel_delta']:.4f} on a 0–255 scale. These measure change, not improvement.</p>"
                    )
    parts.append(f"<p>{len(results)} completed image records. Missing entries are not results.</p>")
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")
    write_json(output / "measurements.json", metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
