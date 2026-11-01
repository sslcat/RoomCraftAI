"""CPU-only verification and native-pixel crops of actual V0.7 outputs."""

import json
import statistics

from PIL import Image, ImageDraw
from v07_evaluate import ROOT, gallery, load_protocol, run_name

from roomcraft.common import sha256, write_json
from roomcraft.preprocess import read_rgb


def main():
    protocol = load_protocol(ROOT / "configs/v07_evaluation.json")
    output = ROOT / "outputs/v07"
    gallery(protocol, output)
    status = json.loads((output / "run-status.json").read_text())
    cases = {case["id"]: case for case in protocol["cases"]}
    records = []
    for entry in status:
        if entry["status"] not in ("completed_v07", "reused_actual_v06"):
            continue
        folder = output / run_name(entry) / entry["variant"]
        meta = json.loads((folder / "metadata.json").read_text())
        if sha256(folder / "raw.png") != meta["output_sha256"]:
            raise ValueError("raw generation differs from published output")
        if meta["quantization"] != "none" or meta["compositing"] != "none":
            raise ValueError("unexpected processing")
        records.append(
            {
                **entry,
                "domain": cases[entry["case"]]["domain"],
                "metadata": (folder / "metadata.json").relative_to(ROOT).as_posix(),
                "total_seconds": meta["total_seconds"],
                "input_sha256": meta["input_sha256"],
                "peak_gpu_reserved_bytes": meta["peak_gpu_reserved_bytes"],
            }
        )
    groups = []
    for domain in ("real", "synthetic"):
        for size in (512, 768, 1024):
            for variant in ("baseline", "adapter"):
                group = [
                    r
                    for r in records
                    if r["domain"] == domain
                    and r["size"] == size
                    and r["variant"] == variant
                    and r["seed"] == 42
                ]
                if group:
                    groups.append(
                        {
                            "domain": domain,
                            "size": size,
                            "variant": variant,
                            "images": len(group),
                            "generation_seconds_mean": statistics.mean(
                                r["generation_seconds"] for r in group
                            ),
                            "generation_seconds_range": [
                                min(r["generation_seconds"] for r in group),
                                max(r["generation_seconds"] for r in group),
                            ],
                            "maximum_allocated_gib": max(
                                r["peak_gpu_allocated_bytes"] for r in group
                            )
                            / 2**30,
                        }
                    )
    write_json(
        ROOT / "experiments/v07_inference_results.json",
        {
            "protocol_sha256": sha256(ROOT / "configs/v07_evaluation.json"),
            "new_generations": sum(r["status"] == "completed_v07" for r in records),
            "reused_v06_images": sum(r["status"] == "reused_actual_v06" for r in records),
            "pending_or_failed": [
                r for r in status if r["status"] not in ("completed_v07", "reused_actual_v06")
            ],
            "quality_score": None,
            "groups_seed_42": groups,
            "runs": records,
        },
    )
    for case in (c for c in protocol["cases"] if c["domain"] == "real"):
        strip = Image.new("RGB", (6 * 256, 288), "white")
        draw = ImageDraw.Draw(strip)
        for i, (size, variant) in enumerate(
            (s, v) for s in (512, 768, 1024) for v in ("baseline", "adapter")
        ):
            path = output / case["id"] / str(size) / "seed-42" / variant / "output.png"
            draw.text((i * 256 + 4, 8), f"{size} {variant}", fill="black")
            if path.exists():
                image = read_rgb(path)
                cx, cy = int(image.width * 0.65), int(image.height * 0.68)
                x = min(max(0, cx - 128), image.width - 256)
                y = min(max(0, cy - 128), image.height - 256)
                strip.paste(image.crop((x, y, x + 256, y + 256)), (i * 256, 32))
        strip.save(output / f"{case['id']}-native-crops.png")
    (output / "native-crops.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Actual native-pixel crops</title>'
        "<style>body{font:16px system-ui;margin:24px}.strip{overflow:auto}img{max-width:none}</style>"
        "<h1>Actual real-photo generation: 256x256 native crops</h1>"
        "<p>No upscaling or sharpening. Each crop is 256 image pixels, shown at 256 CSS pixels at 100% zoom. "
        "Higher-resolution crops cover a smaller field of view. Furniture/noise changes across sizes; "
        "these are visual evidence, not a numerical sharpness benchmark.</p>"
        + "".join(
            f'<h2>{case["id"]}</h2><div class="strip"><img src="{case["id"]}-native-crops.png" alt="Native crops"></div>'
            for case in protocol["cases"]
            if case["domain"] == "real"
        ),
        encoding="utf-8",
    )
    print(json.dumps({"verified_images": len(records), "groups_seed_42": groups}))


if __name__ == "__main__":
    main()
