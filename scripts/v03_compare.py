"""Run the fixed three-photo comparison only with --run; otherwise report actual status."""

import argparse
import html
import json
import re
from pathlib import Path

from roomcraft.common import model_spec, sha256, write_json
from roomcraft.editing import EditConfig, edit, edit_preflight

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Execute local GPU inference; never provisions resources"
    )
    parser.add_argument("--ids", nargs="*", choices=["windows", "apartment", "hotel"])
    args = parser.parse_args()
    plan = json.loads((ROOT / "configs/v03_comparison.json").read_text())
    config = EditConfig(**plan["qwen_settings"])
    sources = {s["id"]: s for s in json.loads((ROOT / "experiments/v02_sources.json").read_text())}
    if args.run:
        preflight = edit_preflight(config.offload)
        if not preflight["can_attempt"]:
            raise RuntimeError("No inference started: " + "; ".join(preflight["reasons"]))
    rows, sections = [], []
    dest = ROOT / "outputs/v03"
    dest.mkdir(parents=True, exist_ok=True)
    for case in plan["cases"]:
        image = ROOT / case["image"]
        baseline = ROOT / case["sdxl"]
        base_meta = json.loads((baseline / "metadata.json").read_text())
        if sha256(image) != case["sha256"] or base_meta["input_sha256"] != case["sha256"]:
            raise ValueError("Source mismatch; comparison is invalid")
        if base_meta["settings"]["seed"] != config.seed or base_meta["adapter_sha256"] is not None:
            raise ValueError("Baseline must be unchanged and use the comparison seed")
        output = dest / case["id"]
        meta_path = output / "metadata.json"
        if args.run and (not args.ids or case["id"] in args.ids) and not meta_path.exists():
            edit(image, case["instruction"], output, config)
        qwen = json.loads(meta_path.read_text()) if meta_path.exists() else None
        if qwen:
            if (
                qwen["base"] != model_spec("qwen")
                or qwen["input_sha256"] != case["sha256"]
                or qwen["settings"] != plan["qwen_settings"]
                or qwen["prompt"] != case["instruction"]
                or qwen["working_size"] != base_meta["working_size"]
            ):
                raise ValueError("Qwen artifact differs from preregistered comparison")
            if sha256(output / "output.png") != qwen["output_sha256"]:
                raise ValueError("Qwen output integrity mismatch")
        row = {
            "id": case["id"],
            "goal": case["goal"],
            "status": "generated; visual review required"
            if qwen
            else "not executed; no Qwen image or quality result",
            "sdxl_metadata": str((baseline / "metadata.json").relative_to(ROOT)).replace("\\", "/"),
            "qwen_metadata": str(meta_path.relative_to(ROOT)).replace("\\", "/") if qwen else None,
            "qwen_generation_seconds": qwen["generation_seconds"] if qwen else None,
            "qwen_peak_gpu_allocated_bytes": qwen["peak_gpu_allocated_bytes"] if qwen else None,
            "quality_review": None,
        }
        rows.append(row)
        source = sources[case["id"]]
        base_url = "../v02/" + baseline.name
        panels = "".join(
            f'<figure><img src="{base_url}/{filename}.png"><figcaption>{caption}</figcaption></figure>'
            for filename, caption in [
                ("input", "Input"),
                ("raw", "SDXL raw"),
                ("output", "SDXL hard composite"),
            ]
        )
        panels += (
            f'<figure><img src="{case["id"]}/raw.png"><figcaption>Qwen raw, uncomposited</figcaption></figure>'
            if qwen
            else '<figure class="pending">Qwen NOT RUN<br>No output or quality score</figure>'
        )
        author = html.escape(html.unescape(re.sub(r"<[^>]+>", "", source["author_html"])))
        sections.append(
            f"<section><h2>{case['id']}: {html.escape(case['goal'])}</h2><div>{panels}</div>"
            f"<p>Instruction: {html.escape(case['instruction'])}</p>"
            f'<p><a href="{html.escape(source["source_page"])}">Source photograph</a>: {author}; '
            f'<a href="{html.escape(source["license_url"])}">{source["license"]}</a>. '
            "Resized; SDXL panels AI-edited. Any Qwen result is also an AI edit. BY-SA adaptations retain BY-SA; no endorsement.</p></section>"
        )
    write_json(ROOT / "experiments/v03_results.json", {"protocol": plan["protocol"], "cases": rows})
    (dest / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RoomCraft V0.3 comparison status</title>'
        "<style>body{font:16px system-ui;padding:20px}section{margin-bottom:35px}section>div{display:flex;gap:10px}figure{margin:0;width:25%}img{width:100%}.pending{background:#eee;display:grid;place-content:center;text-align:center}</style>"
        "<h1>V0.3: comparison status, not a quality benchmark</h1><p>Baseline: V0.2, seed 42, strength 0.99. "
        "Qwen target: 40 steps, true CFG 4, BF16, seed 42. Judge geometry on raw outputs; SDXL compositing copies protected pixels.</p>"
        + "".join(sections),
        encoding="utf-8",
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
