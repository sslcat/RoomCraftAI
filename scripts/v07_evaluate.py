"""Inference-only, hash-locked V0.7 comparisons; never provisions or trains."""

import argparse
import gc
import html
import json
import os
import time
from pathlib import Path

from roomcraft.common import model_spec, sha256, write_json
from roomcraft.editing import EditConfig, edit, edit_size
from roomcraft.preprocess import read_rgb
from roomcraft.qwen_evaluation import compare_qwen

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_HASH = "1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6"


def load_protocol(path):
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol["settings"] != {
        "steps": 40,
        "true_cfg_scale": 4.0,
        "negative_prompt": " ",
        "offload": False,
    }:
        raise ValueError("protocol settings differ from the locked V0.6 comparison")
    if protocol["adapter_sha256"] != ADAPTER_HASH:
        raise ValueError("V0.6 adapter identity mismatch")
    ids = [case["id"] for case in protocol["cases"]]
    if len(ids) != len(set(ids)) or any(not name.replace("_", "").isalnum() for name in ids):
        raise ValueError("case ids must be unique safe names")
    for case in protocol["cases"]:
        source = (ROOT / case["source"]).resolve()
        if not source.is_relative_to(ROOT) or sha256(source) != case["source_sha256"]:
            raise ValueError("source identity mismatch")
        if not case["instruction"].strip() or case["domain"] not in ("real", "synthetic"):
            raise ValueError("invalid case")
    for run in protocol["runs"]:
        if run["case"] not in ids or run["size"] not in (512, 768, 1024):
            raise ValueError("invalid run")
        EditConfig(size=run["size"], seed=run["seed"]).validate()
    keys = [(r["case"], r["size"], r["seed"]) for r in protocol["runs"]]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate runs")
    checkpoint = ROOT / protocol["checkpoint"]
    if sha256(checkpoint / "adapter.safetensors") != ADAPTER_HASH:
        raise ValueError("checkpoint changed")
    return protocol


def run_name(run):
    return f"{run['case']}/{run['size']}/seed-{run['seed']}"


def verified_existing(folder, case, config, adapter_hash):
    """Resume only complete outputs matching input, instruction and every setting."""
    from dataclasses import asdict

    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    if (
        meta["input_sha256"] != case["source_sha256"]
        or meta["prompt"] != case["instruction"]
        or meta["settings"] != asdict(config)
        or meta["adapter_sha256"] != adapter_hash
        or sha256(folder / "output.png") != meta["output_sha256"]
        or meta["base"] != model_spec("qwen")
    ):
        raise ValueError("existing output does not match locked experiment")
    expected = edit_size(read_rgb(ROOT / case["source"]), config.size)
    if tuple(meta["working_size"]) != expected or read_rgb(folder / "output.png").size != expected:
        raise ValueError("existing output has incorrect native dimensions")


def gallery(protocol, output):
    """Pending cells are text, never fabricated or upscaled generation results."""
    output.mkdir(parents=True, exist_ok=True)
    review_path = ROOT / "experiments/v07_visual_review.json"
    reviews = json.loads(review_path.read_text(encoding="utf-8")) if review_path.exists() else []
    for domain, filename in (("real", "real-photo.html"), (None, "resolution.html")):
        parts = [
            '<!doctype html><html lang="en"><meta charset="utf-8">',
            "<title>RoomCraft V0.7 comparison</title>",
            "<style>body{font:16px system-ui;margin:24px}table{border-collapse:collapse}"
            "td,th{border:1px solid #bbb;padding:8px;vertical-align:top}"
            "img{max-width:100%;height:auto} .strip{overflow:auto}</style>",
            "<h1>V0.7 matched comparisons</h1><p>"
            "Only completed, hash-verified runs appear as results. "
            "Synthetic 512/seed-42 images are reused actual V0.6 outputs, not new V0.7 runs. "
            "Open PNG links for native pixels; preview fitting is display only. "
            "AI visual review is not an objective quality benchmark.</p>",
        ]
        for case in protocol["cases"]:
            if domain and case["domain"] != domain:
                continue
            parts += [f"<h2>{html.escape(case['id'])} ({case['domain']})</h2>"]
            credit = case["provenance"]
            source_link = Path(os.path.relpath(ROOT / case["source"], output)).as_posix()
            parts += [
                f"<p>{html.escape(case['instruction'])}</p>",
                f"<p>{html.escape(credit['author'])} — "
                f'<a href="{html.escape(credit["source_url"], quote=True)}">source</a>; '
                f'<a href="{html.escape(credit["license_url"], quote=True)}">'
                f"{html.escape(credit['license'])}</a>. "
                "Preview resized; generated derivatives, when present, are AI edits. "
                "Photo/derivative rights are separate from the code license.</p>",
                f'<a href="{source_link}">Original input</a><br>'
                f'<img width="384" src="{source_link}" alt="Original room">',
                "<table><tr><th>Native long side / seed</th><th>Unchanged Qwen</th>"
                "<th>RoomCraft adapter</th></tr>",
            ]
            for run in (r for r in protocol["runs"] if r["case"] == case["id"]):
                parts.append(f"<tr><td>{run['size']} / {run['seed']}</td>")
                for label in ("baseline", "adapter"):
                    folder = output / run_name(run) / label
                    if (folder / "metadata.json").exists():
                        verified_existing(
                            folder,
                            case,
                            EditConfig(size=run["size"], seed=run["seed"]),
                            ADAPTER_HASH if label == "adapter" else None,
                        )
                        relative = folder.relative_to(output).as_posix()
                        parts.append(
                            f'<td><a href="{relative}/output.png">Native PNG</a> '
                            f'<a href="{relative}/metadata.json">metadata</a><br>'
                            f'<img width="384" src="{relative}/output.png" alt="{label}"></td>'
                        )
                    else:
                        failure = folder.parent / f"{label}-failure.json"
                        if failure.exists():
                            link = failure.relative_to(output).as_posix()
                            parts.append(f'<td>FAILED — <a href="{link}">failure record</a></td>')
                        else:
                            parts.append("<td>NOT EXECUTED</td>")
                parts.append("</tr>")
                note = next(
                    (
                        r["findings"]
                        for r in reviews
                        if r["case"] == run["case"]
                        and r["size"] == run["size"]
                        and r["seed"] == run["seed"]
                    ),
                    None,
                )
                if note:
                    parts.append(
                        '<tr><td colspan="3"><b>AI visual inspection (subjective):</b> '
                        + html.escape(note)
                        + "</td></tr>"
                    )
            parts.append("</table>")
        parts.append("</html>")
        (output / filename).write_text("\n".join(parts), encoding="utf-8")
    status = []
    cases = {case["id"]: case for case in protocol["cases"]}
    for run in protocol["runs"]:
        for label in ("baseline", "adapter"):
            folder = output / run_name(run) / label
            entry = {
                **run,
                "variant": label,
                "status": "not_executed",
                "expected_native_size": edit_size(
                    read_rgb(ROOT / cases[run["case"]]["source"]), run["size"]
                ),
                "generation_seconds": None,
                "peak_gpu_allocated_bytes": None,
            }
            if (folder / "metadata.json").exists():
                meta = json.loads((folder / "metadata.json").read_text())
                entry.update(
                    {
                        key: meta[key]
                        for key in (
                            "generation_seconds",
                            "peak_gpu_allocated_bytes",
                            "working_size",
                            "output_sha256",
                            "adapter_sha256",
                        )
                    }
                )
                entry["status"] = (
                    "reused_actual_v06" if (folder / "v07-reuse.json").exists() else "completed_v07"
                )
            elif (folder.parent / f"{label}-failure.json").exists():
                entry["status"] = "failed"
            status.append(entry)
    write_json(output / "run-status.json", status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/v07_evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/v07")
    parser.add_argument("--run", action="store_true", help="Run on an already approved GPU")
    parser.add_argument(
        "--deadline-unix", type=float, help="Stop starting images with <5 minutes remaining"
    )
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    if args.run:
        import torch

        checkpoint = ROOT / protocol["checkpoint"]
        cases = {case["id"]: case for case in protocol["cases"]}
        for run in protocol["runs"]:
            case = cases[run["case"]]
            config = EditConfig(size=run["size"], seed=run["seed"])
            parent = args.output / run_name(run)
            failed = False
            for label, adapter in (("baseline", None), ("adapter", checkpoint)):
                folder = parent / label
                if (folder / "metadata.json").exists():
                    verified_existing(folder, case, config, ADAPTER_HASH if adapter else None)
                    continue
                if folder.exists():
                    raise ValueError(f"Incomplete output needs inspection: {folder}")
                if args.deadline_unix and time.time() + 300 >= args.deadline_unix:
                    gallery(protocol, args.output)
                    write_json(
                        args.output / "time-limit.json",
                        {"status": "stopped_before_deadline", "epoch": time.time()},
                    )
                    return
                try:
                    print(f"START {run_name(run)}/{label}", flush=True)
                    edit(ROOT / case["source"], case["instruction"], folder, config, adapter)
                    print(f"DONE {run_name(run)}/{label}", flush=True)
                except torch.OutOfMemoryError:
                    write_json(
                        parent / f"{label}-failure.json",
                        {
                            "status": "CUDA out of memory",
                            "run": run,
                            "native_resolution_fallback": False,
                        },
                    )
                    failed = True
                    break
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()
            if not failed and not (parent / "review").exists():
                compare_qwen(parent / "baseline", parent / "adapter", parent / "review")
            gallery(protocol, args.output)
    gallery(protocol, args.output)


if __name__ == "__main__":
    main()
