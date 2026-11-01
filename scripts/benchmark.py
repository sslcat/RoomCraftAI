"""Matched baseline/adapter sweep on an explicit, rights-reviewed test manifest.

Separate inference processes isolate per-request CUDA peak counters. This is a
cold-process protocol, not an in-memory serving throughput benchmark.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path, validate_manifest
from roomcraft.evaluation import compare_runs

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--adapter", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--seeds", type=int, nargs="+", default=[17, 42, 123])
parser.add_argument("--size", type=int, default=1024)
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()
rows = [r for r in validate_manifest(args.manifest) if r["split"] == "test"]
if not rows:
    raise SystemExit("No rights-reviewed test examples; fixtures are not allowed")
if args.output.exists():
    raise SystemExit("Choose a new benchmark directory")
args.output.mkdir(parents=True)
write_json(
    args.output / "protocol.json",
    {
        "manifest_sha256": sha256(args.manifest),
        "adapter_sha256": sha256(args.adapter / "adapter.safetensors"),
        "seeds": args.seeds,
        "size": args.size,
        "steps": args.steps,
        "timing_protocol": "separate process per request; shared on-disk weights cache",
    },
)
for index, row in enumerate(rows):
    for seed in args.seeds:
        case = args.output / f"case-{index:05d}-seed-{seed}"
        common = [
            sys.executable,
            "-m",
            "roomcraft",
            "infer",
            "--image",
            str(local_path(args.manifest.parent, row["input"])),
            "--mask",
            str(local_path(args.manifest.parent, row["mask"])),
            "--style",
            row["prompt"],
            "--seed",
            str(seed),
            "--size",
            str(args.size),
            "--steps",
            str(args.steps),
        ]
        for name in ("baseline", "candidate"):
            command = [*common, "--output", str(case / name)]
            if name == "candidate":
                command += ["--adapter", str(args.adapter)]
            subprocess.run(command, check=True)
        result = compare_runs(case / "baseline", case / "candidate", case / "comparison")
        with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"id": row["id"], "seed": seed, "result": result}) + "\n")
