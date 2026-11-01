"""Run matched baseline/adapter generation on manifest TEST pairs only, after training approval."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path
from roomcraft.editing import EditConfig, edit
from roomcraft.qwen_data import validate_pairs
from roomcraft.qwen_evaluation import compare_qwen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ids", nargs="+", help="Predeclared subset of TEST ids; default is all")
    parser.add_argument(
        "--run", action="store_true", help="Actually run GPU inference; default writes a plan only"
    )
    args = parser.parse_args()
    rows = [r for r in validate_pairs(args.manifest) if r["split"] == "test"]
    if args.ids is not None:
        if len(set(args.ids)) != len(args.ids) or not set(args.ids) <= {r["id"] for r in rows}:
            raise ValueError("comparison ids must be distinct held-out TEST pairs")
        rows = [r for r in rows if r["id"] in args.ids]
    if args.output.exists():
        raise ValueError("evaluation output must be new")
    config = EditConfig(size=args.size, seed=args.seed)
    config.validate()
    record = json.loads((args.checkpoint / "config.json").read_text(encoding="utf-8"))
    if record.get("format") != "roomcraft-qwen-lora-v1":
        raise ValueError("need a Qwen training checkpoint")
    if record.get("manifest_sha256") != sha256(args.manifest):
        raise ValueError("held-out split must come from the exact training manifest")
    write_json(
        args.output / "protocol.json",
        {
            "manifest_sha256": sha256(args.manifest),
            "adapter_sha256": sha256(args.checkpoint / "adapter.safetensors"),
            "settings": asdict(config),
            "cases": rows,
            "status": "generation_requested" if args.run else "plan_only",
        },
    )
    if not args.run:
        return
    for row in rows:
        image = local_path(args.manifest.parent, row["source"])
        case = args.output / row["id"]
        for label, adapter in (("baseline", None), ("adapter", args.checkpoint)):
            edit(image, row["instruction"], case / label, config, adapter)
            gc.collect()
            torch.cuda.empty_cache()
        compare_qwen(case / "baseline", case / "adapter", case / "review")
        write_json(case / "review" / "provenance.json", row)


if __name__ == "__main__":
    main()
