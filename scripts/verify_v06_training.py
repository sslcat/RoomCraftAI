"""Fail the approved GPU workflow unless genuine trainer artifacts meet the smoke/pilot contract."""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file

from roomcraft.common import model_spec, sha256, write_json


def verify_run(folder: Path, expected_steps: list[int], manifest_hash: str):
    config = json.loads((folder / "config.json").read_text())
    done = json.loads((folder / "completed.json").read_text())
    entries = [json.loads(line) for line in (folder / "loss.jsonl").read_text().splitlines()]
    if [e["step"] for e in entries] != expected_steps:
        raise ValueError("optimizer update log does not match expected continuation")
    if config["base"] != model_spec("qwen") or config["manifest_sha256"] != manifest_hash:
        raise ValueError("wrong model or dataset")
    if done["status"] != "training_completed" or done["steps"] != expected_steps[-1]:
        raise ValueError("training did not complete")
    if config["rank"] != 8:
        raise ValueError("unexpected adapter rank")
    for entry in entries:
        for key in ("train_loss", "gradient_norm", "elapsed_seconds"):
            if not math.isfinite(entry[key]) or entry[key] <= 0:
                raise ValueError(f"invalid {key}")
        if "validation_loss" in entry and not math.isfinite(entry["validation_loss"]):
            raise ValueError("invalid validation loss")
        if entry["gradient_tensors"] != 958 or entry["peak_gpu_allocated_bytes"] <= 0:
            raise ValueError("missing full-architecture gradient/memory evidence")
    checkpoint = folder / f"step-{expected_steps[-1]:06d}"
    state = torch.load(checkpoint / "state.pt", map_location="cpu", weights_only=True)
    if state["step"] != expected_steps[-1] or len(state["optimizer"]["state"]) != 958:
        raise ValueError("missing optimizer state")
    for values in state["optimizer"]["state"].values():
        for value in values.values():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise ValueError("nonfinite optimizer state")
        if int(values["step"]) != expected_steps[-1]:
            raise ValueError("optimizer step counter did not resume")
    tensors = load_file(str(checkpoint / "adapter.safetensors"))
    if len(tensors) != 958 or sum(t.numel() for t in tensors.values()) != 23_543_808:
        raise ValueError("wrong adapter tensor layout")
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise ValueError("nonfinite adapter")
    if not any(t.count_nonzero() for n, t in tensors.items() if n.endswith(".b")):
        raise ValueError("zero-initialized B matrices never changed")
    return {
        "checkpoint": str(checkpoint),
        "adapter_sha256": sha256(checkpoint / "adapter.safetensors"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    digest = sha256(args.manifest)
    if args.pilot:
        result = verify_run(args.root / "pilot", list(range(1, 101)), digest)
        output = args.root / "pilot_verified.json"
    else:
        first = verify_run(args.root / "smoke-first", [1], digest)
        second = verify_run(args.root / "smoke-resumed", [2], digest)
        config = json.loads((args.root / "smoke-resumed/config.json").read_text())
        if Path(config["resume"]).resolve() != Path(first["checkpoint"]).resolve():
            raise ValueError("second update did not resume first checkpoint")
        if first["adapter_sha256"] == second["adapter_sha256"]:
            raise ValueError("adapter bytes did not change after resumed update")
        result = {"first": first, "resumed": second}
        output = args.root / "smoke_verified.json"
    if output.exists():
        raise ValueError("verification output already exists")
    write_json(output, {"status": "verified_real_training_artifacts", **result})


if __name__ == "__main__":
    main()
