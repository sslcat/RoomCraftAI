"""Bounded full-model RGA experiment on an already approved RoomCraft Pod."""

import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from v07_evaluate import load_protocol, verified_existing

from roomcraft.common import sha256, write_json
from roomcraft.editing import EditConfig, edit
from roomcraft.rga_data import validate_geometry

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/v09-rga"
OUT = ROOT / "outputs/v09"


def training(name, config, deadline, resume=None):
    command = [
        sys.executable,
        "scripts/rga_v08.py",
        "train",
        "--cache",
        "runs/v06-first/cache/cache.json",
        "--geometry",
        "data/v08_geometry/geometry.json",
        "--lora",
        "runs/v06-first/pilot/step-000100",
        "--config",
        config,
        "--output",
        str(RUN / name),
    ]
    if resume:
        command += ["--resume", str(resume)]
    with (RUN / (name + ".log")).open("x") as log:
        result = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, timeout=max(1, deadline - time.time())
        )
    if result.returncode:
        raise RuntimeError(f"{name} failed; inspect durable log")


def resume_check():
    a = RUN / "smoke-resumed/step-000002"
    b = RUN / "smoke-uninterrupted/step-000002"
    left, right = (
        load_file(str(a / "adapter.safetensors")),
        load_file(str(b / "adapter.safetensors")),
    )
    if left.keys() != right.keys():
        raise RuntimeError("resume keys differ")
    diffs = {k: float((left[k] - right[k]).abs().max()) for k in left}
    state_a = torch.load(a / "state.pt", map_location="cpu", weights_only=True)
    state_b = torch.load(b / "state.pt", map_location="cpu", weights_only=True)

    def equal(x, y):
        if isinstance(x, torch.Tensor):
            return torch.equal(x, y)
        if isinstance(x, dict):
            return x.keys() == y.keys() and all(equal(x[k], y[k]) for k in x)
        if isinstance(x, list):
            return len(x) == len(y) and all(equal(i, j) for i, j in zip(x, y, strict=True))
        return x == y

    exact = all(v == 0 for v in diffs.values()) and equal(state_a, state_b)
    logs = [
        json.loads(line) for line in (RUN / "smoke-resumed/loss.jsonl").read_text().splitlines()
    ]
    evidence = {
        "exact_weights_and_optimizer_state": exact,
        "tensor_max_abs_differences": diffs,
        "resumed_step": state_a["step"],
        "encoder_gradient_norm": logs[-1]["encoder_gradient_norm"],
        "resumed_sha256": sha256(a / "adapter.safetensors"),
        "uninterrupted_sha256": sha256(b / "adapter.safetensors"),
    }
    write_json(RUN / "resume-verification.json", evidence)
    if not exact or evidence["encoder_gradient_norm"] <= 0:
        raise RuntimeError("smoke resume/encoder gate failed; no pilot")


def evaluation(deadline, calibrated=False):
    experiment = json.loads((ROOT / "configs/rga_v08_experiment.json").read_text())
    if calibrated:
        experiment.update(
            sizes=[1024],
            seeds=[42],
            expected_images=24,
            selection_reason="One predeclared gain calibration; all six original primary native-1024 cases, seed 42. Full 96-image matrix remains unexecuted.",
        )
    protocol_path = ROOT / experiment["v07_protocol"]
    if sha256(protocol_path) != experiment["v07_protocol_sha256"]:
        raise ValueError("evaluation protocol changed")
    protocol = load_protocol(protocol_path)
    checkpoint = RUN / "pilot/step-000100"
    if not checkpoint.exists():
        completed = json.loads((RUN / "pilot/completed.json").read_text())
        if completed["status"] != "rga_validation_guard_stopped":
            raise ValueError("no completed pilot checkpoint")
        candidates = [
            json.loads(line)
            for line in (RUN / "pilot/loss.jsonl").read_text().splitlines()
            if "validation_loss" in line
        ]
        chosen = min(candidates, key=lambda row: row["validation_loss"])
        checkpoint = RUN / "pilot" / f"step-{chosen['step']:06d}"
        write_json(
            RUN / "evaluation-checkpoint-selection.json",
            {
                "reason": "validation safety guard stopped pilot; evaluate lowest fixed-validation-loss saved checkpoint",
                "checkpoint": str(checkpoint),
                "step": chosen["step"],
            },
        )
    lora = ROOT / protocol["checkpoint"]
    write_json(
        OUT / "protocol.json",
        {
            **experiment,
            "status": "executing_V09_authorized",
            "cases": protocol["cases"],
            "rga_sha256": sha256(checkpoint / "adapter.safetensors"),
            "order": "1024 then 768, seed 42 then 17, real then synthetic; reuse verified V07 A/B",
        },
    )
    cases = sorted(protocol["cases"], key=lambda c: c["domain"] != "real")
    results = []
    for size in (1024,) if calibrated else (1024, 768):
        for seed in (42,) if calibrated else (42, 17):
            for case in cases:
                config = EditConfig(size=size, seed=seed, **experiment["settings"])
                parent = OUT / case["id"] / str(size) / f"seed-{seed}"
                for label, old, weights, rga, strength in (
                    ("A_qwen", "baseline", None, None, 0),
                    ("B_lora", "adapter", lora, None, 0),
                    ("C_lora_rga", None, lora, checkpoint, 1),
                    ("D_rga_disabled", None, lora, checkpoint, 0),
                ):
                    folder = parent / label
                    prior = (
                        ROOT / "outputs/v07" / case["id"] / str(size) / f"seed-{seed}" / str(old)
                    )
                    if old and (prior / "metadata.json").exists():
                        verified_existing(
                            prior, case, config, protocol["adapter_sha256"] if weights else None
                        )
                        shutil.copytree(prior, folder)
                        write_json(
                            folder / "reuse.json",
                            {"source": str(prior.relative_to(ROOT)), "new_generation": False},
                        )
                    else:
                        if deadline - time.time() < 180:
                            write_json(OUT / "results.json", results)
                            return
                        print(
                            json.dumps(
                                {
                                    "generating": str(folder.relative_to(ROOT)),
                                    "remaining_seconds": deadline - time.time(),
                                }
                            ),
                            flush=True,
                        )
                        edit(
                            ROOT / case["source"],
                            case["instruction"],
                            folder,
                            config,
                            weights,
                            rga_checkpoint=rga,
                            rga_strength=strength,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                    results.append(
                        {
                            "case": case["id"],
                            "domain": case["domain"],
                            "size": size,
                            "seed": seed,
                            "arm": label,
                            "reused": (folder / "reuse.json").exists(),
                            "sha256": sha256(folder / "output.png"),
                        }
                    )
                    write_json(OUT / "results.json", results)
                write_json(
                    parent / "disabled-equivalence.json",
                    {
                        "identical_png_bytes": sha256(parent / "B_lora/output.png")
                        == sha256(parent / "D_rga_disabled/output.png")
                    },
                )
    write_json(OUT / "completed.json", {"images": len(results)})


def main():
    global RUN, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deadline-unix", type=float, required=True)
    parser.add_argument("--attempt", type=int, choices=(1, 2, 3, 4), default=1)
    args = parser.parse_args()
    if args.attempt >= 2:
        RUN = ROOT / f"runs/v09-rga-retry{args.attempt - 1}"
        OUT = ROOT / f"outputs/v09-retry{args.attempt - 1}"
    calibrated = args.attempt == 4
    if calibrated:
        RUN = ROOT / "runs/v09-rga-calibrated"
        OUT = ROOT / "outputs/v09-calibrated"
    smoke_config = (
        "configs/rga_v09_calibrated_smoke.json" if calibrated else "configs/rga_v08_smoke.json"
    )
    pilot_config = (
        "configs/rga_v09_calibrated_pilot.json" if calibrated else "configs/rga_v08_pilot.json"
    )
    RUN.mkdir(parents=True, exist_ok=False)
    OUT.mkdir(parents=True, exist_ok=False)
    try:
        geometry = validate_geometry(ROOT / "data/v08_geometry/geometry.json")
        if len(geometry["rows"]) != 100:
            raise ValueError("incomplete geometry")
        expected = "1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6"
        if sha256(ROOT / "runs/v06-first/pilot/step-000100/adapter.safetensors") != expected:
            raise ValueError("frozen LoRA changed")
        write_json(
            RUN / "preflight.json",
            {
                "geometry_rows": 100,
                "frozen_lora_sha256": expected,
                "gpu": torch.cuda.get_device_name(),
                "bf16": torch.cuda.is_bf16_supported(),
                "gpu_memory": torch.cuda.mem_get_info(),
                "deadline_unix": args.deadline_unix,
            },
        )
        training("smoke-uninterrupted", smoke_config, args.deadline_unix - 1200)
        training(
            "smoke-resumed",
            smoke_config,
            args.deadline_unix - 1200,
            RUN / "smoke-uninterrupted/step-000001",
        )
        resume_check()
        if args.deadline_unix - time.time() < 3600:
            raise RuntimeError("insufficient pilot/evaluation reserve")
        training("pilot", pilot_config, args.deadline_unix - 1200)
        evaluation(args.deadline_unix, calibrated)
    except Exception as error:
        write_json(
            RUN / "failure.json",
            {"type": type(error).__name__, "message": str(error), "epoch": time.time()},
        )
        raise
    finally:
        records = {
            p.relative_to(ROOT).as_posix(): sha256(p)
            for p in RUN.rglob("*")
            if p.is_file() and p.name != "artifact-hashes.json"
        }
        write_json(RUN / "artifact-hashes.json", records)


if __name__ == "__main__":
    main()
