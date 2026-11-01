"""RoomCraft V0.8: CPU geometry preparation, approval-gated GPU training/evaluation."""

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch

from roomcraft.common import model_spec, sha256, write_json
from roomcraft.editing import EditConfig, edit
from roomcraft.rga import RGAConfig
from roomcraft.rga_data import prepare_geometry, validate_geometry
from roomcraft.rga_training import RGATrainConfig, train_rga

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--manifest", type=Path, default=ROOT / "data/v06_ready/manifest.json")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--size", type=int, default=512)
    prepare.add_argument("--annotations", type=Path)
    train = sub.add_parser("train")
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--geometry", type=Path, required=True)
    train.add_argument("--lora", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--resume", type=Path)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument(
        "--experiment", type=Path, default=ROOT / "configs/rga_v08_experiment.json"
    )
    evaluate.add_argument("--rga-checkpoint", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.action == "prepare":
        prepare_geometry(args.manifest, args.output, args.size, args.annotations)
        record = validate_geometry(args.output / "geometry.json")
        print(json.dumps({"prepared": len(record["rows"]), "size": record["size"]}))
    elif args.action == "train":
        record = json.loads(args.config.read_text(encoding="utf-8"))
        config = RGATrainConfig(**{**record, "geometry": RGAConfig(**record["geometry"])})
        train_rga(args.cache, args.geometry, args.lora, args.output, config, args.resume)
    else:
        experiment = json.loads(args.experiment.read_text(encoding="utf-8"))
        protocol_path = ROOT / experiment["v07_protocol"]
        if sha256(protocol_path) != experiment["v07_protocol_sha256"]:
            raise ValueError("changed held-out evaluation protocol")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        lora = ROOT / protocol["checkpoint"]
        if sha256(lora / "adapter.safetensors") != protocol["adapter_sha256"]:
            raise ValueError("changed frozen LoRA")
        if args.output.exists():
            raise ValueError("evaluation output must be new")
        if args.run:
            if args.rga_checkpoint is None or not all(
                (args.rga_checkpoint / name).is_file()
                for name in ("config.json", "adapter.safetensors")
            ):
                raise ValueError("no trained RGA checkpoint supplied")
            rga = json.loads((args.rga_checkpoint / "config.json").read_text(encoding="utf-8"))
            if (
                rga.get("format") != "roomcraft-rga-v1"
                or rga.get("base") != model_spec("qwen")
                or rga.get("frozen_lora_sha256") != protocol["adapter_sha256"]
            ):
                raise ValueError("incompatible RGA checkpoint identity")
            RGAConfig(**rga["geometry"]).validate()
        write_json(
            args.output / "protocol.json",
            {
                **experiment,
                "status": "execution_requested"
                if args.run
                else "plan_only_no_training_or_generation",
                "rga_checkpoint": str(args.rga_checkpoint) if args.rga_checkpoint else None,
                "cases": protocol["cases"],
                "rga_weights_exist": args.rga_checkpoint is not None
                and (args.rga_checkpoint / "adapter.safetensors").is_file(),
            },
        )
        if not args.run:
            return
        for case in protocol["cases"]:
            source = ROOT / case["source"]
            if sha256(source) != case["source_sha256"]:
                raise ValueError("changed evaluation input")
            for size in experiment["sizes"]:
                for seed in experiment["seeds"]:
                    config = EditConfig(size=size, seed=seed, **experiment["settings"])
                    for label, weights, geometry, scale in (
                        ("A_qwen", None, None, 0),
                        ("B_lora", lora, None, 0),
                        ("C_lora_rga", lora, args.rga_checkpoint, 1),
                        ("D_rga_disabled", lora, args.rga_checkpoint, 0),
                    ):
                        folder = args.output / case["id"] / str(size) / f"seed-{seed}" / label
                        edit(
                            source,
                            case["instruction"],
                            folder,
                            config,
                            weights,
                            rga_checkpoint=geometry,
                            rga_strength=scale,
                        )
                        gc.collect()
                        torch.cuda.empty_cache()
                    parent = folder.parent
                    if sha256(parent / "B_lora/output.png") != sha256(
                        parent / "D_rga_disabled/output.png"
                    ):
                        write_json(
                            parent / "disabled-ablation-warning.json",
                            {
                                "warning": "B/D differ; inspect numerical nondeterminism/settings before attributing effects to geometry"
                            },
                        )
        write_json(
            args.output / "completed.json",
            {"status": "evaluation_completed", "settings": asdict(config)},
        )


if __name__ == "__main__":
    main()
