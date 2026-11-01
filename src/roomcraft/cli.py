"""Command line interface; importing/help never downloads model weights."""

import argparse
import json
from pathlib import Path

from roomcraft.common import write_json
from roomcraft.data import validate_manifest
from roomcraft.editing import EditConfig, edit, edit_preflight
from roomcraft.evaluation import compare_runs, summarize_human
from roomcraft.inference import InferenceConfig, infer
from roomcraft.preprocess import prepare, save_prepared
from roomcraft.training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="roomcraft")
    commands = parser.add_subparsers(dest="command", required=True)
    preprocess = commands.add_parser("preprocess", help="CPU-only image/mask/edge preparation")
    inference = commands.add_parser(
        "infer", help="GPU SDXL inference; downloads weights on first run"
    )
    for sub in (preprocess, inference):
        sub.add_argument("--image", type=Path, required=True)
        sub.add_argument("--mask", type=Path)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--size", type=int, default=1024)
    inference.add_argument("--style", required=True)
    inference.add_argument("--prompt", help="Exact positive prompt for controlled comparisons")
    inference.add_argument("--seed", type=int, default=42)
    inference.add_argument("--steps", type=int, default=30)
    inference.add_argument("--guidance", type=float, default=7.5)
    inference.add_argument("--strength", type=float, default=0.99)
    inference.add_argument("--conditioning", choices=["none", "canny", "depth"], default="none")
    inference.add_argument("--control-scale", type=float, default=0.5)
    inference.add_argument("--depth", type=Path)
    inference.add_argument("--adapter", type=Path)
    inference.add_argument("--offload", action="store_true")
    editing = commands.add_parser("edit", help="Qwen instruction editing; use environments/qwen")
    editing.add_argument("--image", type=Path, required=True)
    editing.add_argument("--instruction", required=True)
    editing.add_argument("--output", type=Path, required=True)
    editing.add_argument("--size", type=int, default=1024)
    editing.add_argument("--steps", type=int, default=40)
    editing.add_argument("--true-cfg-scale", type=float, default=4.0)
    editing.add_argument("--seed", type=int, default=42)
    editing.add_argument("--negative-prompt", default=" ")
    editing.add_argument("--adapter", type=Path, help="RoomCraft Qwen checkpoint directory")
    editing.add_argument(
        "--offload",
        action="store_true",
        help="Whole-model CPU offload; still requires a large GPU and host RAM",
    )
    preflight = commands.add_parser(
        "edit-preflight", help="Check Qwen resources without downloading weights"
    )
    preflight.add_argument("--offload", action="store_true")
    preflight.add_argument("--output", type=Path)
    validate = commands.add_parser("validate-manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--allow-fixture", action="store_true")
    training = commands.add_parser("train")
    training.add_argument("--manifest", type=Path, required=True)
    training.add_argument("--config", type=Path, required=True)
    training.add_argument("--output", type=Path, required=True)
    training.add_argument("--resume", type=Path)
    evaluation = commands.add_parser("compare")
    evaluation.add_argument("--baseline", type=Path, required=True)
    evaluation.add_argument("--candidate", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    evaluation.add_argument("--regions", type=Path)
    human = commands.add_parser("human-summary")
    human.add_argument("scores", type=Path)
    pairs = commands.add_parser(
        "validate-pairs", help="Validate rights-reviewed aligned Qwen pairs"
    )
    pairs.add_argument("manifest", type=Path)
    split = commands.add_parser(
        "split-pairs", help="Deterministic property-level train/validation/test split"
    )
    split.add_argument("manifest", type=Path)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    cache = commands.add_parser("qwen-cache", help="GPU encode frozen VAE/VLM conditioning")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--size", type=int, default=512)
    qtrain = commands.add_parser(
        "train-qwen", help="Train Qwen 2511 LoRA from reviewed cached pairs"
    )
    qtrain.add_argument("--cache", type=Path, required=True)
    qtrain.add_argument("--config", type=Path, required=True)
    qtrain.add_argument("--output", type=Path, required=True)
    qtrain.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.command == "preprocess":
        image, mask = prepare(args.image, args.mask, args.size)
        save_prepared(image, mask, args.output)
        write_json(
            args.output / "preprocessing.json",
            {
                "status": "CPU preprocessing only; no generation",
                "size": list(image.size),
                "mask_origin": "user" if args.mask else "heuristic",
            },
        )
    elif args.command == "infer":
        config = InferenceConfig(
            **{key: getattr(args, key) for key in InferenceConfig.__dataclass_fields__}
        )
        print(
            json.dumps(
                infer(
                    args.image,
                    args.mask,
                    args.style,
                    args.output,
                    config,
                    args.depth,
                    args.adapter,
                    prompt_override=args.prompt,
                ),
                indent=2,
            )
        )
    elif args.command == "edit":
        config_edit = EditConfig(
            **{key: getattr(args, key) for key in EditConfig.__dataclass_fields__}
        )
        print(
            json.dumps(
                edit(args.image, args.instruction, args.output, config_edit, args.adapter), indent=2
            )
        )
    elif args.command == "edit-preflight":
        result = edit_preflight(args.offload)
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.command == "validate-manifest":
        print(f"Validated {len(validate_manifest(args.manifest, args.allow_fixture))} examples")
    elif args.command == "train":
        config_train = TrainConfig(**json.loads(args.config.read_text(encoding="utf-8")))
        train(args.manifest, args.output, config_train, args.resume)
    elif args.command == "compare":
        print(
            json.dumps(
                compare_runs(args.baseline, args.candidate, args.output, args.regions), indent=2
            )
        )
    elif args.command == "human-summary":
        print(json.dumps(summarize_human(args.scores), indent=2))
    elif args.command in ("validate-pairs", "split-pairs"):
        from roomcraft.qwen_data import split_pairs, validate_pairs

        if args.command == "validate-pairs":
            print(f"Validated {len(validate_pairs(args.manifest))} pairs")
        else:
            split_pairs(args.manifest, args.output, args.seed)
    elif args.command == "qwen-cache":
        from roomcraft.qwen_cache import cache_pairs

        cache_pairs(args.manifest, args.output, args.size)
    elif args.command == "train-qwen":
        from roomcraft.qwen_training import QwenTrainConfig, train_qwen

        train_qwen(
            args.cache,
            args.output,
            QwenTrainConfig(**json.loads(args.config.read_text(encoding="utf-8"))),
            args.resume,
        )
