"""Audit actual V0.6 inputs and the full Qwen META architecture without pretrained downloads."""

import json
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel
from diffusers.pipelines.qwenimage import pipeline_qwenimage_edit_plus as upstream
from PIL import Image

from roomcraft.common import environment, model_spec, sha256, write_json
from roomcraft.editing import edit_preflight, edit_size
from roomcraft.lora import parameter_report
from roomcraft.qwen_data import validate_pairs
from roomcraft.qwen_objective import inject_qwen_lora
from roomcraft.qwen_training import QwenTrainConfig


def main():
    manifest = Path("data/v06_ready/manifest.json")
    rows = validate_pairs(manifest)
    architecture = json.loads(Path("docs/sources/v03/transformer--config.json").read_text())
    with torch.device("meta"):
        model = QwenImageTransformer2DModel.from_config(architecture)
        projections = inject_qwen_lora(model, 8, 8)
    parameters = parameter_report(model)
    parameters.pop("tensors")
    dimensions = set()
    for row in rows:
        with Image.open(manifest.parent / row["source"]) as source:
            target = edit_size(source, 512)
            reference = upstream.calculate_dimensions(
                upstream.VAE_IMAGE_SIZE, source.width / source.height
            )
            vlm = upstream.calculate_dimensions(
                upstream.CONDITION_IMAGE_SIZE, source.width / source.height
            )
            dimensions.add((target, reference, vlm))
    configs = {}
    for name in ("first", "smoke", "pilot"):
        path = Path(f"configs/qwen_train_v06_{name}.json")
        value = json.loads(path.read_text())
        QwenTrainConfig(**value).validate()
        configs[name] = {"sha256": sha256(path), "settings": value}
    write_json(
        Path("experiments/v06_readiness.json"),
        {
            "status": "cpu_data_and_meta_architecture_checks_only",
            "base": model_spec("qwen"),
            "environment": environment(),
            "manifest_sha256": sha256(manifest),
            "validated_pairs": len(rows),
            "projections": len(projections),
            "parameters": parameters,
            "adapter_gradient_adam_bytes": parameters["trainable"] * 16,
            "dimensions_target_reference_vlm": sorted(dimensions),
            "configs": configs,
            "local_gpu_preflight": edit_preflight(),
            "upstream_pipeline_sha256": sha256(Path(upstream.__file__)),
            "pretrained_weights_loaded": False,
            "genuine_optimizer_updates": 0,
        },
    )


if __name__ == "__main__":
    main()
