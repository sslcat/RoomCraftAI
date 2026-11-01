"""Measure selected LoRA counts on the full meta architecture; never downloads weights."""

import json
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel

from roomcraft.common import model_spec, write_json
from roomcraft.lora import parameter_report
from roomcraft.qwen_objective import inject_qwen_lora


def main():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "docs/sources/v03/transformer--config.json").read_text())
    records = []
    for rank in (8, 16):
        with torch.device("meta"):
            model = QwenImageTransformer2DModel.from_config(config)
            targets = inject_qwen_lora(model, rank, rank)
        report = parameter_report(model)
        report.pop("tensors")
        records.append(
            {
                "rank": rank,
                "projections": len(targets),
                "parameters": report,
                "fp32_adapter_gradient_adam_gib": report["trainable"] * 16 / 2**30,
            }
        )
    result = {
        "status": "meta_architecture_only_no_weights",
        "base": model_spec("qwen"),
        "excluded_projection": "transformer_blocks.59.attn.to_add_out",
        "reason": "Final text output is discarded and has no path to the image loss",
        "counts": records,
    }
    write_json(root / "experiments/v04_model_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
