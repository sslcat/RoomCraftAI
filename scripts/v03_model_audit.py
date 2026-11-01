"""Count full Qwen architecture and candidate LoRA parameters on meta; no weight download."""

import json
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel
from torch import nn

from roomcraft.common import write_json

ROOT = Path(__file__).resolve().parents[1]
config = json.loads((ROOT / "docs/sources/v03/transformer--config.json").read_text())
with torch.device("meta"):
    model = QwenImageTransformer2DModel.from_config(config)
targets = (
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
)
layers = [
    (name, module)
    for name, module in model.named_modules()
    if isinstance(module, nn.Linear) and any(name.endswith("." + target) for target in targets)
]
parameters = sum(p.numel() for p in model.parameters())
files = json.loads((ROOT / "docs/sources/v03/model-api.json").read_text())["siblings"]
weight_files = [f for f in files if f["rfilename"].endswith(".safetensors")]
record = {
    "method": "Allocation-free meta architecture count; no model weights or training",
    "transformer_parameters": parameters,
    "transformer_bf16_gib": parameters * 2 / 2**30,
    "stored_weight_bytes": sum(f["size"] for f in weight_files),
    "stored_weight_gib": sum(f["size"] for f in weight_files) / 2**30,
    "zero_cond_t": model.config.zero_cond_t,
    "lora_targets": list(targets),
    "lora_target_linear_layers": len(layers),
    "lora_rank16_parameters": sum(16 * (m.in_features + m.out_features) for _, m in layers),
    "lora_rank8_parameters": sum(8 * (m.in_features + m.out_features) for _, m in layers),
    "example_layers": [
        {"name": n, "in": m.in_features, "out": m.out_features} for n, m in layers[:8]
    ],
}
write_json(ROOT / "experiments/v03_model_audit.json", record)
print(json.dumps(record, indent=2))
