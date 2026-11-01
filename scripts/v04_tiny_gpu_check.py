"""Test random miniature Qwen architecture in local BF16. NOT pretrained/model training evidence."""

import json
import time
from pathlib import Path

import torch
from diffusers import QwenImageTransformer2DModel

from roomcraft.common import environment, seed_all, write_json
from roomcraft.qwen_objective import gradient_report, inject_qwen_lora, paired_loss


def main():
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA unavailable")
    seed_all(42)
    model = QwenImageTransformer2DModel(
        in_channels=64,
        out_channels=16,
        num_layers=2,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=16,
        axes_dims_rope=(2, 2, 4),
        zero_cond_t=True,
    ).to(device="cuda", dtype=torch.bfloat16)
    targets = inject_qwen_lora(model, 2, 2)
    model.enable_gradient_checkpointing()
    model.train()
    data = {
        "target": torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
        "source": torch.randn(1, 4, 64, device="cuda", dtype=torch.bfloat16),
        "text": torch.randn(1, 3, 16, device="cuda", dtype=torch.bfloat16),
        "text_mask": torch.ones(1, 3, device="cuda", dtype=torch.bool),
        "shapes": torch.tensor([[1, 2, 2], [1, 2, 2]]),
    }
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    rows = []
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = paired_loss(model, data, 42)
        loss.backward()
        gradients = gradient_report(model)
        optimizer.step()
        rows.append({"step": step + 1, "loss": float(loss.detach()), **gradients})
    torch.cuda.synchronize()
    result = {
        "status": "tiny_random_architecture_test_only",
        "pretrained_weights_used": False,
        "roomcraft_adapter_saved": False,
        "environment": environment(),
        "projection_count": len(targets),
        "rows": rows,
        "seconds": time.perf_counter() - started,
        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
    }
    write_json(Path("experiments/v04_tiny_gpu_check.json"), result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
