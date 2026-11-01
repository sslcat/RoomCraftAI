"""Free local compiled-stack audit; --exercise requires an approved Blackwell Pod."""

import argparse
import json
from importlib import metadata
from pathlib import Path

import torch
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    versions = {
        name: metadata.version(name)
        for name in ("torch", "torchvision", "diffusers", "transformers")
    }
    arch = torch.cuda.get_arch_list()
    if (
        versions
        != {
            "torch": "2.8.0+cu128",
            "torchvision": "0.23.0+cu128",
            "diffusers": "0.37.1",
            "transformers": "4.57.6",
        }
        or torch.version.cuda != "12.8"
        or "sm_120" not in arch
    ):
        raise RuntimeError("pinned cu128/Blackwell stack not present")
    result = {
        "versions": versions,
        "compiled_architectures": arch,
        "cuda": torch.version.cuda,
        "blackwell_gpu_exercised": False,
    }
    if args.exercise:
        if torch.cuda.get_device_capability() != (12, 0) or not torch.cuda.is_bf16_supported():
            raise RuntimeError("need actual sm_120 Blackwell with BF16")
        q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        loss = F.scaled_dot_product_attention(q, q, q).float().square().mean()
        loss.backward()
        if not torch.isfinite(loss) or q.grad is None or not torch.isfinite(q.grad).all():
            raise RuntimeError("BF16 SDPA gradient failure")
        matrix = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        if not torch.isfinite(matrix @ matrix).all():
            raise RuntimeError("BF16 matrix kernel failure")
        torch.cuda.synchronize()
        result.update(
            blackwell_gpu_exercised=True,
            gpu=torch.cuda.get_device_name(),
            limitation="Kernel probe only; full Qwen loading and optimizer smoke still required",
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
