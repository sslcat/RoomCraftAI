"""Single-GPU Qwen 2511 SFT: cached frozen conditioning and original attention LoRA."""

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.common import environment, model_spec, seed_all, sha256, write_json
from roomcraft.data import local_path
from roomcraft.lora import load_adapter, parameter_report
from roomcraft.qwen_cache import require_qwen_gpu, validate_cache
from roomcraft.qwen_objective import gradient_report, inject_qwen_lora, paired_loss


@dataclass
class QwenTrainConfig:
    steps: int = 100
    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 0.0001
    weight_decay: float = 0.01
    accumulation: int = 4
    seed: int = 42
    gradient_checkpointing: bool = True
    deterministic: bool = True
    clip_norm: float = 1.0
    validation_every: int = 10
    checkpoint_every: int = 10
    min_free_gpu_gib: float = 60.0

    def validate(self) -> None:
        for key in ("steps", "rank", "accumulation", "validation_every", "checkpoint_every"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("alpha", "learning_rate", "clip_norm", "min_free_gpu_gib"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"invalid {key}")
        if (
            not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not 0 <= self.seed < 2**31
        ):
            raise ValueError("invalid weight decay or seed")


def load_training_adapter(model: Any, checkpoint: Path) -> dict[str, Any]:
    record = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if record.get("format") != "roomcraft-qwen-lora-v1" or record.get("base") != model_spec("qwen"):
        raise ValueError("adapter does not match pinned Qwen 2511/project format")
    inject_qwen_lora(model, record["rank"], record["alpha"])
    load_adapter(model, checkpoint / "adapter.safetensors")
    model.eval()
    return {"sha256": sha256(checkpoint / "adapter.safetensors"), "config": record}


def sample_seed(seed: int, step: int, micro: int, accumulation: int) -> int:
    return (seed + 1_000_003 * (step * accumulation + micro + 1)) % (2**63 - 1)


def train_qwen(
    cache_path: Path, output: Path, config: QwenTrainConfig, resume: Path | None = None
) -> None:
    config.validate()
    cache = validate_cache(cache_path)
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory, including for resumed continuations")
    require_qwen_gpu(config.min_free_gpu_gib)
    seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic, warn_only=True)
    spec = model_spec("qwen")
    record = {
        **asdict(config),
        "format": "roomcraft-qwen-lora-v1",
        "base": spec,
        "cache_sha256": sha256(cache_path),
        "manifest_sha256": cache["manifest_sha256"],
        "objective": "uniform-sigma-unit-weight-target-flow-v1",
        "implementation_sha256": {
            name: sha256(Path(__file__).with_name(name))
            for name in (
                "qwen_training.py",
                "qwen_objective.py",
                "qwen_cache.py",
                "qwen_data.py",
                "lora.py",
                "checkpoint.py",
            )
        },
        "environment": environment(),
    }
    model = (
        import_module("diffusers")
        .QwenImageTransformer2DModel.from_pretrained(
            spec["repo"],
            subfolder="transformer",
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
        )
        .to("cuda")
    )
    inject_qwen_lora(model, config.rank, config.alpha)
    if config.gradient_checkpointing:
        model.enable_gradient_checkpointing()
    model.train()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # BF16 needs no loss scaling.
    start = resume_checkpoint(resume, model, optimizer, scaler, record) if resume else 0
    if start >= config.steps:
        raise ValueError("steps must exceed resumed optimizer step")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {**record, "resume": str(resume) if resume else None})
    write_json(output / "parameters.json", parameter_report(model))
    train_rows = [r for r in cache["rows"] if r["split"] == "train"]
    val_rows = [r for r in cache["rows"] if r["split"] == "validation"]

    def tensors(row: dict[str, Any]) -> dict[str, torch.Tensor]:
        loaded = load_file(str(local_path(cache_path.parent, row["file"])))
        return {
            k: v
            if k == "shapes"
            else v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in loaded.items()
        }

    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with (output / "loss.jsonl").open("x", encoding="utf-8") as log:
        for step in range(start, config.steps):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for micro in range(config.accumulation):
                seed = sample_seed(config.seed, step, micro, config.accumulation)
                seed_all(seed)
                row = train_rows[random.Random(seed).randrange(len(train_rows))]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = paired_loss(model, tensors(row), seed)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite training loss")
                (loss / config.accumulation).backward()
                loss_value += float(loss.detach()) / config.accumulation
            gradients = gradient_report(model)
            torch.nn.utils.clip_grad_norm_(parameters, config.clip_norm, error_if_nonfinite=True)
            optimizer.step()
            entry: dict[str, Any] = {"step": step + 1, "train_loss": loss_value, **gradients}
            if (step + 1) % config.validation_every == 0 or step + 1 == config.steps:
                model.eval()
                values = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i, row in enumerate(val_rows):
                        values.append(
                            float(paired_loss(model, tensors(row), config.seed + 10_000 + i))
                        )
                model.train()
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError("nonfinite validation loss")
                entry["validation_loss"] = sum(values) / len(values)
            torch.cuda.synchronize()
            entry.update(
                elapsed_seconds=time.perf_counter() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps:
                save_checkpoint(
                    output / f"step-{step + 1:06d}", model, optimizer, scaler, step + 1, record
                )
    write_json(
        output / "completed.json",
        {
            "steps": config.steps,
            "status": "training_completed",
            "base": spec,
            "cache_sha256": sha256(cache_path),
        },
    )
