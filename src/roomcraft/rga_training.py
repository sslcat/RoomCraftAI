"""RGA-only flow training with frozen pretrained Qwen and frozen RoomCraft LoRA."""

import hashlib
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from importlib import import_module
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.common import environment, model_spec, seed_all, sha256, write_json
from roomcraft.data import local_path
from roomcraft.lora import parameter_report
from roomcraft.qwen_cache import require_qwen_gpu, validate_cache
from roomcraft.qwen_objective import gradient_report
from roomcraft.qwen_training import load_training_adapter, sample_seed
from roomcraft.rga import GeometryConditionedQwen, RGAConfig, rga_loss, rga_record
from roomcraft.rga_data import validate_geometry


def frozen_digest(model: torch.nn.Module) -> str:
    """Hash every frozen tensor's actual bytes, including the loaded LoRA."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class RGATrainConfig:
    steps: int = 2
    learning_rate: float = 1e-4
    accumulation: int = 1
    seed: int = 42
    weight_decay: float = 0.0
    clip_norm: float = 1.0
    architecture_lambda: float = 0.0
    checkpoint_every: int = 1
    validation_every: int = 1
    gradient_checkpointing: bool = True
    deterministic: bool = True
    min_free_gpu_gib: float = 60.0
    validation_loss_ceiling: float | None = None
    validation_patience: int = 2
    geometry: RGAConfig = field(default_factory=RGAConfig)

    def validate(self) -> None:
        self.geometry.validate()
        if (
            any(
                not isinstance(v, int) or v < 1
                for v in (
                    self.steps,
                    self.accumulation,
                    self.checkpoint_every,
                    self.validation_every,
                    self.validation_patience,
                )
            )
            or not 0 <= self.seed < 2**31
            or self.geometry.strength <= 0
        ):
            raise ValueError("invalid RGA steps/seed/accumulation/strength")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.learning_rate, self.clip_norm, self.min_free_gpu_gib)
        ):
            raise ValueError("invalid positive training parameter")
        if any(
            not math.isfinite(v) or v < 0 for v in (self.weight_decay, self.architecture_lambda)
        ):
            raise ValueError("invalid loss/decay parameter")
        if self.validation_loss_ceiling is not None and (
            not math.isfinite(self.validation_loss_ceiling) or self.validation_loss_ceiling <= 0
        ):
            raise ValueError("invalid validation loss ceiling")


def train_rga(
    cache_path: Path,
    geometry_path: Path,
    lora: Path,
    output: Path,
    config: RGATrainConfig,
    resume: Path | None = None,
) -> None:
    config.validate()
    cache = validate_cache(cache_path)
    geometry = validate_geometry(geometry_path, cache)
    provenance = json.loads((cache_path.parent / "provenance.json").read_text(encoding="utf-8"))
    source_hashes = {r["id"]: r["sha256"]["source"] for r in provenance["pairs"]}
    if any(row["source_sha256"] != source_hashes[row["id"]] for row in geometry["rows"]):
        raise ValueError("geometry source disagrees with sealed latent-cache provenance")
    if config.architecture_lambda and any(
        not r["verified_visible_architecture"] for r in geometry["rows"] if r["split"] != "test"
    ):
        raise ValueError(
            "weighted architecture loss requires verified visible masks for all train/val rows"
        )
    if output.exists():
        raise ValueError("RGA run output must be new, including for resumed continuation")
    require_qwen_gpu(config.min_free_gpu_gib)
    seed_all(config.seed)
    torch.use_deterministic_algorithms(config.deterministic, warn_only=False)
    spec = model_spec("qwen")
    base = (
        import_module("diffusers")
        .QwenImageTransformer2DModel.from_pretrained(
            spec["repo"],
            subfolder="transformer",
            revision=spec["revision"],
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
            low_cpu_mem_usage=True,
            local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1",
        )
        .to("cuda")
    )
    lora_info = load_training_adapter(base, lora)
    model = GeometryConditionedQwen(base, config.geometry).to(device="cuda")
    if config.gradient_checkpointing:
        base.enable_gradient_checkpointing()
    record = {
        **asdict(config),
        **rga_record(config.geometry),
        "base": spec,
        "frozen_lora_sha256": lora_info["sha256"],
        "cache_sha256": sha256(cache_path),
        "geometry_cache_sha256": sha256(geometry_path),
        "manifest_sha256": cache["manifest_sha256"],
        "objective": "uniform-sigma-target-flow-plus-optional-masked-velocity-v1",
        "environment": environment(),
        "implementation_sha256": {
            n: sha256(Path(__file__).with_name(n))
            for n in (
                "rga.py",
                "rga_training.py",
                "rga_data.py",
                "qwen_objective.py",
                "checkpoint.py",
            )
        },
    }
    params = list(model.geometry.parameters())
    optimizer = torch.optim.AdamW(params, lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    start = resume_checkpoint(resume, model, optimizer, scaler, record) if resume else 0
    if start >= config.steps:
        raise ValueError("steps must exceed resumed optimizer step")
    output.mkdir(parents=True)
    write_json(output / "config.json", record)
    write_json(output / "parameters.json", parameter_report(model))
    if any(p.requires_grad for p in base.parameters()) or sum(p.numel() for p in params) != 728512:
        raise RuntimeError("production RGA-only parameter/freeze gate failed")
    frozen_before = frozen_digest(base)
    write_json(output / "frozen-before.json", {"sha256": frozen_before, "all_frozen": True})
    geometry_rows = {r["id"]: r for r in geometry["rows"]}
    train = [r for r in cache["rows"] if r["split"] == "train"]
    val = [r for r in cache["rows"] if r["split"] == "validation"]

    def sample(row: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        tensors = load_file(str(local_path(cache_path.parent, row["file"])))
        tensors = {
            k: v
            if k == "shapes"
            else v.to(device="cuda", dtype=torch.bfloat16 if v.is_floating_point() else v.dtype)
            for k, v in tensors.items()
        }
        entry = geometry_rows[row["id"]]
        g = load_file(str(local_path(geometry_path.parent, entry["file"])))["geometry"]
        return tensors, g.to(device="cuda", dtype=torch.float32)

    # Full pretrained prediction equality before any optimization, not just loss equality.
    if not start:
        tensors, g = sample(val[0])
        predictions: list[torch.Tensor] = []

        def capture(_module: Any, _args: Any, result: Any) -> None:
            predictions.append(result[0].detach().cpu())

        def disable(_module: Any, args: Any, kwargs: dict[str, Any]) -> tuple[Any, Any]:
            kwargs.pop("controlnet_block_samples", None)
            return args, kwargs

        handle = base.register_forward_hook(capture)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            rga_loss(model, tensors, g, config.seed)
            hook = base.register_forward_pre_hook(disable, with_kwargs=True)
            try:
                rga_loss(model, tensors, g, config.seed)
            finally:
                hook.remove()
                handle.remove()
        difference = float((predictions[0].float() - predictions[1].float()).abs().max())
        write_json(
            output / "zero-baseline.json",
            {
                "maximum_prediction_difference": difference,
                "prediction_shape": list(predictions[0].shape),
                "geometry_shape": list(g.shape),
            },
        )
        if difference != 0:
            raise RuntimeError("zero RGA changed pretrained baseline prediction")
        del predictions
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    bad_validations = (
        int(json.loads((resume / "config.json").read_text()).get("validation_guard_streak", 0))
        if resume
        else 0
    )
    stopped = False
    with (output / "loss.jsonl").open("x", encoding="utf-8") as log:
        for step in range(start, config.steps):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            components = {"flow": 0.0, "architecture": 0.0}
            for micro in range(config.accumulation):
                seed = sample_seed(config.seed, step, micro, config.accumulation)
                seed_all(seed)
                row = train[random.Random(seed).randrange(len(train))]
                tensors, g = sample(row)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts = rga_loss(model, tensors, g, seed, config.architecture_lambda)
                if not torch.isfinite(loss):
                    raise RuntimeError("nonfinite RGA loss")
                (loss / config.accumulation).backward()
                total += float(loss.detach()) / config.accumulation
                for key in components:
                    components[key] += parts[key] / config.accumulation
            gradients = gradient_report(model)
            if any(p.grad is not None for p in base.parameters()):
                raise RuntimeError("frozen Qwen/LoRA received parameter gradients")
            encoder_squared = sum(
                float(p.grad.float().square().sum())
                for p in model.geometry.encoder.parameters()
                if p.grad is not None
            )
            torch.nn.utils.clip_grad_norm_(params, config.clip_norm, error_if_nonfinite=True)
            optimizer.step()
            entry: dict[str, Any] = {
                "step": step + 1,
                "loss": total,
                **components,
                **gradients,
                "encoder_gradient_norm": encoder_squared**0.5,
            }
            if (step + 1) % config.validation_every == 0 or step + 1 == config.steps:
                model.eval()
                values = []
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    for i, row in enumerate(val):
                        tensors, g = sample(row)
                        values.append(
                            float(
                                rga_loss(
                                    model,
                                    tensors,
                                    g,
                                    config.seed + 10_000 + i,
                                    config.architecture_lambda,
                                )[0]
                            )
                        )
                if not all(math.isfinite(v) for v in values):
                    raise RuntimeError("nonfinite RGA validation loss")
                entry["validation_loss"] = sum(values) / len(values)
                if config.validation_loss_ceiling is not None:
                    bad_validations = (
                        bad_validations + 1
                        if entry["validation_loss"] > config.validation_loss_ceiling
                        else 0
                    )
                    stopped = bad_validations >= config.validation_patience
                    entry["validation_guard_streak"] = bad_validations
                    entry["validation_guard_stopped"] = stopped
            torch.cuda.synchronize()
            entry.update(
                elapsed_seconds=time.perf_counter() - started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
            log.write(json.dumps(entry, allow_nan=False) + "\n")
            log.flush()
            print(json.dumps(entry), flush=True)
            if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps or stopped:
                save_checkpoint(
                    output / f"step-{step + 1:06d}",
                    model,
                    optimizer,
                    scaler,
                    step + 1,
                    {**record, "validation_guard_streak": bad_validations},
                )
            if stopped:
                break
    frozen_after = frozen_digest(base)
    write_json(
        output / "frozen-after.json",
        {
            "sha256": frozen_after,
            "unchanged": frozen_after == frozen_before,
            "parameter_gradients_absent": True,
        },
    )
    if frozen_after != frozen_before:
        raise RuntimeError("frozen Qwen/LoRA tensor bytes changed")
    write_json(
        output / "completed.json",
        {
            "steps": step + 1,
            "status": "rga_validation_guard_stopped" if stopped else "rga_training_completed",
        },
    )
