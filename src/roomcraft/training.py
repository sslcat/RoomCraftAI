"""Single-GPU SDXL inpainting LoRA trainer; no upstream training-script subprocesses."""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from roomcraft.checkpoint import resume_checkpoint, save_checkpoint
from roomcraft.common import environment, model_spec, seed_all, sha256, write_json
from roomcraft.data import RoomDataset
from roomcraft.lora import inject_lora, parameter_report
from roomcraft.objective import inpaint_input, region_loss


@dataclass
class TrainConfig:
    size: int = 512
    steps: int = 1000
    rank: int = 8
    alpha: float = 8.0
    learning_rate: float = 0.0001
    accumulation: int = 4
    precision: str = "fp16"
    seed: int = 42
    outside_weight: float = 0.25
    checkpoint_every: int = 100
    validate_every: int = 100
    validation_samples: int = 8
    gradient_checkpointing: bool = True
    allow_fixture: bool = False

    def validate(self) -> None:
        if self.size < 64 or self.size % 64 or self.size > 2048:
            raise ValueError("size must be a multiple of 64 in [64,2048]")
        if (
            min(
                self.steps,
                self.rank,
                self.accumulation,
                self.checkpoint_every,
                self.validate_every,
                self.validation_samples,
            )
            < 1
        ):
            raise ValueError("counts must be positive")
        if self.precision not in {"fp16", "bf16", "fp32"}:
            raise ValueError("precision must be fp16, bf16 or fp32")
        if self.learning_rate <= 0 or self.alpha <= 0 or not 0 <= self.outside_weight <= 1:
            raise ValueError("invalid learning rate, alpha or loss weight")


def example_loss(
    pipe: Any,
    scheduler: Any,
    example: dict[str, Any],
    config: TrainConfig,
    dtype: torch.dtype,
    seed: int,
) -> tuple[torch.Tensor, dict[str, list[int]]]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    source = example["source"].unsqueeze(0).to("cuda", torch.float32)
    target = example["target"].unsqueeze(0).to("cuda", torch.float32)
    mask = example["mask"].unsqueeze(0).to("cuda", torch.float32)
    with torch.no_grad():
        # Keep the original VAE in float32 to prevent fp16 encoding overflow.
        latent = (
            pipe.vae.encode(target).latent_dist.sample(generator=generator)
            * pipe.vae.config.scaling_factor
        )
        masked = (
            pipe.vae.encode(source * (1 - mask)).latent_dist.sample(generator=generator)
            * pipe.vae.config.scaling_factor
        )
        latent, masked = latent.to(dtype), masked.to(dtype)
        small_mask = F.interpolate(mask, size=latent.shape[-2:], mode="nearest").to(dtype)
        embeds, _, pooled, _ = pipe.encode_prompt(
            prompt=example["prompt"], device="cuda", do_classifier_free_guidance=False
        )
        noise = torch.randn(latent.shape, generator=generator, device="cuda", dtype=dtype)
        timestep = torch.randint(
            0, scheduler.config.num_train_timesteps, (1,), generator=generator, device="cuda"
        )
        noisy = scheduler.add_noise(latent, noise, timestep)
        height, width = source.shape[-2:]
        time_ids = torch.tensor([[height, width, 0, 0, height, width]], device="cuda", dtype=dtype)
    model_input = inpaint_input(noisy, small_mask, masked)
    with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
        predicted = pipe.unet(
            model_input,
            timestep,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": time_ids},
        ).sample
    if scheduler.config.prediction_type == "epsilon":
        objective = noise
    elif scheduler.config.prediction_type == "v_prediction":
        objective = scheduler.get_velocity(latent, noise, timestep)
    else:
        raise ValueError("unsupported scheduler prediction type")
    shapes = {
        "source": list(source.shape),
        "latent": list(latent.shape),
        "unet_input": list(model_input.shape),
        "prediction": list(predicted.shape),
        "text_embeddings": list(embeds.shape),
        "pooled_text": list(pooled.shape),
        "time_ids": list(time_ids.shape),
    }
    return region_loss(predicted, objective, small_mask, config.outside_weight), shapes


def train(manifest: Path, output: Path, config: TrainConfig, resume: Path | None = None) -> None:
    from diffusers import DDPMScheduler, StableDiffusionXLInpaintPipeline

    config.validate()
    train_data = RoomDataset(manifest, "train", config.size, config.allow_fixture)
    val_data = RoomDataset(manifest, "val", config.size, config.allow_fixture)
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for full SDXL training; CPU tests exercise the training primitives"
        )
    if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU does not support bf16")
    if output.exists() and any(output.iterdir()):
        raise ValueError("use a new output directory, including when resuming")
    output.mkdir(parents=True, exist_ok=True)
    seed_all(config.seed)
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[config.precision]
    spec = model_spec("inpaint")
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        spec["repo"],
        revision=spec["revision"],
        variant="fp16",
        torch_dtype=dtype,
        use_safetensors=True,
        add_watermarker=False,
    ).to("cuda")
    for module in (pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False).eval()
    pipe.vae.to(dtype=torch.float32)
    pipe.vae.enable_tiling()
    inject_lora(pipe.unet, config.rank, config.alpha)
    if config.gradient_checkpointing:
        pipe.unet.enable_gradient_checkpointing()
    pipe.unet.train()
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    parameters = [p for p in pipe.unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")
    saved_config = {**asdict(config), "base": spec, "manifest_sha256": sha256(manifest)}
    start = resume_checkpoint(resume, pipe.unet, optimizer, scaler, saved_config) if resume else 0
    if start >= config.steps:
        raise ValueError("steps must exceed resumed step")
    write_json(output / "config.json", saved_config)
    write_json(output / "environment.json", environment())
    report = parameter_report(pipe.unet)
    write_json(output / "parameters.json", report)
    print(f"Trainable parameters: {report['trainable']:,} / {report['total']:,}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    for step in range(start, config.steps):
        began = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        # Step-derived RNG makes restart independent of skipped DataLoader batches.
        seed_all(config.seed + step)
        for micro in range(config.accumulation):
            sample_seed = config.seed + step * config.accumulation + micro
            index = int(
                torch.randint(
                    len(train_data), (1,), generator=torch.Generator().manual_seed(sample_seed)
                )
            )
            loss, shapes = example_loss(
                pipe, scheduler, train_data[index], config, dtype, sample_seed
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite loss")
            total_loss += loss.detach().item() / config.accumulation
            scaler.scale(loss / config.accumulation).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        grad_max = max(p.grad.detach().abs().max().item() for p in parameters if p.grad is not None)
        scaler.step(optimizer)
        scaler.update()
        record: dict[str, Any] = {
            "step": step + 1,
            "loss": total_loss,
            "gradient_norm_before_clip": grad_norm.item(),
            "gradient_max_after_clip": grad_max,
        }
        if step == start:
            write_json(output / "tensor_shapes.json", shapes)
        if (step + 1) % config.validate_every == 0 or step + 1 == config.steps:
            pipe.unet.eval()
            with torch.no_grad():
                losses = [
                    example_loss(
                        pipe, scheduler, val_data[i], config, dtype, 1000000 + config.seed + i
                    )[0].item()
                    for i in range(min(len(val_data), config.validation_samples))
                ]
            record["validation_loss"] = sum(losses) / len(losses)
            pipe.unet.train()
        torch.cuda.synchronize()
        record.update(
            seconds=time.perf_counter() - began,
            peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        with (output / "metrics.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps(record), flush=True)
        if (step + 1) % config.checkpoint_every == 0 or step + 1 == config.steps:
            save_checkpoint(
                output / f"step-{step + 1:06d}",
                pipe.unet,
                optimizer,
                scaler,
                step + 1,
                saved_config,
            )
