"""UI orchestration without Torch imports; real inference runs in the pinned worker."""

from __future__ import annotations

import gc
import hashlib
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

from roomcraft.gaming import GamingConfig, compose_gaming

ROOT = Path(__file__).resolve().parents[2]
LORA_SHA = "1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6"
RGA_SHA = "31625ddd4f82f10c8ff08947ffe68aa8d08682813c2b47e11cab15321ae56fca"
LORA_CONFIG_SHA = "3fa226e3f68e32f78515df1c159470156374b1cea0cdcdae92f022b6c966b01a"
RGA_CONFIG_SHA = "a0f84d8ec7c8773a7dbf4d5da2d9e52c05d48420b2988721431de9e024302537"
BASE = json.loads(Path(__file__).with_name("models.json").read_text())["qwen"]
PRESETS: dict[str, list[dict[str, str]]] = json.loads(
    Path(__file__).with_name("presets.json").read_text()
)
PRESERVATION = (
    " Preserve the original walls, windows, doors, ceiling, flooring, camera viewpoint and room "
    "perspective. Do not add, remove or move architectural openings. Place furniture on the floor "
    "at realistic scale with physically supported monitors and credible contact shadows. Keep "
    "door swings and circulation clear; fit furniture to the available space rather than changing "
    "the room. Keep existing light sources and integrate new lights naturally."
)


class AppError(RuntimeError):
    """An actionable error safe to show in the local UI."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def instruction(category: str, preset: str, extra: str) -> str:
    choices = PRESETS.get(category, [])
    selected = next((p for p in choices if p["id"] == preset), None)
    if selected is None:
        raise AppError("Choose a preset from the selected category.")
    if len(extra) > 2000:
        raise AppError("Keep custom instructions under 2,000 characters.")
    return (
        selected["instruction"]
        + (" Additional wishes: " + extra.strip() if extra.strip() else "")
        + PRESERVATION
    )


def clean_upload(source: Path, destination: Path) -> None:
    """Verify actual image bytes, reject animation/bombs, remove EXIF, preserve orientation."""
    try:
        if not source.is_file() or source.stat().st_size > 20 * 1024**2:
            raise AppError("Upload a JPEG, PNG or WebP under 20 MB.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(source) as probe:
                if (
                    probe.format not in ("JPEG", "PNG", "WEBP")
                    or getattr(probe, "n_frames", 1) != 1
                ):
                    raise AppError("Use a single JPEG, PNG or WebP photograph.")
                w, h = probe.size
                if min(w, h) < 64 or w * h > 24_000_000 or max(w, h) / min(w, h) > 8:
                    raise AppError(
                        "Use an image of at least 64 pixels per side and at most 24 megapixels."
                    )
                probe.verify()
            with Image.open(source) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                rgb.info.clear()
                rgb.save(destination, format="PNG")
    except AppError:
        raise
    except (
        OSError,
        ValueError,
        UnidentifiedImageError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise AppError(
            "This file could not be read as a photograph. Try a fresh JPEG or PNG."
        ) from exc


@dataclass(frozen=True)
class AppConfig:
    worker_python: Path = (
        ROOT
        / "environments/qwen/.venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
    )
    lora: Path = ROOT / "outputs/v06/trained/pilot/step-000100"
    rga: Path = ROOT / "runs/v09-rga-calibrated/pilot/step-000100"
    remote_session: Path | None = None
    timeout: int = 900


def require_checkpoints(config: AppConfig, model: str, use_rga: bool) -> None:
    if model not in ("base", "roomcraft"):
        raise AppError("Choose unchanged Qwen or RoomCraft LoRA.")
    if use_rga and model != "roomcraft":
        raise AppError("RoomCraft RGA requires the RoomCraft LoRA model.")
    if config.remote_session is not None:
        return  # The actual remote worker verifies its checkpoint bytes.
    for folder, expected, title, needed in (
        (config.lora, LORA_SHA, "RoomCraft LoRA", model == "roomcraft"),
        (config.rga, RGA_SHA, "RoomCraft RGA", use_rga),
    ):
        if needed and (
            not (folder / "adapter.safetensors").is_file()
            or not (folder / "config.json").is_file()
            or digest(folder / "adapter.safetensors") != expected
            or digest(folder / "config.json")
            != (LORA_CONFIG_SHA if title == "RoomCraft LoRA" else RGA_CONFIG_SHA)
        ):
            raise AppError(
                f"The verified {title} checkpoint is missing or does not match this release. Install the matching checkpoint; base Qwen has not been substituted."
            )


class AppService:
    def __init__(self, directory: Path, config: AppConfig) -> None:
        self.directory = directory.resolve()
        self.directory.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.sessions: dict[str, dict[str, dict[str, Any]]] = {}
        self.touched: dict[str, float] = {}
        self.lock = threading.Lock()

    def session(self) -> str:
        token = uuid.uuid4().hex
        self.sessions[token] = {}
        self.touched[token] = time.time()
        (self.directory / token).mkdir()
        return token

    def records(self, token: str) -> dict[str, dict[str, Any]]:
        if token not in self.sessions:
            raise AppError("This session has expired. Refresh the page to start again.")
        self.touched[token] = time.time()
        return self.sessions[token]

    def clear(self, token: str) -> None:
        self.records(token)
        folder = (self.directory / token).resolve()
        if folder.parent != self.directory:
            raise AppError("Invalid session directory.")
        shutil.rmtree(folder)
        folder.mkdir()
        self.sessions[token] = {}

    def cleanup(self, age: float = 3600) -> None:
        for token in list(self.sessions):
            if time.time() - self.touched[token] > age:
                self.clear(token)
                self.sessions.pop(token)
                self.touched.pop(token)
                (self.directory / token).rmdir()

    def checkpoint(self, model: str, use_rga: bool) -> None:
        require_checkpoints(self.config, model, use_rga)

    def generate(
        self,
        token: str,
        upload: str | None,
        category: str,
        preset: str,
        extra: str,
        model: str,
        seed: int,
        size: int,
        steps: int,
        cfg: float,
        use_rga: bool,
        gaming: GamingConfig | None = None,
    ) -> dict[str, Any]:
        self.records(token)
        if not upload:
            raise AppError("Upload a room photograph first.")
        studio = None
        if category == "Gaming setups" and gaming is not None:
            try:
                prompt, studio = compose_gaming(gaming, extra)
            except ValueError as exc:
                raise AppError(str(exc)) from exc
        else:
            prompt = instruction(category, preset, extra)
        if (
            not 0 <= seed < 2**63
            or size not in (512, 768, 1024)
            or not 1 <= steps <= 60
            or not 1 <= cfg <= 10
        ):
            raise AppError("Check the seed, resolution, steps and guidance settings.")
        self.checkpoint(model, use_rga)
        job = self.directory / token / uuid.uuid4().hex
        job.mkdir()
        try:
            clean_upload(Path(upload), job / "upload.png")
            request = {
                "model": model,
                "instruction": prompt,
                "seed": seed,
                "size": size,
                "steps": steps,
                "cfg": cfg,
                "rga": use_rga,
                "studio": studio,
            }
            (job / "request.json").write_text(json.dumps(request), encoding="utf-8")
            if self.config.remote_session is not None:
                command = [
                    sys.executable,
                    str(ROOT / "scripts/app_runpod.py"),
                    "infer",
                    "--state",
                    str(self.config.remote_session),
                    "--job",
                    str(job),
                ]
            else:
                if not self.config.worker_python.is_file():
                    raise AppError(
                        "The Qwen worker environment is missing. Install environments/qwen using the launch guide."
                    )
                command = [
                    str(self.config.worker_python),
                    str(ROOT / "scripts/app_worker.py"),
                    "--job",
                    str(job),
                    "--lora",
                    str(self.config.lora),
                    "--rga",
                    str(self.config.rga),
                ]
            with self.lock, (job / "worker.log").open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=self.config.timeout,
                    check=False,
                )
            response_path = job / "response.json"
            if not response_path.is_file():
                raise AppError(
                    "The image worker stopped before returning a result. Check the GPU connection and restart the approved worker."
                )
            response = json.loads(response_path.read_text(encoding="utf-8"))
            if result.returncode or response.get("error"):
                raise AppError(
                    response.get("error", "The GPU worker failed. Restart it and try again.")
                )
            label = (
                f"{studio['theme_definition']['name']} / {studio['monitor_count']} monitor(s)"
                if studio
                else preset
            )
            return self.accept(token, job, response["metadata"], "generated", label)
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(job)
            raise AppError(
                "Generation timed out. Check the GPU worker; a Runpod spending deadline still applies independently."
            ) from exc
        except Exception:
            # Keep the error text in memory, not uploaded photographs or failed partial results.
            shutil.rmtree(job)
            raise

    def accept(
        self, token: str, job: Path, metadata: dict[str, Any], origin: str, label: str
    ) -> dict[str, Any]:
        output = job / "result/output.png"
        with Image.open(output) as image:
            image.verify()
        with Image.open(output) as image:
            if list(image.size) != metadata["working_size"] or image.format != "PNG":
                raise AppError("The worker returned unexpected image dimensions or format.")
        if digest(output) != metadata["output_sha256"] or metadata["base"] != BASE:
            raise AppError("The worker returned an image or model identity that did not verify.")
        request_path = job / "request.json"
        if origin == "generated":
            request = json.loads(request_path.read_text())
            expected = LORA_SHA if request["model"] == "roomcraft" else None
            expected_settings = {
                "size": request["size"],
                "steps": request["steps"],
                "true_cfg_scale": request["cfg"],
                "seed": request["seed"],
                "negative_prompt": " ",
                "offload": False,
            }
            if (
                metadata["adapter_sha256"] != expected
                or metadata["settings"] != expected_settings
                or metadata["prompt"] != request["instruction"]
                or metadata["input_sha256"] != digest(job / "upload.png")
                or max(metadata["working_size"]) != request["size"]
            ):
                raise AppError(
                    "The worker returned a model, source or settings mismatch. No fallback was accepted."
                )
            if bool(metadata.get("geometry_adapter")) != request["rga"]:
                raise AppError("Geometry adapter settings did not match the request.")
            if request["rga"] and metadata["geometry_adapter"]["checkpoint_sha256"] != RGA_SHA:
                raise AppError("The worker returned an unrecognized geometry checkpoint.")
            metadata["application"] = {
                "version": "1.1.0",
                "origin": "generated",
                "adapter_version": "v06-pilot-step-000100" if expected else None,
                "studio": request.get("studio"),
            }
            (job / "result/metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
        # Match only the comparison preview to the generated dimensions; never upscale output.
        clean_upload(job / "upload.png", job / "source.png")
        with Image.open(job / "source.png") as source:
            source.resize(tuple(metadata["working_size"]), Image.Resampling.LANCZOS).save(
                job / "before.png"
            )
        record = {
            "id": job.name,
            "path": str(output),
            "before": str(job / "before.png"),
            "source": str(job / "source.png"),
            "metadata_path": str(job / "result/metadata.json"),
            "metadata": metadata,
            "origin": origin,
            "label": label,
        }
        records = self.records(token)
        records[job.name] = record
        while len(records) > 8:
            old = records.pop(next(iter(records)))
            shutil.rmtree(self.directory / token / old["id"])
        gc.collect()
        return record

    def preserve(
        self,
        token: str,
        metadata: dict[str, Any] | None,
        mask: str | None,
        foreground: str | None,
        feather: int,
    ) -> dict[str, Any]:
        """Create a separately labeled local derivative; original result stays in history."""
        from roomcraft.preservation import selective_composite

        records = self.records(token)
        if not metadata or not mask:
            raise AppError("Select a result and upload an architecture protection mask first.")
        match = next(
            (
                r
                for r in records.values()
                if r["metadata"]["output_sha256"] == metadata.get("output_sha256")
            ),
            None,
        )
        if match is None:
            raise AppError("Select a result from this session before preserving architecture.")
        job = self.directory / token / uuid.uuid4().hex
        (job / "result").mkdir(parents=True)
        try:
            shutil.copyfile(match["source"], job / "upload.png")
            clean_upload(Path(mask), job / "protection.png")
            if foreground:
                clean_upload(Path(foreground), job / "foreground.png")
            with (
                Image.open(match["source"]) as original,
                Image.open(match["path"]) as raw,
                Image.open(job / "protection.png") as protect,
            ):
                if protect.size != original.size:
                    raise AppError("Protection mask must match the original photograph dimensions.")
                fg = None
                if foreground:
                    with Image.open(job / "foreground.png") as image:
                        if image.size != original.size:
                            raise AppError(
                                "Foreground mask must match the original photograph dimensions."
                            )
                        fg = image.convert("L").resize(raw.size, Image.Resampling.NEAREST)
                result, alpha = selective_composite(
                    original.resize(raw.size, Image.Resampling.LANCZOS),
                    raw,
                    protect.convert("L").resize(raw.size, Image.Resampling.NEAREST),
                    fg,
                    feather,
                    match_lighting=True,
                )
                result.save(job / "result/output.png")
                alpha.save(job / "result/protection-alpha.png")
            meta = json.loads(json.dumps(match["metadata"]))
            meta["application"] = {**meta.get("application", {}), "origin": "selective_composite"}
            meta["selective_preservation"] = {
                "method": "inward-feathered source composite",
                "lighting_match": True,
                "feather_pixels": feather,
                "raw_output_sha256": metadata["output_sha256"],
                "mask_sha256": digest(job / "protection.png"),
                "foreground_sha256": digest(job / "foreground.png") if foreground else None,
                "alpha_sha256": digest(job / "result/protection-alpha.png"),
                "warning": "Manual mask; inspect object occlusion, seams and lighting. Not model geometry preservation.",
            }
            meta["output_sha256"] = digest(job / "result/output.png")
            (job / "result/metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            return self.accept(
                token,
                job,
                meta,
                "selective_composite",
                "Selective preservation · " + match["label"],
            )
        except Exception:
            shutil.rmtree(job)
            raise

    def example(self, token: str, model: str) -> dict[str, Any]:
        self.records(token)
        if model not in ("base", "roomcraft"):
            raise AppError("Choose a model first.")
        examples = ROOT / "docs/assets/example"
        job = self.directory / token / uuid.uuid4().hex
        (job / "result").mkdir(parents=True)
        try:
            shutil.copyfile(examples / "source.png", job / "upload.png")
            shutil.copyfile(examples / f"{model}.png", job / "result/output.png")
            shutil.copyfile(examples / f"{model}.json", job / "result/metadata.json")
            meta = json.loads((job / "result/metadata.json").read_text())
            return self.accept(token, job, meta, "saved_example", f"Saved {model} example")
        except Exception:
            shutil.rmtree(job)
            raise

    def select(self, token: str, index: int) -> dict[str, Any]:
        records = list(reversed(list(self.records(token).values())))
        if not 0 <= index < len(records):
            raise AppError("That generation is no longer in this session.")
        return records[index]

    def gallery(self, token: str) -> list[tuple[str, str]]:
        return [
            (r["path"], ("Saved example · " if r["origin"] == "saved_example" else "") + r["label"])
            for r in reversed(list(self.records(token).values()))
        ]
