"""App validation, actual saved artifacts, and failure isolation; no fake quality claims."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from roomcraft.app_service import (
    BASE,
    LORA_SHA,
    PRESETS,
    ROOT,
    AppConfig,
    AppError,
    AppService,
    clean_upload,
    digest,
    instruction,
    require_checkpoints,
)


@pytest.fixture
def service(tmp_path):
    return AppService(
        tmp_path / "app",
        AppConfig(worker_python=tmp_path / "missing-python", lora=tmp_path / "missing-lora"),
    )


def test_presets_have_six_gaming_choices_and_preservation_constraints():
    assert len(PRESETS["Gaming setups"]) == 6
    ids = [p["id"] for choices in PRESETS.values() for p in choices]
    assert len(ids) == len(set(ids))
    for category, choices in PRESETS.items():
        for p in choices:
            text = instruction(category, p["id"], "oak desk")
            assert "oak desk" in text and "windows" in text and "camera viewpoint" in text
    with pytest.raises(AppError):
        instruction("Bedrooms", "gaming_minimal", "")
    with pytest.raises(AppError):
        instruction("Gaming setups", "gaming_minimal", "x" * 2001)


def test_upload_rejects_disguised_text_animation_and_huge_dimensions(tmp_path):
    fake = tmp_path / "fake.jpg"
    fake.write_text("not a photograph")
    with pytest.raises(AppError, match="could not be read"):
        clean_upload(fake, tmp_path / "safe.png")
    animated = tmp_path / "animation.webp"
    Image.new("RGB", (64, 64)).save(
        animated, save_all=True, append_images=[Image.new("RGB", (64, 64), "red")], duration=100
    )
    with pytest.raises(AppError, match="single"):
        clean_upload(animated, tmp_path / "safe.png")
    large = tmp_path / "large.png"
    Image.new("1", (5001, 5000)).save(large)
    with pytest.raises(AppError, match="24 megapixels"):
        clean_upload(large, tmp_path / "safe.png")


def test_upload_removes_exif_and_honors_orientation(tmp_path):
    source = tmp_path / "portrait.jpg"
    exif = Image.Exif()
    exif[274] = 6
    exif[315] = "private photographer metadata"
    Image.new("RGB", (80, 120), "green").save(source, exif=exif)
    clean_upload(source, tmp_path / "clean.png")
    with Image.open(tmp_path / "clean.png") as result:
        assert result.size == (120, 80) and not result.getexif()


def test_missing_or_wrong_lora_never_falls_back_to_base(tmp_path):
    config = AppConfig(lora=tmp_path)
    require_checkpoints(config, "base", False)
    with pytest.raises(AppError, match="not been substituted"):
        require_checkpoints(config, "roomcraft", False)
    (tmp_path / "adapter.safetensors").write_bytes(b"wrong bytes")
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(AppError, match="does not match"):
        require_checkpoints(config, "roomcraft", False)
    with pytest.raises(AppError, match="requires"):
        require_checkpoints(config, "base", True)


@pytest.mark.parametrize("model,expected", [("base", None), ("roomcraft", LORA_SHA)])
def test_actual_saved_example_integrity_and_session_isolation(service, model, expected):
    token, other = service.session(), service.session()
    record = service.example(token, model)
    assert record["origin"] == "saved_example"
    assert record["metadata"]["base"] == BASE
    assert record["metadata"]["adapter_sha256"] == expected
    assert digest(Path(record["path"])) == record["metadata"]["output_sha256"]
    with Image.open(record["path"]) as image:
        assert image.size == (768, 1024) and image.format == "PNG"
    with Image.open(record["before"]) as before:
        assert before.size == (768, 1024)
    assert service.gallery(other) == []
    assert service.select(token, 0)["id"] == record["id"]
    with pytest.raises(AppError):
        service.select(other, 0)


def test_history_limit_expiry_and_clear_only_remove_app_files(service, tmp_path):
    token = service.session()
    protected = tmp_path / "keep.txt"
    protected.write_text("unrelated project data")
    first = service.example(token, "base")
    for _ in range(8):
        service.example(token, "roomcraft")
    assert len(service.gallery(token)) == 8 and not Path(first["path"]).exists()
    service.clear(token)
    assert service.gallery(token) == [] and protected.read_text() == "unrelated project data"
    service.touched[token] = 0
    service.cleanup()
    with pytest.raises(AppError, match="expired"):
        service.records(token)


def test_worker_missing_and_timeout_clean_uploaded_photos(service, monkeypatch, tmp_path):
    token = service.session()
    args = (
        token,
        str(ROOT / "docs/assets/example/source.png"),
        "Gaming setups",
        "gaming_minimal",
        "",
        "base",
        42,
        1024,
        40,
        4,
        False,
    )
    with pytest.raises(AppError, match="environment is missing"):
        service.generate(*args)
    assert not list((service.directory / token).iterdir())
    python = tmp_path / "python.exe"
    python.write_text("only an existence fixture; never executed")
    service.config = AppConfig(worker_python=python)

    def timeout(*a, **kw):
        raise subprocess.TimeoutExpired("unit-test worker", 1)

    monkeypatch.setattr(subprocess, "run", timeout)
    with pytest.raises(AppError, match="timed out"):
        service.generate(*args)
    assert not list((service.directory / token).iterdir())


def test_wrong_adapter_receipt_is_rejected(service):
    token = service.session()
    record = service.example(token, "roomcraft")
    job = Path(record["path"]).parents[1]
    request = {
        "model": "base",
        "instruction": record["metadata"]["prompt"],
        "seed": 42,
        "size": 1024,
        "steps": 40,
        "cfg": 4,
        "rga": False,
    }
    (job / "request.json").write_text(json.dumps(request))
    with pytest.raises(AppError, match="mismatch"):
        service.accept(token, job, record["metadata"], "generated", "unit-test")


def test_fresh_paid_approval_and_runtime_binding_are_required(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import app_runpod

    with pytest.raises(ValueError, match="approval"):
        app_runpod.approval(ROOT / "configs/app_approval.example.json")
    app_runpod.control.STATE = tmp_path
    record = {"podId": "roomcrafttest", "namePrefix": "roomcraft-mvp-", "deadlineUnix": 0}
    (tmp_path / "session.json").write_text(json.dumps(record))
    (tmp_path / "app-ready.json").write_text(
        json.dumps({"podId": "roomcrafttest", "runtime": "/workspace/unrelated-project"})
    )
    with pytest.raises(ValueError, match="binding"):
        app_runpod.infer(tmp_path / ("a" * 32))
    (tmp_path / "app-ready.json").write_text(
        json.dumps(
            {
                "podId": "roomcrafttest",
                "runtime": "/workspace/RoomCraftAI/app-runtime/roomcrafttest",
            }
        )
    )
    with pytest.raises(ValueError, match="eleven minutes"):
        app_runpod.infer(tmp_path / ("a" * 32))


def test_cloud_budget_gate_precedes_any_provisioning(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import app_runpod

    monkeypatch.setattr(app_runpod.control, "STATE", tmp_path)
    monkeypatch.setattr(app_runpod, "approval", lambda path: {"budget_usd": 1, "max_minutes": 60})
    monkeypatch.setattr(
        app_runpod,
        "quote",
        lambda volume: {"combinedHourlyUsd": 4, "networkVolumeHourlyUsd": 0.025},
    )
    calls = []
    monkeypatch.setattr(app_runpod.control, "api", lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="exceed"):
        app_runpod.start(tmp_path / "approval.json", "only-roomcraft")
    assert not calls and not (tmp_path / "session.json").exists()


def test_runpod_api_failure_does_not_upload_or_modify_resources(tmp_path, monkeypatch):
    import time

    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    import app_runpod

    monkeypatch.setattr(app_runpod.control, "STATE", tmp_path)
    record = {
        "podId": "onlyroomcraft",
        "namePrefix": "roomcraft-mvp-",
        "deadlineUnix": time.time() + 1800,
    }
    (tmp_path / "session.json").write_text(json.dumps(record))
    (tmp_path / "app-ready.json").write_text(
        json.dumps(
            {
                "podId": "onlyroomcraft",
                "runtime": "/workspace/RoomCraftAI/app-runtime/onlyroomcraft",
            }
        )
    )
    (tmp_path / "guard-status.json").write_text(
        json.dumps({"podId": "onlyroomcraft", "epoch": time.time()})
    )
    calls = []

    def unavailable(method, path):
        calls.append((method, path))
        return 503, None

    monkeypatch.setattr(app_runpod.control, "api", unavailable)
    with pytest.raises(RuntimeError, match="HTTP 503"):
        app_runpod.infer(tmp_path / ("b" * 32))
    assert calls == [("GET", "pods/onlyroomcraft")]
