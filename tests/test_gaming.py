"""Theme semantics and UI-to-worker contracts; transport doubles are not quality evidence."""

import itertools
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from roomcraft.app_service import BASE, LORA_SHA, ROOT, AppConfig, AppError, AppService, digest
from roomcraft.gaming import (
    DESKS,
    LIGHTING,
    MONITORS,
    SETUPS,
    THEMES,
    TOWERS,
    GamingConfig,
    compose_gaming,
)


@pytest.mark.parametrize("theme", list(THEMES))
def test_every_theme_has_full_provenance_free_definition_and_reaches_instruction(theme):
    definition = THEMES[theme]
    assert definition["id"] == theme and len(definition["palette"]) == 3
    assert all(c.startswith("#") and len(c) == 7 for c in definition["palette"])
    text, receipt = compose_gaming(GamingConfig(theme=theme))
    for key in (
        "name",
        "palette_names",
        "materials",
        "equipment",
        "lighting",
        "constraints",
        "negative_constraints",
        "decorations",
    ):
        assert definition[key] in text
    assert receipt["theme_definition"] == definition
    assert "original walls, windows, doors" in text and "measured fit" in text


def test_setup_equipment_lighting_cross_product_is_deterministic_and_consistent():
    for setup, desk, monitor, light in itertools.product(SETUPS, DESKS, MONITORS, LIGHTING):
        config = GamingConfig(setup=setup, desk=desk, monitors=monitor, lighting=light)
        text, receipt = compose_gaming(config)
        assert (text, receipt) == compose_gaming(config)
        assert DESKS[desk] in text and MONITORS[monitor][1] in text and LIGHTING[light] in text
        assert receipt["monitor_count"] == MONITORS[monitor][0]
        if setup == "creator":
            assert "microphone boom" in text


def test_every_control_changes_real_prompt_and_receipt():
    default = GamingConfig()
    baseline, _ = compose_gaming(default)
    options = {
        "setup": SETUPS,
        "theme": {**THEMES, "custom": {}},
        "desk": DESKS,
        "tower": TOWERS,
        "monitors": MONITORS,
        "lighting": LIGHTING,
    }
    for field, choices in options.items():
        for value in choices:
            config = replace(default, **{field: value}, custom_theme="Retro observatory")
            prompt, receipt = compose_gaming(config)
            assert receipt["configuration"][field] == value
            if value != getattr(default, field):
                assert prompt != baseline
    for field in (
        "chair",
        "peripherals",
        "headset",
        "speakers",
        "cables",
        "shelving",
        "decorations",
    ):
        prompt, receipt = compose_gaming(replace(default, **{field: not getattr(default, field)}))
        assert prompt != baseline
        assert receipt["configuration"][field] is not getattr(default, field)


def test_optional_equipment_and_lighting_overrides_do_not_silently_add_objects():
    text, _ = compose_gaming(
        GamingConfig(theme="underwater_realm", shelving=False, decorations=False, lighting="none")
    )
    assert "coral-inspired freestanding shelving" not in text
    assert "Do not add shelving" in text and "No RGB or colored lighting" in text
    assert "no actual water" in text and "Keep the room dry" in text
    assert "ignore colored theme-light suggestions" in text


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theme": "fake"},
        {"setup": "fake"},
        {"desk": "fake"},
        {"monitors": "fake"},
        {"tower": "fake"},
        {"lighting": "fake"},
        {"chair": "yes"},
        {"theme": "custom"},
        {"custom_theme": "x" * 501},
    ],
)
def test_invalid_options_are_actionable(kwargs):
    with pytest.raises(ValueError):
        compose_gaming(GamingConfig(**kwargs))
    with pytest.raises(ValueError, match="2,000"):
        compose_gaming(GamingConfig(), "x" * 2001)


@pytest.mark.parametrize(
    "remote,model,rga",
    [
        (False, "base", False),
        (False, "roomcraft", False),
        (False, "roomcraft", True),
        (True, "base", False),
        (True, "roomcraft", True),
    ],
)
def test_unit_transport_receives_composed_prompt_and_round_trips_configuration(
    tmp_path, monkeypatch, remote, model, rga
):
    """Historical pixels stand in ONLY for transport tests; no new inference is claimed."""
    service = AppService(
        tmp_path / "sessions",
        AppConfig(worker_python=Path(sys.executable), remote_session=tmp_path if remote else None),
    )
    token = service.session()
    config = GamingConfig(
        theme="pirates_cove",
        setup="dual",
        tower="under_desk",
        monitors="dual_stacked",
        shelving=True,
    )
    calls = []
    monkeypatch.setattr(service, "checkpoint", lambda *args: None)

    def worker(command, **kwargs):
        import shutil

        calls.append(command)
        job = Path(command[command.index("--job") + 1])
        req = json.loads((job / "request.json").read_text())
        assert req["instruction"] == compose_gaming(config, "Keep the closet clear")[0]
        assert req["studio"]["configuration"]["tower"] == "under_desk"
        assert req["model"] == model and req["rga"] is rga
        (job / "result").mkdir()
        shutil.copyfile(ROOT / "docs/assets/example/base.png", job / "result/output.png")
        meta = json.loads((ROOT / "docs/assets/example/base.json").read_text())
        meta.update(
            prompt=req["instruction"],
            input_sha256=digest(job / "upload.png"),
            adapter_sha256=LORA_SHA if model == "roomcraft" else None,
            base=BASE,
        )
        if rga:
            from roomcraft.app_service import RGA_SHA

            meta["geometry_adapter"] = {"checkpoint_sha256": RGA_SHA}
        (job / "result/metadata.json").write_text(json.dumps(meta))
        (job / "response.json").write_text(json.dumps({"metadata": meta}))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", worker)
    result = service.generate(
        token,
        str(ROOT / "docs/assets/example/source.png"),
        "Gaming setups",
        "gaming_rgb",
        "Keep the closet clear",
        model,
        42,
        1024,
        40,
        4,
        rga,
        config,
    )
    assert ("app_runpod.py" if remote else "app_worker.py") in " ".join(calls[0])
    saved = json.loads(Path(result["metadata_path"]).read_text())
    assert saved["application"]["studio"] == compose_gaming(config, "Keep the closet clear")[1]
    assert saved["application"]["version"] == "1.1.0"
    assert service.select(token, 0)["metadata"] == saved
    assert result["label"] == "Pirate's Cove / 2 monitor(s)"
    service.clear(token)
    assert not service.gallery(token)


def test_invalid_gaming_configuration_never_starts_worker(tmp_path, monkeypatch):
    service = AppService(tmp_path / "sessions", AppConfig())
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("invalid configuration reached worker")
    )
    with pytest.raises(AppError, match="custom theme"):
        service.generate(
            service.session(),
            "anything.png",
            "Gaming setups",
            "gaming_rgb",
            "",
            "base",
            42,
            1024,
            40,
            4,
            False,
            GamingConfig(theme="custom"),
        )
