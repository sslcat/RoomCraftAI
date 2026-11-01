"""Replay genuine public off/on artifacts through request-identity acceptance."""

import copy
import json
import shutil

import pytest

from roomcraft.app_service import ROOT, AppConfig, AppError, AppService

DEMO = ROOT / "docs/assets/rga-underwater"


@pytest.mark.parametrize("enabled", [False, True])
def test_genuine_rga_result_acceptance_and_wrong_switch_rejection(tmp_path, enabled):
    service = AppService(tmp_path / "app", AppConfig())
    token = service.session()
    folder = service.directory / token / ("a" * 32)
    (folder / "result").mkdir(parents=True)
    arm = "rga-on" if enabled else "rga-off"
    metadata = json.loads((DEMO / (arm + ".json")).read_text())
    shutil.copy2(DEMO / (arm + ".png"), folder / "result/output.png")
    shutil.copy2(DEMO / "source.png", folder / "upload.png")
    request = {
        "model": "roomcraft",
        "rga": enabled,
        "size": 1024,
        "steps": 40,
        "cfg": 4.0,
        "seed": 42,
        "instruction": metadata["prompt"],
    }
    (folder / "request.json").write_text(json.dumps(request))
    accepted = service.accept(token, folder, copy.deepcopy(metadata), "generated", arm)
    assert bool(accepted["metadata"].get("geometry_adapter")) == enabled
    request["rga"] = not enabled
    (folder / "request.json").write_text(json.dumps(request))
    with pytest.raises(AppError, match="Geometry adapter settings"):
        service.accept(token, folder, copy.deepcopy(metadata), "generated", arm)
