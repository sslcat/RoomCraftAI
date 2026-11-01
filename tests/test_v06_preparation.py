"""Free checks of split preservation, budget handling, and stop behavior; no cloud mutations."""

import copy
import io
import json
import runpy
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from roomcraft.runpod_guard import PodStopper, allowed_seconds, supervise

ROOT = Path(__file__).resolve().parents[1]


def test_split_correction_keeps_old_holdouts_and_never_changes_images():
    functions = runpy.run_path(str(ROOT / "scripts/prepare_v06_dataset.py"))
    rows = [
        {
            "group_id": f"family_{i:02d}",
            "id": f"family_{i:02d}_v{j:02d}",
            "source": f"original/{i}/{j}.png",
            "split": "validation" if i in (4, 14) else "test" if i in (5, 19) else "train",
        }
        for i in range(20)
        for j in range(5)
    ]
    before = copy.deepcopy(rows)
    revised = functions["corrected_rows"](rows)
    assert rows == before
    assert [sum(r["split"] == s for r in revised) for s in ("train", "validation", "test")] == [
        70,
        15,
        15,
    ]
    for old, new in zip(rows, revised, strict=True):
        assert {k: v for k, v in old.items() if k != "split"} == {
            k: v for k, v in new.items() if k != "split"
        }
        if old["split"] != "train":
            assert old == new
    with pytest.raises(ValueError, match="previously training"):
        functions["corrected_rows"](revised)


def test_smoke_resume_configs_only_change_steps():
    configs = [
        json.loads((ROOT / f"configs/qwen_train_v06_{name}.json").read_text())
        for name in ("first", "smoke")
    ]
    assert [c.pop("steps") for c in configs] == [1, 2]
    assert configs[0] == configs[1]
    assert configs[0]["rank"] == 8 and configs[0]["accumulation"] == 1
    pilot = json.loads((ROOT / "configs/qwen_train_v06_pilot.json").read_text())
    assert pilot["steps"] == 100 and pilot["rank"] == 8 and pilot["accumulation"] == 4


def test_budget_includes_prior_allocation_time_and_storage():
    plan = json.loads((ROOT / "configs/v06_approval_template.json").read_text())
    with pytest.raises(ValueError, match="approval"):
        allowed_seconds(plan, 1000)
    plan.update(
        owner_approved=True, billing_started_utc=datetime.fromtimestamp(1000, UTC).isoformat()
    )
    assert allowed_seconds(plan, 4600) == 3 * 3600
    assert allowed_seconds(plan, 1000 + 5 * 3600) == 0
    plan["budget_usd"] = 26
    assert allowed_seconds(plan, 1000) == 1800  # 26 - 17.5 storage - 6.5 reserve = $2
    plan["pod_hourly_ceiling_usd"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        allowed_seconds(plan, 1000)


def test_stop_uses_exact_v2_identity_and_verifies_readback(monkeypatch):
    calls = []
    states = ["RUNNING", "RUNNING", "EXITED"]

    def response(request, timeout):
        calls.append((request.full_url, request.method, request.data))
        assert timeout == 15
        return io.BytesIO(
            json.dumps(
                {
                    "id": "testpod",
                    "name": "roomcraft-v06-test",
                    "status": states.pop(0),
                    "actions": ["stop"],
                }
            ).encode()
        )

    monkeypatch.setattr("urllib.request.urlopen", response)
    stopped = PodStopper("testpod", "roomcraft-v06-test", "dummy-not-a-credential").stop(delay=0)
    assert stopped["status"] == "EXITED"
    assert [c[1] for c in calls] == ["GET", "POST", "GET"]
    assert calls[1] == (
        "https://api.runpod.io/v2/pods/testpod/action",
        "POST",
        b'{"action":"stop"}',
    )


def test_stop_refuses_wrong_identity_and_does_not_fake_success(monkeypatch):
    stopper = PodStopper("testpod", "roomcraft-v06-test", "dummy")
    monkeypatch.setattr(stopper, "request", lambda *a, **k: {"id": "brasik", "name": "other"})
    with pytest.raises(ValueError, match="identity"):
        stopper.stop(delay=0)
    monkeypatch.setattr(
        stopper,
        "request",
        lambda *a, **k: {
            "id": "testpod",
            "name": "roomcraft-v06-test",
            "status": "STARTING",
            "actions": [],
        },
    )
    with pytest.raises(RuntimeError, match="STOP NOT VERIFIED"):
        stopper.stop(attempts=2, delay=0)


def test_stop_recovers_from_transient_network_failure(monkeypatch):
    stopper = PodStopper("testpod", "roomcraft-v06-test", "dummy")
    calls = []

    def response(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise OSError("temporary network outage")
        return {"id": "testpod", "name": "roomcraft-v06-test", "status": "EXITED"}

    monkeypatch.setattr(stopper, "request", response)
    assert stopper.stop(delay=0)["attempts"] == 2


@pytest.mark.parametrize("exit_code", [0, 3])
def test_actual_child_completion_always_stops(exit_code):
    stopped = []
    assert (
        supervise(
            [sys.executable, "-c", f"raise SystemExit({exit_code})"],
            5,
            lambda: stopped.append(True),
        )
        == exit_code
    )
    assert stopped == [True]


def test_actual_child_timeout_and_launch_failure_always_stop():
    stopped = []
    with pytest.raises(TimeoutError):
        supervise(
            [sys.executable, "-c", "import time; time.sleep(30)"], 0.1, lambda: stopped.append(True)
        )
    with pytest.raises(OSError):
        supervise(
            ["roomcraft-deliberately-nonexistent-executable"], 5, lambda: stopped.append(True)
        )
    assert stopped == [True, True]


@pytest.mark.parametrize("problem", ["wrong_volume", "global_attached", "over_price"])
def test_main_stops_approved_pod_on_failed_launch_gate(tmp_path, monkeypatch, problem):
    from roomcraft.runpod_guard import main

    plan = json.loads((ROOT / "configs/v06_approval_template.json").read_text())
    plan.update(
        owner_approved=True,
        pod_id="testpod",
        pod_name="roomcraft-v06-test",
        network_volume_id="dedicated",
        billing_started_utc=datetime.now(UTC).isoformat(),
    )
    approval = tmp_path / "approval.json"
    approval.write_text(json.dumps(plan))
    log = tmp_path / "log.jsonl"
    monkeypatch.setenv("RUNPOD_API_KEY", "dummy")
    monkeypatch.setattr(
        sys,
        "argv",
        ["guard", "--approval", str(approval), "--log", str(log), "--", "never-launched"],
    )
    pod = {
        "id": "testpod",
        "name": "roomcraft-v06-test",
        "status": "RUNNING",
        "actions": ["stop"],
        "cost": 2.89,
        "mounts": {"network": [{"volumeId": "dedicated", "path": "/workspace"}]},
        "createdAt": plan["billing_started_utc"],
    }
    if problem == "wrong_volume":
        pod["mounts"]["network"][0]["volumeId"] = "brasik"
    elif problem == "global_attached":
        pod["mounts"]["global"] = [{"volumeId": "brasik", "path": "/workspace-global"}]
    else:
        pod["cost"] = 99
    stopped = []
    monkeypatch.setattr(PodStopper, "inspect", lambda self: pod)
    monkeypatch.setattr(
        PodStopper, "stop", lambda self: stopped.append(self.pod_id) or {"status": "EXITED"}
    )
    with pytest.raises(ValueError):
        main()
    assert stopped == ["testpod"]
    assert "stop_verified" in log.read_text() and "dummy" not in log.read_text()
