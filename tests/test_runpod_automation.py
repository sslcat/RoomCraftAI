"""Lifecycle ownership and authentication regression tests, with no cloud requests."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def control(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    module = importlib.import_module("roomcraft_runpod")
    monkeypatch.setattr(module, "STATE", tmp_path)
    return module


def record():
    return {"podId": "roomcraft-test", "name": "roomcraft-v06-test", "volumeId": "dedicated"}


def pod():
    return {
        "id": "roomcraft-test",
        "name": "roomcraft-v06-test",
        "mounts": {"network": [{"volumeId": "dedicated", "path": "/workspace"}]},
        "actions": ["terminate"],
    }


@pytest.mark.parametrize("change", ["name", "id", "volume", "global"])
def test_identity_blocks_unrelated_resources(control, change):
    value = pod()
    if change in ("name", "id"):
        value[change] = "unrelated"
    elif change == "volume":
        value["mounts"]["network"][0]["volumeId"] = "unrelated"
    else:
        value["mounts"]["global"] = [{"volumeId": "forbidden"}]
    with pytest.raises(ValueError, match="identity uncertain"):
        control.identity(value, record())


def test_termination_needs_identity_and_fresh_404(control, monkeypatch):
    calls = []
    responses = iter([(200, pod()), (204, None), (404, None)])

    def api(method, path):
        calls.append((method, path))
        return next(responses)

    monkeypatch.setattr(control, "api", api)
    assert control.terminate(record()) is True
    assert [method for method, _ in calls] == ["GET", "DELETE", "GET"]
    assert all(path == "pods/roomcraft-test" for _, path in calls)


def test_identity_failure_never_sends_delete(control, monkeypatch):
    calls = []

    def api(method, path):
        calls.append(method)
        return 401, None

    monkeypatch.setattr(control, "api", api)
    with pytest.raises(RuntimeError, match="no deletion sent"):
        control.terminate(record())
    assert calls == ["GET"]


@pytest.mark.parametrize("version", ["v07", "v09", "v10-causality", "mvp"])
def test_version_identity_remains_exact_and_requires_approved_prefix(control, version):
    r = {**record(), "name": f"roomcraft-{version}-test", "namePrefix": f"roomcraft-{version}-"}
    p = {**pod(), "name": r["name"]}
    control.identity(p, r)
    with pytest.raises(ValueError, match="identity uncertain"):
        control.identity({**p, "id": "another-pod"}, r)
    with pytest.raises(ValueError, match="identity uncertain"):
        control.identity(p, {**r, "namePrefix": "roomcraft-"})


def test_reused_credential_representation_is_redacted(control):
    from runpod_credential_source import RunpodCredential

    assert "sensitive-test-value" not in repr(RunpodCredential("unit-test", "sensitive-test-value"))


def test_running_pod_without_ssh_is_a_retryable_startup_state(control, monkeypatch):
    r = {**record(), "gpuId": "NVIDIA H100 PCIe", "dataCenter": "US-KS-2", "combinedHourlyUsd": 2.9}
    value = {
        **pod(),
        "status": "RUNNING",
        "gpu": {"id": r["gpuId"], "count": 1, "memory": 188},
        "dataCenterId": "US-KS-2",
        "cudaVersion": "13.2",
        "cost": 2.89,
        "ssh": {"direct": None},
    }
    monkeypatch.setattr(control, "api", lambda *a: (200, value))
    with pytest.raises(RuntimeError, match="still starting"):
        control.transport(r)


def test_pilot_forecast_requires_two_finite_observed_smoke_updates(control):
    module = importlib.import_module("v06_pilot_budget")
    assert module.forecast([10, 9]) == pytest.approx(1825)
    for values in ([], [10], [10, float("nan")], [0, 10]):
        with pytest.raises(ValueError, match="two measured finite"):
            module.forecast(values)
