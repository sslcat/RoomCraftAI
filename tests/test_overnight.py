"""No network requests or cloud resources: strict budget/deadline and authorization contracts."""

import copy
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from roomcraft.overnight import budget_total, run_on_pod, validate_plan


def timestamp(seconds):
    return datetime.fromtimestamp(seconds, UTC).isoformat()


def valid_record():
    plan = json.loads(Path("configs/overnight_v06.json").read_text())
    plan.update(
        deployment_authorized=True,
        capacity_confirmed=True,
        blackwell_runtime_verified=True,
        pod_scoped_permission_verified=True,
        storage_cleanup_arranged=True,
        pod_id="new-roomcraft-only",
        pod_created_utc=timestamp(1000),
    )
    plan["quote"].update(authenticated=True, checked_utc=timestamp(1100), tax_and_other_usd=5)
    plan["cloud_shutdown"].update(
        independent_of_pod_and_windows=True,
        real_stop_test_verified=True,
        pod_id=plan["pod_id"],
        credential_resource_ids=[plan["pod_id"]],
        deadline_utc=timestamp(1000 + 4 * 3600 - 300),
        readback_reference="TEST-ONLY-provider-receipt",
        stop_test_reference="TEST-ONLY-stop-receipt",
    )
    return plan


def test_current_overnight_plan_is_blocked():
    plan = json.loads(Path("configs/overnight_v06.json").read_text())
    with pytest.raises(ValueError, match="not deployment authorization"):
        validate_plan(plan, 1200)


def test_budget_full_storage_and_other_charges():
    plan = valid_record()
    assert budget_total(plan) == Decimal("32.48")
    plan["quote"]["tax_and_other_usd"] = None
    with pytest.raises(ValueError, match="all charges"):
        budget_total(plan)
    plan["quote"]["tax_and_other_usd"] = 13
    with pytest.raises(ValueError, match="exceeds"):
        validate_plan(plan, 1200)


@pytest.mark.parametrize(
    "field",
    [
        "capacity_confirmed",
        "blackwell_runtime_verified",
        "pod_scoped_permission_verified",
        "storage_cleanup_arranged",
    ],
)
def test_unverified_prerequisites_block(field):
    plan = valid_record()
    plan[field] = False
    with pytest.raises(ValueError, match=field):
        validate_plan(plan, 1200)


def test_deadline_never_resets_with_process_start():
    plan = valid_record()
    assert validate_plan(plan, 1200) == 13900
    assert validate_plan(plan, 1300) == 13800
    plan["cloud_shutdown"]["deadline_utc"] = timestamp(1000 + 4 * 3600 + 1)
    with pytest.raises(ValueError, match="deadline"):
        validate_plan(plan, 1200)


def test_local_or_pod_only_watchdog_is_insufficient():
    plan = valid_record()
    plan["cloud_shutdown"]["independent_of_pod_and_windows"] = False
    with pytest.raises(ValueError, match="independent"):
        validate_plan(plan, 1200)
    plan = valid_record()
    plan["cloud_shutdown"]["credential_resource_ids"].append("another-pod")
    with pytest.raises(ValueError, match="restricted server-side"):
        validate_plan(plan, 1200)


def test_wrong_region_global_volume_or_stale_quote_block():
    for field, value in (("data_center", "US-TX-3"), ("global_volume", "forbidden")):
        plan = valid_record()
        plan[field] = value
        with pytest.raises(ValueError):
            validate_plan(plan, 1200)
    plan = valid_record()
    with pytest.raises(ValueError, match="15 minutes"):
        validate_plan(plan, 2500)


def test_pod_identity_checked_before_api_or_stop(monkeypatch):
    monkeypatch.setenv("RUNPOD_POD_ID", "different-pod")
    with pytest.raises(ValueError, match="exact newly approved"):
        run_on_pod(valid_record())


def test_cloud_supervisor_failure_stops_only_approved_pod(monkeypatch):
    from roomcraft import overnight

    plan = valid_record()
    monkeypatch.setattr(overnight.sys, "platform", "linux")
    monkeypatch.setenv("RUNPOD_POD_ID", plan["pod_id"])
    monkeypatch.setenv("ROOMCRAFT_POD_SCOPED_KEY", "dummy")
    stopped = []
    monkeypatch.setattr(
        overnight.PodStopper,
        "inspect",
        lambda self: {"id": plan["pod_id"], "name": plan["pod_name"]},
    )
    monkeypatch.setattr(overnight.PodStopper, "stop", lambda self: stopped.append(self.pod_id))
    bad = copy.deepcopy(plan)
    bad["deployment_authorized"] = False
    with pytest.raises(ValueError, match="NOT READY"):
        run_on_pod(bad)
    assert stopped == [plan["pod_id"]]
