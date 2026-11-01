"""Fail-closed overnight gate and Pod-resident supervisor; never creates paid resources.

This supervisor is independent of Windows, but not of Pod/host failure. A separately
verified cloud deadline is mandatory. Evidence fields are operator audit records,
not cryptographic proof or a substitute for a real provider stop test.
"""

import argparse
import json
import math
import os
import signal
import sys
import time
from datetime import datetime
from decimal import ROUND_CEILING, Decimal
from pathlib import Path
from typing import Any

from roomcraft.runpod_guard import PodStopper, supervise


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("UTC/offset timestamp required")
    return parsed.timestamp()


def budget_total(plan: dict[str, Any]) -> Decimal:
    quote = plan["quote"]
    numbers = [
        quote[k]
        for k in (
            "gpu_usd_hour",
            "container_usd_gb_month",
            "network_usd_gb_month",
            "tax_and_other_usd",
        )
    ]
    if any(not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 for n in numbers):
        raise ValueError(
            "all charges, including tax/scheduler/transfer, need a finite quoted allowance"
        )
    gpu, container, network, extra = (Decimal(str(n)) for n in numbers)
    # Conservative 720-hour month; reserve all 30 days of storage, not just GPU hours.
    return (4 * gpu + 30 * container * 4 / 720 + 250 * network + extra).quantize(
        Decimal("0.01"), rounding=ROUND_CEILING
    )


def validate_plan(plan: dict[str, Any], now: float) -> float:
    if plan.get("deployment_authorized") is not True:
        raise ValueError("NOT READY: conditional budget approval is not deployment authorization")
    if (
        plan.get("data_center") != "US-CO-1"
        or plan.get("network_volume_name") != "roomcraft-training"
    ):
        raise ValueError("wrong approved region or dedicated volume name")
    if (
        plan.get("network_volume_gb") != 250
        or plan.get("network_tier") != "standard"
        or plan.get("mount") != "/workspace"
        or plan.get("global_volume") is not None
    ):
        raise ValueError("storage specification must exclude all Global Volumes")
    for field in (
        "capacity_confirmed",
        "blackwell_runtime_verified",
        "pod_scoped_permission_verified",
        "storage_cleanup_arranged",
    ):
        if plan.get(field) is not True:
            raise ValueError(f"NOT READY: {field}")
    quote = plan["quote"]
    age = now - timestamp(quote["checked_utc"])
    if not quote.get("authenticated") or not 0 <= age <= 900:
        raise ValueError("NOT READY: authenticated quote must be at most 15 minutes old")
    if (
        plan.get("budget_usd") != 40
        or plan.get("max_hours_from_creation") != 4
        or budget_total(plan) > Decimal("40")
    ):
        raise ValueError("NOT READY: complete allocation exceeds the $40/four-hour limits")
    protection = plan["cloud_shutdown"]
    if (
        protection.get("independent_of_pod_and_windows") is not True
        or protection.get("real_stop_test_verified") is not True
    ):
        raise ValueError("NOT READY: independent cloud shutdown has not passed a real stop test")
    if protection.get("pod_id") != plan.get("pod_id") or protection.get(
        "credential_resource_ids"
    ) != [plan.get("pod_id")]:
        raise ValueError("shutdown credential must be restricted server-side to this exact Pod")
    if not protection.get("readback_reference") or not protection.get("stop_test_reference"):
        raise ValueError("missing provider schedule read-back or stop-test evidence")
    created = timestamp(plan["pod_created_utc"])
    deadline = timestamp(protection["deadline_utc"])
    if not created <= now < deadline <= created + 4 * 3600 - 300:
        raise ValueError(
            "deadline must leave five minutes for stopping within four hours of creation"
        )
    return deadline - now


def run_on_pod(plan: dict[str, Any]) -> int:
    # Never accept a generic account credential as a fallback.
    if sys.platform != "linux" or os.environ.get("RUNPOD_POD_ID") != plan.get("pod_id"):
        raise ValueError("must run inside the exact newly approved RoomCraft Pod")
    stopper = PodStopper(
        plan["pod_id"], plan["pod_name"], os.environ.get("ROOMCRAFT_POD_SCOPED_KEY", "")
    )
    pod = stopper.inspect()  # Identity check precedes any stop-on-failure operation.
    try:
        seconds = validate_plan(plan, time.time())
        if pod.get("dataCenterId") != "US-CO-1" or pod.get("createdAt") != plan["pod_created_utc"]:
            raise ValueError("live Pod region/creation time disagrees with protection")
        expected = [{"volumeId": plan["network_volume_id"], "path": "/workspace"}]
        mounts = pod.get("mounts", {})
        if mounts.get("network") != expected or mounts.get("global"):
            raise ValueError("unapproved storage attachment")
        live_cost = pod.get("cost")
        ceiling = (
            float(plan["quote"]["gpu_usd_hour"])
            + 30 * float(plan["quote"]["container_usd_gb_month"]) / 720
        )
        if (
            not isinstance(live_cost, (int, float))
            or not math.isfinite(live_cost)
            or live_cost <= 0
            or live_cost > ceiling
        ):
            raise ValueError("live Pod hourly cost exceeds the quoted compute/container allowance")
        if Path.cwd().resolve() != Path("/workspace/RoomCraftAI"):
            raise ValueError("wrong isolated project directory")
    except Exception:
        stopper.stop()
        raise
    os.environ["ROOMCRAFT_CLOUD_SUPERVISED"] = "yes"
    os.environ["ROOMCRAFT_V06_APPROVED"] = "yes"

    def interrupt(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt("cloud supervisor interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    # Credentials stay in supervisor memory; children do not receive the stop key.
    os.environ.pop("ROOMCRAFT_POD_SCOPED_KEY", None)
    os.environ.pop("RUNPOD_API_KEY", None)
    return supervise(["bash", "scripts/runpod_v06_experiment.sh", "pilot"], seconds, stopper.stop)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--run-on-pod", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    if args.run_on_pod:
        raise SystemExit(run_on_pod(plan))
    try:
        seconds = validate_plan(plan, time.time())
    except (ValueError, KeyError, TypeError) as error:
        print(str(error))
        raise SystemExit(2) from None
    print(f"Recorded gates pass; {seconds:.0f}s to protected deadline. No resources changed.")


if __name__ == "__main__":
    main()
