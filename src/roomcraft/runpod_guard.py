"""External, fail-closed experiment watchdog. No creation, start, deletion, or credentials on disk."""

import argparse
import json
import math
import os
import re
import subprocess
import time
import urllib.request
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any


def allowed_seconds(plan: dict[str, Any], now: float) -> float:
    if plan.get("owner_approved") is not True:
        raise ValueError("explicit owner approval must be recorded before running")
    for key in ("budget_usd", "pod_hourly_ceiling_usd", "max_hours", "reserve_usd"):
        if not math.isfinite(plan[key]) or plan[key] <= 0:
            raise ValueError(f"invalid {key}")
    start = datetime.fromisoformat(plan["billing_started_utc"].replace("Z", "+00:00"))
    if start.tzinfo is None or start.timestamp() > now:
        raise ValueError("billing start must be timezone-aware and no later than now")
    # Reserve the FULL approved storage retention cost, including time after Pod stop.
    storage = plan["storage_reserve_usd"]
    if not math.isfinite(storage) or storage < 0:
        raise ValueError("invalid storage reserve")
    hours = min(
        plan["max_hours"],
        (plan["budget_usd"] - storage - plan["reserve_usd"]) / plan["pod_hourly_ceiling_usd"],
    )
    return max(0.0, start.timestamp() + hours * 3600 - now)


class PodStopper:
    """Only read and stop the exact approved Pod. v2 schema archived in docs/sources/v06."""

    def __init__(self, pod_id: str, name: str, token: str) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", pod_id) or not name.startswith("roomcraft-v06-"):
            raise ValueError("need the exact RoomCraft V0.6 Pod identity")
        if not token:
            raise ValueError("RUNPOD_API_KEY is required; never put it in the approval JSON")
        self.pod_id, self.name, self.token = pod_id, name, token

    def request(self, method: str, action: bool = False) -> dict[str, Any]:
        url = f"https://api.runpod.io/v2/pods/{self.pod_id}" + ("/action" if action else "")
        request = urllib.request.Request(
            url,
            data=b'{"action":"stop"}' if action else None,
            headers={"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            return dict(json.load(response))

    def inspect(self) -> dict[str, Any]:
        pod = self.request("GET")
        if pod.get("id") != self.pod_id or pod.get("name") != self.name:
            raise ValueError("Pod identity mismatch; refusing to operate on another resource")
        return pod

    def stop(self, attempts: int = 12, delay: float = 5) -> dict[str, Any]:
        for attempt in range(attempts):
            try:
                pod = self.inspect()
                if pod.get("status") == "EXITED":
                    return {"id": self.pod_id, "status": "EXITED", "attempts": attempt + 1}
                if "stop" in pod.get("actions", []):
                    self.request("POST", action=True)
                # Always require a fresh GET, not just acceptance of the POST.
            except (OSError, TimeoutError):
                pass
            if attempt + 1 < attempts:
                time.sleep(delay)
        raise RuntimeError(
            f"STOP NOT VERIFIED for {self.pod_id}; stop it in the console immediately"
        )


def supervise(command: list[str], seconds: float, stop: Callable[[], Any]) -> int:
    """Run an SSH workload; stop the remote Pod on success, failure, interruption, or deadline."""
    child = None
    try:
        if seconds <= 0:
            raise TimeoutError("approved allocation time already expired")
        child = subprocess.Popen(command)
        try:
            return child.wait(timeout=seconds)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError("experiment deadline exceeded") from error
    finally:
        try:
            stop()
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    plan = json.loads(args.approval.read_text(encoding="utf-8"))
    seconds = allowed_seconds(plan, time.time())
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or args.log.exists():
        raise ValueError("need command and a new watchdog log")
    stopper = PodStopper(plan["pod_id"], plan["pod_name"], os.environ.get("RUNPOD_API_KEY", ""))
    args.log.parent.mkdir(parents=True, exist_ok=True)
    with args.log.open("x", encoding="utf-8") as log:

        def record(data: dict[str, Any]) -> None:
            log.write(json.dumps({"unix_time": time.time(), **data}) + "\n")
            log.flush()

        def stop() -> None:
            try:
                record({"event": "stop_verified", **stopper.stop()})
            except Exception:
                record({"event": "STOP_UNVERIFIED", "pod_id": stopper.pod_id})
                raise

        # Identity verification is outside the stop-on-failure block to protect unrelated Pods.
        pod = stopper.inspect()
        try:
            if pod.get("status") != "RUNNING" or "stop" not in pod.get("actions", []):
                raise ValueError("Pod must be running with a supported stop action")
            cost = pod.get("cost")
            if not isinstance(cost, (int, float)) or not math.isfinite(cost) or cost <= 0:
                raise ValueError("missing live hourly quote")
            if cost > plan["pod_hourly_ceiling_usd"]:
                raise ValueError("live Pod price exceeds approval")
            mounts = pod.get("mounts", {})
            expected_mount = [{"volumeId": plan["network_volume_id"], "path": "/workspace"}]
            if mounts.get("global") or mounts.get("network") != expected_mount:
                raise ValueError("need the dedicated approved network volume and no Global Volume")
            billed = datetime.fromisoformat(plan["billing_started_utc"].replace("Z", "+00:00"))
            created = datetime.fromisoformat(pod["createdAt"].replace("Z", "+00:00"))
            if billed > created:
                raise ValueError("budget must include all time since Pod creation")
            record({"event": "armed", "seconds_remaining": seconds, "pod_cost_usd_hour": cost})
        except Exception:
            stop()
            raise
        code = supervise(command, allowed_seconds(plan, time.time()), stop)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
