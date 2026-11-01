"""Shared exact-ID lifecycle library; historical allocation entrypoints excluded."""

import ctypes
import datetime
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from runpod_credential_source import NoRedirect, load_runpod_credential, ssh_identity

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".cache/runpod-first-training"


def save(name, value):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / name
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def api(method, path, payload=None):
    credential = load_runpod_credential()
    request = urllib.request.Request(
        "https://api.runpod.io/v2/" + path,
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            **credential.authorization_header(),
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
        },
        method=method,
    )
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            raw = response.read()
            return (response.status, json.loads(raw) if raw else None)
    except urllib.error.HTTPError as error:
        return (error.code, None)


def session():
    return json.loads((STATE / "session.json").read_text())


def identity(pod, record):
    if (
        pod.get("id") != record["podId"]
        or pod.get("name") != record["name"]
        or record.get("namePrefix", "roomcraft-v06-")
        not in (
            "roomcraft-v06-",
            "roomcraft-v07-",
            "roomcraft-v09-",
            "roomcraft-v10-causality-",
            "roomcraft-mvp-",
        )
        or (not record["name"].startswith(record.get("namePrefix", "roomcraft-v06-")))
    ):
        raise ValueError("Pod identity uncertain; refusing lifecycle action")
    if pod.get("mounts", {}).get("global") or pod.get("mounts", {}).get("network") != [
        {"volumeId": record["volumeId"], "path": "/workspace"}
    ]:
        raise ValueError("Pod storage identity uncertain; refusing lifecycle action")


def terminate(record):
    code, pod = api("GET", "pods/" + record["podId"])
    if code == 404:
        return True
    if code != 200:
        raise RuntimeError(f"Pod identity read HTTP {code}; no deletion sent")
    identity(pod, record)
    if "terminate" not in pod.get("actions", []):
        raise ValueError("Provider does not advertise termination")
    code, _ = api("DELETE", "pods/" + record["podId"])
    save(
        "termination-request.json",
        {"podId": record["podId"], "httpStatus": code, "epoch": time.time()},
    )
    if code not in (204, 200, 404):
        raise RuntimeError(f"Termination HTTP {code}")
    code, _ = api("GET", "pods/" + record["podId"])
    return code == 404


def guard():
    awake = (
        bool(ctypes.windll.kernel32.SetThreadExecutionState(2147483649))
        if os.name == "nt"
        else False
    )
    code, _ = api("GET", "catalog/gpus")
    if code != 200:
        raise RuntimeError(f"Guard credential preflight HTTP {code}")
    save(
        "guard-ready.json",
        {
            "pid": os.getpid(),
            "epoch": time.time(),
            "state": "prearmed",
            "preventSleep": awake,
            "credentialSource": "Windows Credential Manager",
            "secretsPersisted": False,
        },
    )
    start = time.time()
    while not (STATE / "session.json").exists():
        if time.time() - start > 1200:
            return
        time.sleep(1)
    record = session()
    created = datetime.datetime.fromisoformat(
        record["createdAt"].replace("Z", "+00:00")
    ).timestamp()
    deadline = min(
        record.get("deadlineUnix", float("inf")),
        created + 4 * 3600 - 120,
        created
        + (40 - 17.5 - 6.5 - record["priorRoomcraftSpendUsd"]) / record["combinedHourlyUsd"] * 3600
        - 120,
    )
    while True:
        now = time.time()
        heartbeat = STATE / "controller-heartbeat"
        age = now - (heartbeat.stat().st_mtime if heartbeat.exists() else created)
        reason = (
            "requested"
            if (STATE / "terminate-now").exists()
            else "deadline"
            if now >= deadline
            else "heartbeat-lost"
            if age > record.get("heartbeatTimeoutSeconds", 600)
            else None
        )
        try:
            code, pod = api("GET", "pods/" + record["podId"])
            if code == 404:
                save(
                    "guard-final.json",
                    {
                        "podId": record["podId"],
                        "state": "pod-absent",
                        "httpStatus": 404,
                        "confirmedUnix": now,
                        "estimatedComputeDiskUsd": (now - created)
                        / 3600
                        * record["combinedHourlyUsd"],
                        "retained30DayStorageUsd": record.get("retained30DayStorageUsd", 17.5),
                    },
                )
                return
            if code != 200:
                raise RuntimeError(f"Guard read HTTP {code}")
            identity(pod, record)
            save(
                "guard-status.json",
                {
                    "pid": os.getpid(),
                    "epoch": now,
                    "podId": record["podId"],
                    "deadlineUnix": deadline,
                    "state": pod["status"],
                    "heartbeatAgeSeconds": age,
                    "reason": reason,
                },
            )
            if reason:
                terminate(record)
        except Exception as error:
            save(
                "guard-error.json",
                {
                    "epoch": now,
                    "type": type(error).__name__,
                    "message": str(error)
                    if isinstance(error, (ValueError, RuntimeError))
                    else "Transport failure; details withheld",
                },
            )
        time.sleep(3 if reason else 15)


def transport(record):
    code, pod = api("GET", "pods/" + record["podId"])
    if code != 200:
        raise RuntimeError(f"Pod read HTTP {code}")
    identity(pod, record)
    if pod.get("status") != "RUNNING":
        raise RuntimeError("Pod not running yet")
    if (
        pod["gpu"]["id"] != record["gpuId"]
        or pod["gpu"]["count"] != 1
        or pod["gpu"]["memory"] < 120
        or (pod["dataCenterId"] != record["dataCenter"])
    ):
        raise ValueError("Allocated hardware/region mismatch")
    if (
        tuple(int(n) for n in pod["cudaVersion"].split(".")) < (12, 8)
        or pod["cost"] > record["combinedHourlyUsd"] + 0.001
    ):
        raise ValueError("Allocated CUDA/price mismatch")
    save("pod-readback.json", {k: v for k, v in pod.items() if k not in ("env", "registry")})
    connection = pod["ssh"]["direct"]
    if not connection:
        raise RuntimeError("Direct SSH endpoint is still starting")
    key = ssh_identity()
    if not key.is_file():
        raise ValueError("Existing authorized SSH identity unavailable")
    opts = [
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "UserKnownHostsFile=" + str(STATE / "known_hosts"),
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=20",
        "-o",
        "ServerAliveCountMax=3",
        "-i",
        str(key),
    ]
    return (connection, opts)
