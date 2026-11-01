"""Bounded app worker using existing RoomCraft auth, guard, SSH and exact-ID deletion.

The UI cannot call start. A fresh explicit owner approval is required before the
operator executes start. Quote/status are read-only. No volume is ever created.
"""

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import roomcraft_runpod as control
from runpod_credential_source import owner_settings

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/app_worker.json").read_text())
REMOTE = CONFIG["remote_root"]
PYTHON = REMOTE + "/environments/qwen/.venv/bin/python"


def approval(path):
    value = json.loads(path.read_text())
    if value.get("mission") != "roomcraft-mvp-inference-v1" or value.get("approved") is not True:
        raise ValueError(
            "Fresh explicit owner approval is required; a template is not authorization"
        )
    epoch = datetime.datetime.fromisoformat(
        value["approved_at_utc"].replace("Z", "+00:00")
    ).timestamp()
    if (
        not 0 <= time.time() - epoch < 86400
        or not 10 <= value["max_minutes"] <= 60
        or not 0 < value["budget_usd"] <= 10
    ):
        raise ValueError("Approval must be fresh with 10–60 minutes and a finite budget up to $10")
    return value


def quote(volume_id):
    if not volume_id or not re.fullmatch(r"[a-zA-Z0-9]+", volume_id):
        raise ValueError("Configure your dedicated RoomCraft regional volume ID")
    code, volume = control.api("GET", "network-volumes/" + volume_id)
    if code != 200 or volume["name"] != "roomcraft-training" or volume["type"] != "STANDARD":
        raise ValueError("The dedicated standard RoomCraft volume could not be verified")
    code, catalog = control.api(
        "GET", "catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE&minCudaVersion=12.8"
    )
    if code != 200:
        raise RuntimeError("Live GPU catalog is unavailable; no resource created")
    matches = [
        g
        for g in catalog["gpus"]
        if g["id"] in CONFIG["gpu_ids"]
        and g["memory"] >= 80
        and any(
            d["id"] == volume["dataCenter"] and d["availability"] in ("LOW", "MEDIUM", "HIGH")
            for d in g.get("dataCenters", [])
        )
    ]
    if not matches:
        raise ValueError("No compatible 80 GB GPU is available beside RoomCraft storage")
    gpu = min(matches, key=lambda g: g["price"]["secure"])
    return {
        "readUnix": time.time(),
        "volume": volume,
        "gpu": gpu,
        "combinedHourlyUsd": gpu["price"]["secure"] + CONFIG["disk_gb"] * 0.10 / 720,
        "networkVolumeHourlyUsd": volume["size"] * 0.07 / 720,
        "transferUsd": 0,
        "currency": "USD",
    }


@contextmanager
def heartbeat():
    stopped = threading.Event()
    (control.STATE / "controller-heartbeat").touch()

    def pulse():
        while not stopped.wait(15):
            (control.STATE / "controller-heartbeat").touch()

    thread = threading.Thread(target=pulse, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stopped.set()


def connect(record):
    ready = json.loads((control.STATE / "guard-status.json").read_text())
    if ready["podId"] != record["podId"] or time.time() - ready["epoch"] > 45:
        raise ValueError("An independent exact-Pod deadline guard is not fresh")
    connection, options = control.transport(record)
    target = connection["username"] + "@" + connection["host"]
    return (
        ["ssh", *options, "-p", str(connection["port"]), target],
        ["scp", *options, "-P", str(connection["port"])],
        target,
    )


def start(approval_path, volume_id):
    approved = approval(approval_path)
    if (control.STATE / "session.json").exists():
        raise ValueError("Use a new state directory; no duplicate allocation is allowed")
    q = quote(volume_id)
    reserve = max(0.50, approved["budget_usd"] * 0.15)
    maximum = (
        q["combinedHourlyUsd"] * approved["max_minutes"] / 60
        + 2 * q["networkVolumeHourlyUsd"]
        + reserve
    )
    if maximum > approved["budget_usd"]:
        raise ValueError("Quoted maximum lifetime, storage and reserve exceed the approved budget")
    control.save("quote.json", {**q, "reservedMaximumUsd": maximum, "reserveUsd": reserve})
    control.save("approval.json", approved)
    flags = (
        subprocess.DETACHED_PROCESS
        | subprocess.CREATE_NEW_PROCESS_GROUP
        | subprocess.CREATE_NO_WINDOW
        if os.name == "nt"
        else 0
    )
    subprocess.Popen(
        [sys.executable, str(Path(__file__).resolve()), "guard", "--state", str(control.STATE)],
        creationflags=flags,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(30):
        if (control.STATE / "guard-ready.json").exists():
            break
        time.sleep(1)
    else:
        raise ValueError("Independent guard did not arm; no Pod created")
    ready = json.loads((control.STATE / "guard-ready.json").read_text())
    if not ready["preventSleep"] or time.time() - ready["epoch"] > 45:
        raise ValueError("This supervised controller requires an awake Windows session")
    gpu, volume = q["gpu"], q["volume"]
    name = "roomcraft-mvp-" + datetime.datetime.now(datetime.UTC).strftime("%Y%m%d-%H%M%S")
    payload = {
        "name": name,
        "image": CONFIG["image"],
        "cloud": "SECURE",
        "gpu": {
            "id": gpu["id"],
            "count": 1,
            "minRamPerGpu": 128,
            "minVcpuCountPerGpu": 8,
            "minCudaVersion": "12.8",
        },
        "disk": CONFIG["disk_gb"],
        "ports": ["22/tcp"],
        "startSsh": True,
        "startJupyter": False,
        "dataCenterIds": [volume["dataCenter"]],
        "mounts": {"network": [{"volumeId": volume["id"], "path": "/workspace"}]},
    }
    code, pod = control.api("POST", "pods", payload)
    if code not in (200, 201):
        raise RuntimeError(f"Pod creation HTTP {code}; no automatic retry")
    created = datetime.datetime.fromisoformat(pod["createdAt"].replace("Z", "+00:00")).timestamp()
    record = {
        "podId": pod["id"],
        "name": name,
        "namePrefix": "roomcraft-mvp-",
        "createdAt": pod["createdAt"],
        "deadlineUnix": created + approved["max_minutes"] * 60 - 120,
        "heartbeatTimeoutSeconds": CONFIG["idle_seconds"],
        "volumeId": volume["id"],
        "dataCenter": volume["dataCenter"],
        "gpuId": gpu["id"],
        "combinedHourlyUsd": q["combinedHourlyUsd"],
        "priorRoomcraftSpendUsd": 0,
        "maximumMissionUsd": approved["budget_usd"],
        "retained30DayStorageUsd": volume["size"] * 0.07,
    }
    control.save("session.json", record)
    try:
        control.identity(pod, record)
        with heartbeat():
            for _ in range(60):
                try:
                    ssh, scp, target = connect(record)
                    break
                except (RuntimeError, FileNotFoundError):
                    time.sleep(5)
            else:
                raise RuntimeError("GPU worker was not ready within five minutes")
            archive = control.STATE / "app-code.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                for p in list((ROOT / "src/roomcraft").glob("*.py")) + [
                    ROOT / "src/roomcraft/models.json",
                    ROOT / "src/roomcraft/presets.json",
                    ROOT / "src/roomcraft/themes.json",
                    ROOT / "scripts/app_worker.py",
                ]:
                    tar.add(p, arcname=p.relative_to(ROOT).as_posix())
            runtime = REMOTE + "/app-runtime/" + record["podId"]
            subprocess.run(
                [
                    *ssh,
                    "test -d "
                    + REMOTE
                    + " && mkdir -p "
                    + REMOTE
                    + "/app-runtime && mkdir "
                    + runtime,
                ],
                check=True,
                timeout=30,
            )
            subprocess.run(
                [*scp, str(archive), target + ":" + runtime + "/code.tar.gz"],
                check=True,
                timeout=120,
            )
            subprocess.run(
                [
                    *ssh,
                    "cd "
                    + runtime
                    + " && tar -xzf code.tar.gz && nvidia-smi --query-gpu=name,memory.total --format=csv",
                ],
                check=True,
                timeout=30,
            )
            control.save(
                "app-ready.json",
                {"podId": record["podId"], "runtime": runtime, "readyUnix": time.time()},
            )
        print(
            json.dumps(
                {
                    "state": str(control.STATE),
                    "podId": record["podId"],
                    "deadlineUnix": record["deadlineUnix"],
                    "idleShutdownSeconds": CONFIG["idle_seconds"],
                    "quotedMaximumUsd": maximum,
                }
            )
        )
    except BaseException:
        control.save(
            "startup-termination.json",
            {"fresh_404": control.terminate(record), "epoch": time.time()},
        )
        raise


def infer(job):
    record = control.session()
    if record.get("namePrefix") != "roomcraft-mvp-":
        raise ValueError("The application requires its own approved MVP worker session")
    ready = json.loads((control.STATE / "app-ready.json").read_text())
    if (
        ready["podId"] != record["podId"]
        or not re.fullmatch(r"[a-zA-Z0-9]+", record["podId"])
        or ready["runtime"] != REMOTE + "/app-runtime/" + record["podId"]
        or not re.fullmatch(r"[a-f0-9]{32}", job.name)
    ):
        raise ValueError("Invalid exact-Pod/job binding")
    if record["deadlineUnix"] - time.time() < 660:
        raise ValueError(
            "Less than eleven minutes remain in the approved worker lifetime. Stop this worker; request a fresh session before generating."
        )
    remote_job = ready["runtime"] + "/jobs/" + job.name
    ssh = None
    try:
        with heartbeat():
            ssh, scp, target = connect(record)
            subprocess.run(
                [*ssh, "mkdir -p " + ready["runtime"] + "/jobs && mkdir " + remote_job],
                check=True,
                timeout=30,
            )
            for name in ("upload.png", "request.json"):
                subprocess.run(
                    [*scp, str(job / name), target + ":" + remote_job + "/" + name],
                    check=True,
                    timeout=60,
                )
            cmd = (
                "cd "
                + ready["runtime"]
                + " && export PYTHONPATH=$PWD/src HF_HOME="
                + REMOTE
                + "/.cache/huggingface TORCH_HOME="
                + REMOTE
                + "/.cache/torch HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8 && "
                + PYTHON
                + " scripts/app_worker.py --job "
                + remote_job
                + " --lora "
                + REMOTE
                + "/outputs/v06/trained/pilot/step-000100 --rga "
                + REMOTE
                + "/runs/v09-rga-calibrated/pilot/step-000100"
            )
            subprocess.run(
                [*ssh, "bash -c " + shlex.quote(cmd)],
                check=False,
                timeout=min(900, record["deadlineUnix"] - time.time() - 60),
            )
            subprocess.run(
                [*scp, target + ":" + remote_job + "/response.json", str(job / "response.json")],
                check=True,
                timeout=30,
            )
            response = json.loads((job / "response.json").read_text())
            if not response.get("error"):
                subprocess.run(
                    [*scp, "-r", target + ":" + remote_job + "/result", str(job)],
                    check=True,
                    timeout=120,
                )
    finally:
        if ssh:
            # Only the unique application job directory is removed; model/cache roots are untouched.
            cleanup = subprocess.run(
                [
                    *ssh,
                    "rm -rf -- "
                    + shlex.quote(remote_job)
                    + " && test ! -e "
                    + shlex.quote(remote_job),
                ],
                check=False,
                timeout=30,
            )
            control.save(
                "job-cleanup-" + job.name + ".json",
                {
                    "podId": record["podId"],
                    "jobId": job.name,
                    "remote_temp_removed_verified": cleanup.returncode == 0,
                    "epoch": time.time(),
                },
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("quote", "start", "guard", "infer", "stop", "status"))
    parser.add_argument("--state", type=Path, default=ROOT / ".cache/runpod-mvp")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--volume", default=owner_settings().get("volume_id"))
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    control.STATE = args.state.resolve()
    control.STATE.mkdir(parents=True, exist_ok=True)
    try:
        if args.action == "quote":
            print(json.dumps(quote(args.volume), indent=2))
        elif args.action == "start":
            if args.approval is None:
                parser.error("--approval is required; start is billable")
            start(args.approval, args.volume)
        elif args.action == "guard":
            control.guard()
        elif args.action == "stop":
            r = control.session()
            if r.get("namePrefix") != "roomcraft-mvp-":
                raise ValueError("Not an MVP worker; refusing lifecycle action")
            confirmed = control.terminate(r)
            control.save(
                "app-termination.json",
                {"podId": r["podId"], "fresh_404": confirmed, "epoch": time.time()},
            )
            if not confirmed:
                raise RuntimeError(
                    "Pod termination is not yet freshly confirmed; the independent guard remains armed"
                )
            print((control.STATE / "app-termination.json").read_text())
        elif args.action == "status":
            r = control.session()
            code, pod = control.api("GET", "pods/" + r["podId"])
            if code == 200:
                control.identity(pod, r)
            print(
                json.dumps(
                    {"podId": r["podId"], "httpStatus": code, "deadlineUnix": r["deadlineUnix"]}
                )
            )
        else:
            if args.job is None:
                parser.error("--job is required")
            infer(args.job.resolve())
    except Exception:
        if args.action == "infer" and args.job:
            # The original diagnostic stays in the private worker log; do not disclose credentials.
            (args.job / "response.json").write_text(
                json.dumps(
                    {
                        "error": "The approved Runpod worker is unavailable, out of time, or failed. Check worker status and restart only under a fresh approved budget."
                    }
                )
            )
        raise


if __name__ == "__main__":
    main()
