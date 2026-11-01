"""Conservative time gate from measured smoke updates; the independent guard caps spend."""

import argparse
import json
import math
import time
from pathlib import Path


def forecast(seconds):
    if len(seconds) != 2 or any(not math.isfinite(v) or v <= 0 for v in seconds):
        raise ValueError("need two measured finite optimizer-update durations")
    # Smoke: one backward update + 15 validation forwards. Pilot: 400 backward
    # microbatches + 150 validation forwards, plus six 40-step CFG generations.
    # Approximate backward as 3 forwards and retain 50% headroom + five minutes
    # loading/checkpoint overhead. This is a planning bound, not a speed claim.
    return 1.5 * max(seconds) * ((400 * 3 + 150 + 6 * 40 * 2) / (3 + 15)) + 300


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    durations = []
    for name in ("smoke-first", "smoke-resumed"):
        rows = [
            json.loads(line) for line in (args.root / name / "loss.jsonl").read_text().splitlines()
        ]
        durations.append(rows[-1]["elapsed_seconds"])
    estimate = forecast(durations)
    remaining = args.deadline - time.time()
    record = {
        "status": "pilot_time_budget_passed"
        if remaining > estimate
        else "pilot_skipped_insufficient_time",
        "measured_smoke_update_seconds": durations,
        "conservative_pilot_and_evaluation_seconds": estimate,
        "seconds_until_guard_deadline": remaining,
        "basis": "Observed smoke timings; approximate forward/backward cost ratio, 50% headroom, five-minute overhead. Actual runtime can differ; independent guard remains authoritative.",
    }
    (args.root / "pilot-budget.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)
    raise SystemExit(0 if remaining > estimate else 3)


if __name__ == "__main__":
    main()
