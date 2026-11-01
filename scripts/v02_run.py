"""Small, serial local-GPU experiment. Existing completed runs are never overwritten."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STYLE = "Scandinavian living room with a blue fabric sofa"
CAPTION = (
    "A blue upholstered two-seat sofa with wooden legs, realistic photograph of a furnished "
    "Scandinavian living room. The sofa is placed against the right wall below the small window, "
    "resting on the wooden floor, natural daylight, realistic contact shadows."
)
PLAN = [
    {"id": "w17-full-floor", "room": "windows", "strength": 1.0, "mask": "floor"},
    {"id": "w01-baseline", "room": "windows", "prompt": None},
    {"id": "w02-caption", "room": "windows"},
    {"id": "w03-floor", "room": "windows", "mask": "floor"},
    {"id": "w04-strength080", "room": "windows", "strength": 0.8},
    {"id": "w05-strength100", "room": "windows", "strength": 1.0},
    {"id": "w06-guidance5", "room": "windows", "guidance": 5.0},
    {"id": "w07-guidance10", "room": "windows", "guidance": 10.0},
    {"id": "w08-steps50", "room": "windows", "steps": 50},
    {"id": "w09-seed17", "room": "windows", "seed": 17},
    {
        "id": "a01-caption",
        "room": "apartment",
        "prompt": "A blue upholstered two-seat sofa with wooden legs, realistic photograph of a furnished "
        "Scandinavian living room. The compact sofa is placed against the right wall on the beige "
        "carpet, realistic contact shadows, natural room lighting.",
    },
    {
        "id": "h01-caption",
        "room": "hotel",
        "prompt": "A comfortable blue upholstered armchair with wooden legs in the corner of a hotel room, "
        "realistic interior photograph. The armchair stands on the parquet floor beside the "
        "curtains, natural window lighting, realistic contact shadows.",
    },
    {"id": "w10-canny050", "room": "windows", "conditioning": "canny", "control_scale": 0.5},
    {"id": "w11-canny025", "room": "windows", "conditioning": "canny", "control_scale": 0.25},
    {"id": "w12-short", "room": "windows", "prompt": "A photo of a blue fabric sofa"},
    {"id": "w13-full-seed17", "room": "windows", "strength": 1.0, "seed": 17},
    {"id": "w14-full-broad", "room": "windows", "strength": 1.0, "mask": "broad"},
    {
        "id": "w15-full-canny050",
        "room": "windows",
        "strength": 1.0,
        "conditioning": "canny",
        "control_scale": 0.5,
    },
    {
        "id": "w16-full-canny025",
        "room": "windows",
        "strength": 1.0,
        "conditioning": "canny",
        "control_scale": 0.25,
    },
    {
        "id": "a02-full",
        "room": "apartment",
        "strength": 1.0,
        "prompt": "A blue upholstered two-seat sofa with wooden legs, realistic photograph of a furnished "
        "Scandinavian living room. The compact sofa is placed against the right wall on the beige "
        "carpet, realistic contact shadows, natural room lighting.",
    },
    {
        "id": "h02-full",
        "room": "hotel",
        "strength": 1.0,
        "prompt": "A comfortable blue upholstered armchair with wooden legs in the corner of a hotel room, "
        "realistic interior photograph. The armchair stands on the parquet floor beside the "
        "curtains, natural window lighting, realistic contact shadows.",
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", nargs="*", help="Run only these experiment IDs")
    parser.add_argument("--include-canny", action="store_true")
    args = parser.parse_args()
    (ROOT / "experiments/v02_plan.json").write_text(
        json.dumps(
            {
                "defaults": {
                    "size": 1024,
                    "steps": 30,
                    "guidance": 7.5,
                    "strength": 0.99,
                    "seed": 42,
                    "mask": "silhouette",
                    "offload": True,
                    "prompt": CAPTION,
                    "style": STYLE,
                },
                "runs": PLAN,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    env = dict(os.environ, HF_HOME=str(ROOT / ".cache/huggingface"), HF_HUB_OFFLINE="1")
    for run in PLAN:
        if args.ids and run["id"] not in args.ids:
            continue
        if run.get("conditioning") and not args.include_canny:
            continue
        output = ROOT / "outputs/v02" / run["id"]
        if (output / "metadata.json").exists():
            print("Already completed:", run["id"], flush=True)
            continue
        room = run["room"]
        command = [
            sys.executable,
            "-m",
            "roomcraft",
            "infer",
            "--image",
            str(ROOT / f"data/v02/{room}.jpg"),
            "--mask",
            str(ROOT / f"data/v02/{room}-{run.get('mask', 'silhouette')}.png"),
            "--style",
            STYLE,
            "--size",
            "1024",
            "--offload",
            "--output",
            str(output),
        ]
        for key, default in [
            ("steps", 30),
            ("guidance", 7.5),
            ("strength", 0.99),
            ("seed", 42),
            ("conditioning", "none"),
            ("control_scale", 0.5),
        ]:
            command.extend(["--" + key.replace("_", "-"), str(run.get(key, default))])
        prompt = run.get("prompt", CAPTION)
        if prompt is not None:
            command.extend(["--prompt", prompt])
        logs = ROOT / "outputs/v02/logs"
        logs.mkdir(parents=True, exist_ok=True)
        print("Running", run["id"], flush=True)
        with (logs / f"{run['id']}.txt").open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True
            )
        print("Completed", run["id"], flush=True)


if __name__ == "__main__":
    main()
