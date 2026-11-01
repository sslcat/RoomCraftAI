"""Launch the local RoomCraft studio; this command never provisions cloud compute."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    from roomcraft.app import launch
    from roomcraft.app_service import AppConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="Open the local studio in your browser")
    parser.add_argument(
        "--worker-python",
        type=Path,
        default=ROOT
        / "environments/qwen/.venv"
        / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python"),
    )
    parser.add_argument(
        "--runpod-session",
        type=Path,
        help="An already approved, deadline-guarded RoomCraft session directory",
    )
    parser.add_argument("--lora", type=Path, default=AppConfig.lora)
    parser.add_argument("--rga", type=Path, default=AppConfig.rga)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("Choose a local port from 1024 to 65535")
    launch(
        AppConfig(
            worker_python=args.worker_python.resolve(),
            lora=args.lora.resolve(),
            rga=args.rga.resolve(),
            remote_session=args.runpod_session.resolve() if args.runpod_session else None,
        ),
        args.port,
        args.open,
    )


if __name__ == "__main__":
    main()
