"""Plot actual JSONL diagnostics; run in an isolated matplotlib environment."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.run / "loss.jsonl").read_text().splitlines()]
    if not rows:
        raise ValueError("no actual training measurements")
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(steps, [r["loss"] for r in rows], alpha=0.6, label="Sampled training loss")
    val = [r for r in rows if "validation_loss" in r]
    ax.plot(
        [r["step"] for r in val],
        [r["validation_loss"] for r in val],
        "o-",
        label="Fixed validation",
    )
    ax.set(ylabel="Conditional flow MSE", title="Loss (not an image-quality score)")
    ax.legend()
    ax = axes[0, 1]
    ax.plot(steps, [r["gradient_norm"] for r in rows], label="All RGA parameters")
    encoder = [r for r in rows if r["encoder_gradient_norm"] > 0]
    ax.plot(
        [r["step"] for r in encoder],
        [r["encoder_gradient_norm"] for r in encoder],
        label="Encoder only; zero first step omitted",
    )
    ax.set(yscale="log", ylabel="L2 norm before clipping", title="Actual gradient flow")
    ax.legend(fontsize=8)
    axes[1, 0].plot(steps, [r["peak_gpu_allocated_bytes"] / 2**30 for r in rows], label="Allocated")
    axes[1, 0].plot(steps, [r["peak_gpu_reserved_bytes"] / 2**30 for r in rows], label="Reserved")
    axes[1, 0].set(ylabel="GiB", title="Cumulative peak GPU memory")
    axes[1, 0].legend()
    axes[1, 1].plot(steps, [r["elapsed_seconds"] / 60 for r in rows])
    axes[1, 1].set(ylabel="Minutes", title="Training/validation loop time (loading excluded)")
    for ax in axes.flat:
        ax.set_xlabel("Optimizer update")
        ax.grid(alpha=0.25)
    fig.suptitle("RoomCraft V0.9 RGA — " + args.run.name)
    fig.savefig(args.run / "diagnostics.png", dpi=160)
    fig.savefig(args.run / "diagnostics.svg")
    (args.run / "plot-environment.json").write_text(
        json.dumps({"matplotlib": matplotlib.__version__}, indent=2)
    )


if __name__ == "__main__":
    main()
