"""Read-only FP32 stage activations of the actual calibrated encoder, on CPU."""

import json
from pathlib import Path

import torch
from safetensors.torch import load_file

from roomcraft.causality import intervene
from roomcraft.common import sha256, write_json
from roomcraft.editing import edit_size
from roomcraft.preprocess import read_rgb
from roomcraft.rga import GeometryEncoder, RGAConfig, geometry_input


def main():
    torch.set_num_threads(6)
    root = Path(__file__).resolve().parents[1]
    checkpoint = root / "runs/v09-rga-calibrated/pilot/step-000100"
    expected = "31625ddd4f82f10c8ff08947ffe68aa8d08682813c2b47e11cab15321ae56fca"
    if sha256(checkpoint / "adapter.safetensors") != expected:
        raise ValueError("calibrated checkpoint changed")
    config = json.loads((checkpoint / "config.json").read_text())["geometry"]
    model = GeometryEncoder(RGAConfig(**config)).eval()
    model.load_state_dict(
        {
            k.removeprefix("geometry."): v
            for k, v in load_file(str(checkpoint / "adapter.safetensors")).items()
        }
    )
    cases = json.loads((root / "configs/v07_evaluation.json").read_text())["cases"]
    rows = []
    with torch.inference_mode():
        for case in cases:
            path = root / case["source"]
            if sha256(path) != case["source_sha256"]:
                raise ValueError("source changed")
            image = read_rgb(path)
            g = geometry_input(image, edit_size(image, 512))
            a, b = g, intervene(g, "zero_edges")
            stages = []
            modules = list(model.encoder) + [model.projection]
            for index, module in enumerate(modules):
                a, b = module(a), module(b)
                if isinstance(module, torch.nn.SiLU) or module is model.projection:
                    flat = a.flatten()
                    # Deterministic bounded subsample for distribution quantiles.
                    sample = flat[:: max(1, flat.numel() // 100000)]
                    q = torch.quantile(sample, torch.tensor([0.01, 0.5, 0.99]))
                    stages.append(
                        {
                            "layer": index,
                            "shape": list(a.shape),
                            "mean": float(a.mean()),
                            "std": float(a.std()),
                            "rms": float(a.square().mean().sqrt()),
                            "min": float(a.min()),
                            "max": float(a.max()),
                            "sampled_p01_p50_p99": q.tolist(),
                            "finite": bool(torch.isfinite(a).all()),
                            "zero_edge_relative_l2": float(
                                (a - b).norm() / a.norm().clamp_min(1e-12)
                            ),
                        }
                    )
            rows.append({"case": case["id"], "source_sha256": sha256(path), "stages": stages})
    write_json(
        root / "outputs/v10-causality/activations.json",
        {
            "checkpoint_sha256": expected,
            "scope": "CPU FP32 encoder-only activations; no full-Qwen inference",
            "cases": rows,
        },
    )
    print(
        json.dumps(
            {"cases": len(rows), "all_finite": all(s["finite"] for r in rows for s in r["stages"])}
        )
    )


if __name__ == "__main__":
    main()
