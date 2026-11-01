"""Local calibrated-RGA diagnostics and explicit, unexecuted full-model intervention plan."""

import argparse
import gc
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors.torch import load_file

from roomcraft.causality import intervene, save_intervention
from roomcraft.common import sha256, write_json
from roomcraft.editing import EditConfig, edit, edit_size
from roomcraft.preprocess import read_rgb
from roomcraft.rga import GeometryEncoder, RGAConfig, geometry_input

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path("runs/v09-rga-calibrated/pilot/step-000100")
EXPECTED = "31625ddd4f82f10c8ff08947ffe68aa8d08682813c2b47e11cab15321ae56fca"


def boundary_mask(record, size, category=None, name=None, segment=None):
    im = Image.new("L", size)
    draw = ImageDraw.Draw(im)
    for line in record["polylines"]:
        if category and line["category"] != category or name and line["name"] != name:
            continue
        for i in range(len(line["normalized_xy"]) - 1):
            if segment is not None and i != segment:
                continue
            if min(line["camera_depth"][i : i + 2]) <= 0:
                continue
            points = [(x * size[0], y * size[1]) for x, y in line["normalized_xy"][i : i + 2]]
            draw.line(points, fill=255, width=max(5, round(max(size) / 64)))
    return torch.from_numpy(np.asarray(im).copy() > 0)


def heatmap(values, destination):
    v = values.detach().float().cpu().numpy()
    limit = max(float(np.quantile(v, 0.995)), 1e-12)
    pixels = cv2.applyColorMap(np.uint8(np.clip(v / limit, 0, 1) * 255), cv2.COLORMAP_INFERNO)
    Image.fromarray(cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)).save(destination)
    return limit


def prepare(output):
    if (output / "plan.json").exists():
        raise ValueError("choose new immutable diagnostic output")
    checkpoint = ROOT / CHECKPOINT
    if sha256(checkpoint / "adapter.safetensors") != EXPECTED:
        raise ValueError("wrong RGA checkpoint; use calibrated V0.9")
    cfg = json.loads((checkpoint / "config.json").read_text())
    model = GeometryEncoder(RGAConfig(**cfg["geometry"])).eval().requires_grad_(False)
    model.load_state_dict(
        {
            k.removeprefix("geometry."): v
            for k, v in load_file(str(checkpoint / "adapter.safetensors")).items()
        }
    )
    torch.set_num_threads(6)
    protocol = json.loads((ROOT / "configs/v07_evaluation.json").read_text())
    cases = protocol["cases"]
    rows = []
    runs = []
    body = [
        "<h1>Geometry interventions: encoder diagnostics only</h1><p>No full pretrained Qwen generation was run. Heatmaps independently scaled to their 99.5th percentile; they do not show architectural preservation. Labels/projections come from actual saved Blender cameras. Real photos lack semantic landmark annotations.</p>"
    ]
    for index, case in enumerate(cases):
        source = ROOT / case["source"]
        if sha256(source) != case["source_sha256"]:
            raise ValueError("source bytes changed")
        image = read_rgb(source)
        donor_case = cases[(index + 1) % 3 + (3 if index >= 3 else 0)]
        donor_image = read_rgb(ROOT / donor_case["source"])
        folder = output / case["id"]
        folder.mkdir(parents=True, exist_ok=True)
        landmark_path = output / "landmarks" / (case["id"] + ".json")
        landmarks = json.loads(landmark_path.read_text()) if landmark_path.exists() else None
        if landmarks and landmarks["source_sha256"] != sha256(source):
            raise ValueError("projected scene/source mismatch")
        arms = ["correct", "disabled", "zero_edges", "shifted", "unrelated"] + (
            ["removed", "random_removed"] if landmarks else []
        )
        source_relative = case["source"]
        for resolution in (512, 1024):
            size = edit_size(image, resolution)
            g = geometry_input(image, size)
            donor = geometry_input(donor_image, size)
            region = (
                boundary_mask(landmarks, size, name="back_window", segment=0) if landmarks else None
            )
            groups = {
                arm: intervene(g, arm, donor=donor, region=region, shift=(0, resolution // 16))
                for arm in arms
            }
            for arm, cues in groups.items():
                path = folder / str(resolution) / (arm + ".json")
                save_intervention(
                    path,
                    cues,
                    source,
                    {
                        "arm": arm,
                        "strength": 0 if arm == "disabled" else 1,
                        "shift_xy": [resolution // 16, 0] if arm == "shifted" else [0, 0],
                        "donor_source_sha256": sha256(ROOT / donor_case["source"])
                        if arm == "unrelated"
                        else None,
                        "selected_boundary": "back_window_sill"
                        if arm in ("removed", "random_removed")
                        else None,
                        "landmark_sha256": sha256(landmark_path) if landmarks else None,
                    },
                )
                if resolution == 1024:
                    runs.append(
                        {
                            "case": case["id"],
                            "domain": case["domain"],
                            "arm": arm,
                            "source": source_relative,
                            "instruction": case["instruction"],
                            "intervention": path.relative_to(ROOT).as_posix(),
                            "strength": 0 if arm == "disabled" else 1,
                        }
                    )
            if resolution != 512:
                continue
            with torch.no_grad():
                base = model(g)
                mean = base.mean(1, keepdim=True)
                stats = {
                    "feature_rms": float(base.square().mean().sqrt()),
                    "spatial_mean_energy_fraction": float(
                        mean.square().sum() * base.shape[1] / base.square().sum()
                    ),
                    "projection_bias_rms": float(model.projection.bias.square().mean().sqrt()),
                }
                changes = {}
                for arm, cues in groups.items():
                    # Disabled has no injected residual, even though encoder inputs are unchanged.
                    features = torch.zeros_like(base) if arm == "disabled" else model(cues)
                    delta = features - base
                    changes[arm] = {
                        "input_edge_L2": float((cues[:, 0] - g[:, 0]).norm()),
                        "relative_feature_L2": float(delta.norm() / base.norm()),
                        "feature_delta_rms": float(delta.square().mean().sqrt()),
                        "changed_edge_pixels": int((cues[:, 0] != g[:, 0]).sum()),
                    }
                    heatmap(
                        delta.square().mean(-1).sqrt().reshape(size[1] // 16, size[0] // 16),
                        folder / (arm + "-feature-delta.png"),
                    )
                    Image.fromarray(np.uint8(cues[0, 0].numpy() * 255)).save(
                        folder / (arm + "-edges.png")
                    )
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    bf = model(g).float()
                stats["cpu_bf16_feature_relative_L2_vs_fp32"] = float(
                    (bf - base).norm() / base.norm()
                )
            probe = g.clone().requires_grad_(True)
            saliency = torch.zeros_like(probe)
            for seed in range(4):
                features = model(probe)
                gen = torch.Generator().manual_seed(901 + seed)
                direction = torch.randn(features.shape, generator=gen) / features.numel() ** 0.5
                grad = torch.autograd.grad((features * direction).sum(), probe)[0]
                saliency += grad.abs() / 4
            scale = heatmap(saliency[0, 0], folder / "edge-jacobian.png")
            stats["encoder_input_gradient_finite"] = bool(torch.isfinite(saliency).all())
            stats["edge_gradient_L1"] = float(saliency[:, 0].sum())
            stats["xy_gradient_L1"] = float(saliency[:, 5:].sum())
            if landmarks:
                structural = boundary_mask(landmarks, size)
                edge = g[0, 0] > 0
                a = saliency[0, 0][structural & edge]
                b = saliency[0, 0][~structural & edge]
                stats["annotated_edge_mean_abs_gradient"] = float(a.mean())
                stats["other_edge_mean_abs_gradient"] = float(b.mean())
                stats["annotated_to_other_mean_gradient_ratio"] = float(a.mean() / b.mean())
                overlay = image.resize(size).convert("RGB")
                pixels = np.asarray(overlay).copy()
                pixels[structural.numpy()] = [255, 40, 40]
                Image.fromarray(pixels).save(folder / "landmark-overlay.png")
            rows.append(
                {
                    "case": case["id"],
                    "domain": case["domain"],
                    "stats": stats,
                    "interventions": changes,
                    "jacobian_display_max": scale,
                }
            )
            body.append(
                f"<h2>{case['id']}</h2><p>Source: {html.escape(case['provenance']['author'])}; <a href='{html.escape(case['provenance']['source_url'], quote=True)}'>original</a>; {html.escape(case['provenance']['license'])}. Diagnostic derivatives only.</p>"
            )
            body.append(
                f"<p>{html.escape(json.dumps(stats))}</p><img width='384' src='{case['id']}/edge-jacobian.png' alt='encoder input Jacobian'>"
            )
            if landmarks:
                body.append(
                    f"<img width='384' src='{case['id']}/landmark-overlay.png' alt='projected source architecture'>"
                )
            for arm in arms:
                body.append(
                    f"<figure style='display:inline-block'><figcaption>{arm}: edge input</figcaption><img width='256' src='{case['id']}/{arm}-edges.png'></figure>"
                )
    write_json(
        output / "diagnostics.json",
        {
            "checkpoint_sha256": EXPECTED,
            "trainable_parameter_count": sum(p.numel() for p in model.parameters()),
            "scope": "Calibrated encoder only. Jacobian uses four seeded random feature projections, NOT full-Qwen flow loss. GroupNorm can transmit local edits globally. No generated-architecture causal effect established.",
            "rows": rows,
        },
    )
    write_json(
        output / "plan.json",
        {
            "status": "prepared_not_cloud_executed",
            "checkpoint": CHECKPOINT.as_posix(),
            "checkpoint_sha256": EXPECTED,
            "lora": protocol["checkpoint"],
            "lora_sha256": protocol["adapter_sha256"],
            "settings": {"size": 1024, "seed": 42, **protocol["settings"]},
            "runs": runs,
            "new_generations": len(runs),
            "landmark_review_required": True,
            "no_paid_authorization": True,
        },
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RGA causality diagnostics</title><body style="font:16px system-ui;max-width:1400px;margin:auto">'
        + "\n".join(body),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(rows),
                "planned_full_model_images": len(runs),
                "full_model_executed": False,
            }
        )
    )


def run(plan_path, output, deadline_unix, include_contrast=False):
    # A process deadline is an additional guard, never a substitute for Pod deletion.
    if deadline_unix is None or not 600 < deadline_unix - time.time() <= 3600:
        raise ValueError("supply a future execution deadline, 10-60 minutes away")
    plan = json.loads(plan_path.read_text())
    if (
        sha256(ROOT / plan["checkpoint"] / "adapter.safetensors") != plan["checkpoint_sha256"]
        or sha256(ROOT / plan["lora"] / "adapter.safetensors") != plan["lora_sha256"]
    ):
        raise ValueError("checkpoint changed")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    try:
        for row in plan["runs"]:
            if deadline_unix - time.time() < 600:
                break
            meta = edit(
                ROOT / row["source"],
                row["instruction"],
                output / row["case"] / row["arm"],
                EditConfig(**plan["settings"]),
                ROOT / plan["lora"],
                rga_checkpoint=ROOT / plan["checkpoint"],
                rga_strength=row["strength"],
                rga_intervention=ROOT / row["intervention"],
            )
            results.append({"case": row["case"], "arm": row["arm"], "metadata": meta})
            write_json(output / "completed.json", results)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "case": row["case"],
                        "arm": row["arm"],
                        "seconds": meta["total_seconds"],
                    }
                ),
                flush=True,
            )
            gc.collect()
            torch.cuda.empty_cache()
        if include_contrast and len(results) == len(plan["runs"]):
            for row in (r for r in plan["runs"] if r["arm"] == "correct"):
                if deadline_unix - time.time() < 600:
                    break
                arm = "edge_contrast_untrained"
                meta = edit(
                    ROOT / row["source"],
                    row["instruction"],
                    output / row["case"] / arm,
                    EditConfig(**plan["settings"]),
                    ROOT / plan["lora"],
                    rga_checkpoint=ROOT / plan["checkpoint"],
                    rga_strength=1,
                    rga_intervention=ROOT / row["intervention"],
                    rga_edge_contrast=True,
                )
                results.append({"case": row["case"], "arm": arm, "metadata": meta})
                write_json(output / "completed.json", results)
                print(
                    json.dumps(
                        {
                            "completed": len(results),
                            "case": row["case"],
                            "arm": arm,
                            "seconds": meta["total_seconds"],
                        }
                    ),
                    flush=True,
                )
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        body = [
            "<h1>Full-model geometry interventions</h1><p>Pixel differences do not establish architectural causality. Native outputs; inspect visible landmarks and occlusions before assigning judgments.</p>"
        ]
        cases = {
            r["id"]: r
            for r in json.loads((ROOT / "configs/v07_evaluation.json").read_text())["cases"]
        }
        for result in results:
            case, arm = result["case"], result["arm"]
            credit = cases[case]["provenance"]
            body.append(
                f"<h2>{html.escape(case)} / {html.escape(arm)}</h2><p>{html.escape(credit['author'])}; {html.escape(credit['license'])}; <a href='{html.escape(credit['source_url'], quote=True)}'>source</a></p><a href='{case}/{arm}/output.png'><img width='512' src='{case}/{arm}/output.png'></a>"
            )
        (output / "index.html").write_text(
            '<!doctype html><meta charset="utf-8">' + "\n".join(body), encoding="utf-8"
        )
        write_json(
            output / "execution.json",
            {
                "completed": len(results),
                "planned_primary": len(plan["runs"]),
                "completed_primary": sum(r["arm"] != "edge_contrast_untrained" for r in results),
                "optional_contrast_completed": sum(
                    r["arm"] == "edge_contrast_untrained" for r in results
                ),
                "deadline_unix": deadline_unix,
                "finished_unix": time.time(),
                "pod_termination": "must be verified separately by authenticated API",
            },
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=ROOT / "outputs/v10-causality/plan.json")
    parser.add_argument("--deadline-unix", type=float)
    parser.add_argument("--include-edge-contrast", action="store_true")
    args = parser.parse_args()
    prepare(args.output.resolve()) if args.action == "prepare" else run(
        args.plan, args.output, args.deadline_unix, args.include_edge_contrast
    )
