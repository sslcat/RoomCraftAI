"""Blender camera projections of known architectural openings, not inferred segmentation."""

import argparse
import json
import sys
from pathlib import Path


def project(bpy, family):
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    scene = bpy.context.scene
    f = family
    w, d, h = f["width"], f["depth"], f["height"]
    lines = []

    def polyline(name, category, points):
        projected = [world_to_camera_view(scene, scene.camera, Vector(p)) for p in points]
        lines.append(
            {
                "name": name,
                "category": category,
                "world": points,
                "normalized_xy": [[p.x, 1 - p.y] for p in projected],
                "camera_depth": [p.z for p in projected],
            }
        )

    x, half, z = f["window_offset"], f["window_width"] / 2, f["window_sill"]
    polyline(
        "back_window",
        "window",
        [
            (x - half, d, z),
            (x + half, d, z),
            (x + half, d, h - 0.32),
            (x - half, d, h - 0.32),
            (x - half, d, z),
        ],
    )
    if f["side_window"]:
        side = w / 2 if f["door_side"] == "left" else -w / 2
        y = d * 0.58
        polyline(
            "side_window",
            "window",
            [
                (side, y - 0.85, 0.95),
                (side, y + 0.85, 0.95),
                (side, y + 0.85, h - 0.35),
                (side, y - 0.85, h - 0.35),
                (side, y - 0.85, 0.95),
            ],
        )
    side = -w / 2 if f["door_side"] == "left" else w / 2
    y = f["door_center"]
    polyline(
        "door",
        "door",
        [
            (side, y - 0.475, 0),
            (side, y - 0.475, 2.18),
            (side, y + 0.475, 2.18),
            (side, y + 0.475, 0),
        ],
    )
    for x in (-w / 2, w / 2):
        polyline(f"corner_{x}", "wall", [(x, d, 0), (x, d, h)])
        for z in (0, h):
            polyline(f"side_{x}_{z}", "wall", [(x, 0.05, z), (x, d, z)])
    for z in (0, h):
        polyline(f"back_{z}", "wall", [(-w / 2, d, z), (w / 2, d, z)])
    return {
        "schema": "roomcraft-architectural-projection-v1",
        "coordinate_system": "normalized image x/y, top-left origin; outside-frame points retained; depth>0 required",
        "limitation": "Known source-room opening/corner projections; not target visibility masks. Furniture may occlude architecture. Frame molding extends beyond opening rectangles.",
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "camera_matrix": [list(r) for r in scene.camera.matrix_world],
        "camera_lens": scene.camera.data.lens,
        "polylines": lines,
    }


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    for name in ("family_05_v00", "family_18_v00", "family_19_v00"):
        folder = root / "data/v05/pairs" / name
        record = json.loads((folder / "render.json").read_text())
        bpy.ops.wm.open_mainfile(filepath=str(folder / "scene.blend"))
        data = project(bpy, record["family"])
        data.update(
            scene_id=name,
            source_sha256=record["sha256"]["source"],
            scene_sha256=record["scene_sha256"],
            provenance="Saved original Blender scene and its actual camera; source bytes must match V0.6 manifest.",
        )
        (args.output / (name + ".json")).write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
