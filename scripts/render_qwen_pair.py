"""Run inside Blender on an authorized scene: blender -b scene.blend -P this.py -- --output ...

Renders two states of the SAME scene/camera. Output is unreviewed synthetic data, not photographs.
Never saves or modifies the supplied .blend file. All external asset rights must be documented.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--assets",
        type=Path,
        required=True,
        help="Per-asset rights ledger, including scene, textures and HDRIs",
    )
    parser.add_argument("--collection", default="RoomCraftFurniture")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    assets = json.loads(args.assets.read_text(encoding="utf-8"))
    if not assets.get("assets") or any(
        a.get("commercial_training") is not True or not a.get("license_url") or not a.get("sha256")
        for a in assets["assets"]
    ):
        raise ValueError(
            "rights ledger incomplete; do not infer permission from asset availability"
        )
    if args.output.exists():
        raise ValueError("render output must be new")
    scene = bpy.context.scene
    collection = bpy.data.collections.get(args.collection)
    if not collection or not collection.all_objects or not scene.camera:
        raise ValueError("need active camera and a furniture-only collection")
    if scene.camera in collection.all_objects.values() or any(
        o.type == "LIGHT" for o in collection.all_objects
    ):
        raise ValueError("camera/environment lights must stay outside furniture collection")
    args.output.mkdir(parents=True)
    scene.render.engine = "CYCLES"
    scene.cycles.seed = args.seed
    scene.cycles.use_animated_seed = False
    scene.render.image_settings.file_format = "PNG"
    camera = {
        "matrix_world": [list(r) for r in scene.camera.matrix_world],
        "lens": scene.camera.data.lens,
        "sensor_width": scene.camera.data.sensor_width,
        "shift_x": scene.camera.data.shift_x,
        "shift_y": scene.camera.data.shift_y,
    }
    original_hidden = collection.hide_render
    try:
        for state, hidden in (("source", True), ("target", False)):
            collection.hide_render = hidden
            scene.render.filepath = str((args.output / f"{state}.png").resolve())
            bpy.ops.render.render(write_still=True)
    finally:
        collection.hide_render = original_hidden
    record = {
        "kind": "render_pair",
        "status": "REQUIRES_ALIGNMENT_AND_RIGHTS_REVIEW",
        "blender": bpy.app.version_string,
        "scene_file": Path(bpy.data.filepath).name,
        "scene_sha256": hashlib.sha256(Path(bpy.data.filepath).read_bytes()).hexdigest(),
        "assets": assets,
        "camera": camera,
        "seed": args.seed,
        "frame": scene.frame_current,
        "resolution": [
            scene.render.resolution_x,
            scene.render.resolution_y,
            scene.render.resolution_percentage,
        ],
        "samples": scene.cycles.samples,
        "view_transform": scene.view_settings.view_transform,
        "exposure": scene.view_settings.exposure,
        "furniture_collection": args.collection,
    }
    (args.output / "render_provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
