"""Blender: repair the diagnosed imported plant defect into NEW folders; preserve V0.5."""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_room_dataset import Builder, digest, dump, invariant_state  # noqa: E402


def repair(identifier, output_root):
    project = Path(__file__).resolve().parents[1]
    source = project / "data/v05/pairs" / identifier
    output = output_root / "pairs" / identifier
    if output.exists():
        raise ValueError("repair output already exists; no overwrite")
    original = json.loads((source / "render.json").read_text())
    if "potted_plant_02" not in original["cc0_asset_ids"]:
        raise ValueError("pair does not use the diagnosed asset")
    if digest(source / "scene.blend") != original["scene_sha256"]:
        raise ValueError("parent scene changed")
    bpy.ops.wm.open_mainfile(filepath=str(source / "scene.blend"))
    furniture = bpy.data.collections["RoomCraftFurniture"]

    def state_hash():
        return hashlib.sha256(
            json.dumps(invariant_state(bpy, furniture), sort_keys=True).encode()
        ).hexdigest()

    if state_hash() != original["architecture_camera_lighting_sha256"]:
        raise ValueError("saved parent architectural state differs")
    placement = next(p for p in original["placements"] if p["name"] == "Potted plant")
    for obj in list(furniture.objects):
        if obj.name.startswith("CC0_potted_plant_02_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    builder = Builder.__new__(Builder)
    builder.bpy, builder.config, builder.furniture = bpy, original["settings"], furniture
    builder.asset_templates, builder.used_assets = {}, set()
    added = builder.asset("potted_plant_01", *placement["center"], 0.65)
    bpy.context.view_layer.update()
    pot = next(o for o in added if o.name.endswith("_pot"))
    floor_z = min((pot.matrix_world @ v.co).z for v in pot.data.vertices)
    if abs(floor_z) > 1e-5:
        raise ValueError("replacement pot is not grounded")
    # Every addition remains in the furniture collection; architecture, camera and lights match.
    for hidden in (True, False):
        furniture.hide_render = hidden
        bpy.context.view_layer.update()
        if state_hash() != original["architecture_camera_lighting_sha256"]:
            raise ValueError("repair changed paired architectural/camera/light state")
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"
    if not any(d.use for d in prefs.devices):
        raise RuntimeError("OptiX GPU unavailable")
    scene = bpy.context.scene
    scene.cycles.device = "GPU"
    output.mkdir(parents=True)
    for name in ("source.png", "invariant_state.json"):
        shutil.copy2(source / name, output / name)
    scene.render.filepath = str((output / "target.png").resolve())
    started = time.perf_counter()
    bpy.ops.render.render(write_still=True)
    seconds = time.perf_counter() - started
    bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str((output / "scene.blend").resolve()), compress=True)
    record = dict(original)
    record.update(
        status="repaired_pending_visual_review",
        scene_sha256=digest(output / "scene.blend"),
        sha256={"source": digest(output / "source.png"), "target": digest(output / "target.png")},
        generation_seconds={"source": 0.0, "target": seconds},
        cc0_asset_ids=sorted(
            (set(original["cc0_asset_ids"]) - {"potted_plant_02"}) | {"potted_plant_01"}
        ),
        repair={
            "reason": "Plant 02 pot floats 0.0862 m above floor in saved scene; whole-asset bounds hid the defect",
            "action": "Replace with already licensed, grounded plant 01; preserve all other furniture and architecture",
            "script_sha256": digest(__file__),
            "parent_render_sha256": digest(source / "render.json"),
            "parent_target_sha256": original["sha256"]["target"],
            "source_reused_byte_identical": True,
            "replacement_pot_min_z_m": floor_z,
        },
    )
    dump(output / "render.json", record)
    print("REPAIR_COMPLETE " + identifier, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    for identifier in args.ids:
        repair(identifier, args.output)


if __name__ == "__main__":
    main()
