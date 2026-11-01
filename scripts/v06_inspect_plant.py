"""Read-only Blender mesh-height diagnostic for a saved V0.5 scene; does not save changes."""

import json
import sys
from pathlib import Path

import bpy


def main():
    output = Path(sys.argv[sys.argv.index("--") + 1])
    rows = []
    for obj in bpy.data.objects:
        if obj.type != "MESH" or "plant" not in obj.name.lower():
            continue
        heights = sorted((obj.matrix_world @ v.co).z for v in obj.data.vertices)
        by_material = {}
        for polygon in obj.data.polygons:
            mat = obj.material_slots[polygon.material_index].name
            zs = [(obj.matrix_world @ obj.data.vertices[i].co).z for i in polygon.vertices]
            values = by_material.setdefault(mat, [float("inf"), -float("inf")])
            values[0] = min(values[0], *zs)
            values[1] = max(values[1], *zs)
        rows.append(
            {
                "name": obj.name,
                "z_range": [heights[0], heights[-1]],
                "material_z_ranges": by_material,
            }
        )
    output.write_text(
        json.dumps({"scene": bpy.data.filepath, "objects": rows}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
