"""Original procedural interiors. Run using Blender --background --factory-startup --python.

Architecture/material recipes authored for RoomCraft; optional individually verified CC0 furniture.
Does not mark renders accepted: visual review and V0.4 sealing are separate steps.
"""

import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def family_parameters(index, seed):
    rng = random.Random(seed + index * 1009)
    kind = ("living", "bedroom", "dining")[index % 3]
    family = {
        "family": f"family_{index:02d}",
        "index": index,
        "kind": kind,
        "width": round(rng.uniform(4.6, 6.8), 3),
        "depth": round(rng.uniform(5.2, 7.0), 3),
        "height": round(rng.uniform(2.7, 3.25), 3),
        "window_width": round(rng.uniform(1.65, 2.7), 3),
        "window_offset": round(rng.uniform(-0.55, 0.55), 3),
        "window_sill": round(rng.uniform(0.75, 1.0), 3),
        "side_window": index % 3 != 1,
        "door_side": "left" if index % 2 == 0 else "right",
        "wall_palette": index % 5,
        "floor_palette": index % 3,
        "floor_type": "limestone" if index % 5 == 4 else "timber",
        "ceiling_palette": index % 3,
        "arrangement": index % 2,
    }
    family["door_center"] = 1.375 if kind == "living" else round(family["depth"] * 0.45, 3)
    if kind == "bedroom" and (family["depth"] < 5.6 or family["width"] / family["depth"] > 1.1):
        family["arrangement"] = 0
    return family


def variant_parameters(family, variant, seed):
    rng = random.Random(seed + family["index"] * 1009 + variant * 7919)
    return {
        "id": f"{family['family']}_v{variant:02d}",
        "seed": seed + family["index"] * 1009 + variant * 7919,
        "variant": variant,
        "camera_x": round((variant - 2) * 0.16 + (family["index"] % 3 - 1) * 0.3, 3),
        "camera_y": round(rng.uniform(0.23, 0.38), 3),
        "camera_height": round(rng.uniform(1.42, 1.65), 3),
        "lens": 23 + variant % 3,
        "fabric_palette": (family["index"] + variant) % 5,
        "sun_elevation": 0.38 + variant * 0.065,
        "sun_rotation": 0.5 + family["index"] * 0.19,
        "artificial_watts": 35 if variant % 2 == 0 else 100,
    }


class Builder:
    def __init__(self, bpy, family, variant, config):
        self.bpy = bpy
        self.f = family
        self.v = variant
        self.rng = random.Random(variant["seed"])
        self.furniture = bpy.data.collections.new("RoomCraftFurniture")
        bpy.context.scene.collection.children.link(self.furniture)
        self.current_furniture = False
        self.placements = []
        self.materials = {}
        self.config = config
        self.asset_templates = {}
        self.used_assets = set()

    def asset(self, identifier, x, y, width, rotation=0):
        from mathutils import Matrix, Vector

        bpy = self.bpy
        folder = ROOT / self.config["asset_root"] / identifier
        if identifier not in self.asset_templates:
            existing_images = set(bpy.data.images)
            with bpy.data.libraries.load(str(folder / f"{identifier}_2k.blend"), link=False) as (
                source,
                target,
            ):
                target.objects = source.objects
            objects = [o for o in target.objects if o and o.type == "MESH"]
            for image in set(bpy.data.images) - existing_images:
                file = folder / "textures" / Path(image.filepath).name
                if not file.exists():
                    raise ValueError(f"asset texture missing: {file}")
                image.filepath = str(file.resolve())
                image.reload()
            points = [o.matrix_world @ Vector(c) for o in objects for c in o.bound_box]
            minimum = Vector(tuple(min(p[i] for p in points) for i in range(3)))
            maximum = Vector(tuple(max(p[i] for p in points) for i in range(3)))
            self.asset_templates[identifier] = (objects, minimum, maximum)
        objects, minimum, maximum = self.asset_templates[identifier]
        scale = width / (maximum.x - minimum.x)
        origin = Vector(((minimum.x + maximum.x) / 2, (minimum.y + maximum.y) / 2, minimum.z))
        transform = (
            Matrix.Translation(Vector((x, y, 0)))
            @ Matrix.Rotation(rotation, 4, "Z")
            @ Matrix.Scale(scale, 4)
            @ Matrix.Translation(-origin)
        )
        result = []
        for template in objects:
            obj = template.copy()
            obj.parent = None
            obj.name = f"CC0_{identifier}_{template.name}"
            self.furniture.objects.link(obj)
            obj.matrix_world = transform @ template.matrix_world
            result.append(obj)
        self.used_assets.add(identifier)
        return result

    def finish(self, obj, name, material):
        obj.name = name
        if material:
            obj.data.materials.append(material)
        if self.current_furniture:
            for collection in list(obj.users_collection):
                collection.objects.unlink(obj)
            self.furniture.objects.link(obj)
        return obj

    def box(self, name, pos, size, mat, bevel=0.01, rotation=0):
        bpy = self.bpy
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos)
        obj = bpy.context.object
        obj.dimensions = size
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.rotation_euler[2] = rotation
        if bevel:
            mod = obj.modifiers.new("Soft manufactured edges", "BEVEL")
            mod.width = min(bevel, min(size) * 0.45)
            mod.segments = 4
            obj.modifiers.new("Weighted surface normals", "WEIGHTED_NORMAL")
        return self.finish(obj, name, mat)

    def sphere(self, name, pos, scale, mat):
        bpy = self.bpy
        bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, location=pos)
        obj = bpy.context.object
        obj.scale = scale
        for p in obj.data.polygons:
            p.use_smooth = True
        return self.finish(obj, name, mat)

    def cylinder(self, name, pos, radius, depth, mat, vertices=48):
        bpy = self.bpy
        bpy.ops.mesh.primitive_cylinder_add(
            vertices=vertices, radius=radius, depth=depth, location=pos
        )
        obj = bpy.context.object
        bevel = obj.modifiers.new("Edge radius", "BEVEL")
        bevel.width = min(0.012, radius * 0.1)
        bevel.segments = 3
        obj.modifiers.new("Normals", "WEIGHTED_NORMAL")
        return self.finish(obj, name, mat)

    def rod(self, name, start, end, radius, mat):
        from mathutils import Vector

        delta = Vector(end) - Vector(start)
        obj = self.cylinder(name, (Vector(start) + Vector(end)) / 2, radius, delta.length, mat, 16)
        obj.rotation_euler = delta.to_track_quat("Z", "Y").to_euler()
        return obj

    def material(self, name, color, roughness=0.5, texture=None):
        mat = self.bpy.data.materials.new(name)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        shader = nodes.get("Principled BSDF")
        shader.inputs["Base Color"].default_value = (*color, 1)
        shader.inputs["Roughness"].default_value = roughness
        if texture:
            coord = nodes.new("ShaderNodeTexCoord")
            mapping = nodes.new("ShaderNodeVectorMath")
            mapping.operation = "MULTIPLY"
            mapping.inputs[1].default_value = (1.2, 28, 3) if texture == "wood" else (140, 140, 140)
            links.new(coord.outputs["Generated"], mapping.inputs[0])
            noise = nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = 2
            noise.inputs["Detail"].default_value = 3
            links.new(mapping.outputs["Vector"], noise.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].position = 0.18
            ramp.color_ramp.elements[1].position = 0.82
            ramp.color_ramp.elements[0].color = (
                *[c * (0.5 if texture == "wood" else 0.72) for c in color],
                1,
            )
            ramp.color_ramp.elements[1].color = (*[min(1, c * 1.13) for c in color], 1)
            links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
            links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = 0.18
            bump.inputs["Distance"].default_value = 0.001 if texture == "fabric" else 0.0005
            links.new(noise.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs["Normal"], shader.inputs["Normal"])
            if texture == "fabric":
                shader.inputs["Sheen Weight"].default_value = 0.22
        return mat

    def make_materials(self):
        walls = [
            (0.76, 0.72, 0.64),
            (0.63, 0.68, 0.64),
            (0.75, 0.69, 0.62),
            (0.69, 0.72, 0.74),
            (0.83, 0.80, 0.74),
        ]
        fabrics = [
            (0.57, 0.51, 0.40),
            (0.22, 0.34, 0.30),
            (0.54, 0.25, 0.15),
            (0.21, 0.29, 0.37),
            (0.73, 0.69, 0.59),
        ]
        woods = [(0.38, 0.23, 0.11), (0.23, 0.12, 0.055), (0.54, 0.38, 0.21)]
        self.materials = {
            "wall": self.material("Painted mineral plaster", walls[self.f["wall_palette"]], 0.82),
            "white": self.material("Warm white trim", (0.85, 0.83, 0.77), 0.55),
            "wood": self.material("Natural timber", woods[self.f["floor_palette"]], 0.38, "wood"),
            "fabric": self.material(
                "Woven upholstery", fabrics[self.v["fabric_palette"]], 0.9, "fabric"
            ),
            "linen": self.material("Undyed linen", (0.79, 0.75, 0.65), 0.95, "fabric"),
            "rug": self.material("Wool flatweave", (0.43, 0.40, 0.33), 1, "fabric"),
            "dark": self.material("Dark bronze", (0.045, 0.035, 0.025), 0.3),
            "ceramic": self.material("Unglazed ceramic", (0.49, 0.39, 0.26), 0.65),
            "leaf": self.material("Leaf green", (0.075, 0.19, 0.06), 0.55),
            "soil": self.material("Potting soil", (0.033, 0.022, 0.012), 1),
            "stone": self.material("Honed limestone", (0.51, 0.49, 0.43), 0.72, "stone"),
            "ceiling": self.material(
                "Matte ceiling plaster",
                [(0.85, 0.83, 0.78), (0.72, 0.73, 0.69), (0.9, 0.89, 0.85)][
                    self.f["ceiling_palette"]
                ],
                0.85,
            ),
        }

    def wall_with_openings(self, name, axis, fixed, low, high, height, openings):
        """Disjoint wall rectangles: true apertures, no hidden wall behind glazing/doors."""
        edges = sorted({low, high, *[o[k] for o in openings for k in (0, 1)]})
        for i, (a, b) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            middle = (a + b) / 2
            holes = [o for o in openings if o[0] <= middle <= o[1]]
            spans = [(0, height)] if not holes else [(0, holes[0][2]), (holes[0][3], height)]
            for j, (z0, z1) in enumerate(spans):
                if z1 - z0 < 0.001:
                    continue
                pos = (
                    ((a + b) / 2, fixed, (z0 + z1) / 2)
                    if axis == "X"
                    else (fixed, (a + b) / 2, (z0 + z1) / 2)
                )
                size = (b - a, 0.18, z1 - z0) if axis == "X" else (0.18, b - a, z1 - z0)
                self.box(f"{name}_{i}_{j}", pos, size, self.materials["wall"], 0)

    def window(self, name, axis, fixed, center, width, sill, height):
        mat = self.materials["white"]

        def pos(u, z):
            return (u, fixed, z) if axis == "X" else (fixed, u, z)

        def size(u, z):
            return (u, 0.22, z) if axis == "X" else (0.22, u, z)

        for u in (center - width / 2, center, center + width / 2):
            self.box(
                name + "_upright", pos(u, sill + height / 2), size(0.055, height + 0.08), mat, 0.006
            )
        for z in (sill, sill + height):
            self.box(name + "_rail", pos(center, z), size(width + 0.12, 0.065), mat, 0.006)
        self.box(
            name + "_sill",
            pos(center, sill - 0.035),
            (width + 0.25, 0.36, 0.055) if axis == "X" else (0.36, width + 0.25, 0.055),
            mat,
            0.008,
        )
        # Clear glass slab, real transmission; no image texture or outside-photo asset.
        glass = self.material(name + " glazing", (0.96, 0.99, 1), 0.02)
        bsdf = glass.node_tree.nodes.get("Principled BSDF")
        bsdf.inputs["Transmission Weight"].default_value = 1
        bsdf.inputs["IOR"].default_value = 1.45
        self.box(
            name + "_glass",
            pos(center, sill + height / 2),
            (width, 0.006, height) if axis == "X" else (0.006, width, height),
            glass,
            0,
        )

    def architecture(self):
        f, m = self.f, self.materials
        w, d, h = f["width"], f["depth"], f["height"]
        dc = f["door_center"]
        self.box("Subfloor", (0, d / 2, -0.07), (w + 0.2, d + 0.2, 0.12), m["wood"], 0)
        # Individual staggered floorboards with physical joints and subtle tonal variation.
        board_width = 0.6 if f["floor_type"] == "limestone" else 0.19
        board_length = 0.6 if f["floor_type"] == "limestone" else 1.6
        floor_material = m["stone"] if f["floor_type"] == "limestone" else m["wood"]
        for row in range(math.ceil(d / board_width)):
            y = row * board_width
            plank_width = min(board_width, d - y)
            offset = 0 if f["floor_type"] == "limestone" else (row % 3) * 0.53
            x = -w / 2 - offset
            while x < w / 2:
                a, b = max(x, -w / 2), min(x + board_length, w / 2)
                if b - a > 0.01:
                    self.box(
                        "Stone floor tile"
                        if f["floor_type"] == "limestone"
                        else "Timber floorboard",
                        ((a + b) / 2, y + plank_width / 2, -0.006),
                        (b - a - 0.002, plank_width - 0.0015, 0.012),
                        floor_material,
                        0.001,
                    )
                x += board_length
        self.box("Ceiling", (0, d / 2, h + 0.07), (w + 0.3, d + 0.2, 0.14), m["ceiling"], 0)
        wc, ww, sill = f["window_offset"], f["window_width"], f["window_sill"]
        wh = h - sill - 0.32
        self.wall_with_openings(
            "Back wall", "X", d, -w / 2, w / 2, h, [(wc - ww / 2, wc + ww / 2, sill, sill + wh)]
        )
        self.window("Back window", "X", d, wc, ww, sill, wh)
        for side, x in (("left", -w / 2), ("right", w / 2)):
            openings = []
            if f["door_side"] == side:
                openings.append((dc - 0.475, dc + 0.475, 0, 2.18))
            if f["side_window"] and side != f["door_side"]:
                center = d * 0.58
                openings.append((center - 0.85, center + 0.85, 0.95, h - 0.35))
                self.window(side + " window", "Y", x, center, 1.7, 0.95, h - 1.3)
            self.wall_with_openings(side + " wall", "Y", x, 0, d, h, openings)
            # Baseboard segmented around door; a closed door avoids impossible furniture occlusion.
            intervals = [(0, dc - 0.475), (dc + 0.475, d)] if side == f["door_side"] else [(0, d)]
            for a, b in intervals:
                self.box(
                    side + " baseboard",
                    (x * 0.995, (a + b) / 2, 0.065),
                    (0.055, b - a, 0.13),
                    m["white"],
                    0.008,
                )
            if side == f["door_side"]:
                inner = x + (0.06 if side == "left" else -0.06)
                self.box(
                    "Closed paneled door", (x, dc, 1.07), (0.05, 0.92, 2.14), m["white"], 0.006
                )
                for y in (dc - 0.5, dc + 0.5):
                    self.box("Door casing", (inner, y, 1.1), (0.08, 0.075, 2.2), m["white"], 0.006)
                self.box("Door header", (inner, dc, 2.2), (0.08, 1.08, 0.075), m["white"], 0.006)
                for z in (0.6, 1.57):
                    self.box(
                        "Door recessed panel",
                        (inner * 0.999, dc, z),
                        (0.012, 0.72, 0.72),
                        m["wall"],
                        0.008,
                    )
                self.rod(
                    "Door handle",
                    (inner, dc + 0.325, 1.02),
                    (inner, dc + 0.185, 1.02),
                    0.018,
                    m["dark"],
                )
        self.box("Back baseboard", (0, d - 0.1, 0.065), (w, 0.055, 0.13), m["white"], 0.008)
        self.box("Front wall behind camera", (0, -0.1, h / 2), (w + 0.2, 0.18, h), m["wall"], 0)
        # Original exterior courtyard, never a third-party background image.
        self.box("Exterior paving", (0, d + 4, -0.15), (30, 20, 0.1), m["white"], 0)
        self.box("Exterior boundary", (0, d + 6, 0.7), (24, 0.2, 1.7), m["wall"], 0.03)

    def lighting_camera(self):
        from mathutils import Vector

        bpy, f, v = self.bpy, self.f, self.v
        scene = bpy.context.scene
        world = bpy.data.worlds.new("Procedural daylight sky")
        scene.world = world
        world.use_nodes = True
        nodes, links = world.node_tree.nodes, world.node_tree.links
        sky = nodes.new("ShaderNodeTexSky")
        sky.sky_type = "MULTIPLE_SCATTERING"
        sky.sun_elevation = v["sun_elevation"]
        sky.sun_rotation = v["sun_rotation"]
        sky.sun_disc = True
        nodes.get("Background").inputs["Strength"].default_value = 0.35
        links.new(sky.outputs["Color"], nodes.get("Background").inputs["Color"])

        # Large exterior daylight source speeds converged interior indirect lighting.
        def area(name, pos, target, energy, size, color):
            data = bpy.data.lights.new(name, "AREA")
            data.energy, data.shape, data.size, data.color = energy, "DISK", size, color
            obj = bpy.data.objects.new(name, data)
            scene.collection.objects.link(obj)
            obj.location = pos
            obj.visible_camera = False
            obj.visible_glossy = False
            obj.visible_transmission = False
            obj.rotation_euler = (Vector(target) - obj.location).to_track_quat("-Z", "Y").to_euler()

        area(
            "Exterior window daylight",
            (0, f["depth"] + 0.8, 2.8),
            (0, 1.5, 0.7),
            450,
            3,
            (0.86, 0.93, 1),
        )
        for x in (-1.25, 1.25):
            for y in (f["depth"] * 0.33, f["depth"] * 0.68):
                area(
                    "Recessed ceiling light",
                    (x, y, f["height"] - 0.04),
                    (x, y, 0),
                    v["artificial_watts"],
                    0.3,
                    (1, 0.78, 0.55),
                )
                self.cylinder(
                    "Ceiling light trim",
                    (x, y, f["height"] - 0.025),
                    0.09,
                    0.025,
                    self.materials["white"],
                )
        bpy.ops.object.camera_add(location=(v["camera_x"], v["camera_y"], v["camera_height"]))
        camera = bpy.context.object
        camera.name = "RoomCraft locked camera"
        camera.rotation_euler = (
            (Vector((0, f["depth"] * 0.66, 1.12)) - camera.location)
            .to_track_quat("-Z", "Y")
            .to_euler()
        )
        camera.data.lens = v["lens"]
        camera.data.sensor_width = 36
        camera.data.clip_start = 0.05
        scene.camera = camera

    def transform_group(self, objects, pivot, destination, angle):
        from mathutils import Matrix, Vector

        transform = (
            Matrix.Translation(Vector(destination))
            @ Matrix.Rotation(angle, 4, "Z")
            @ Matrix.Translation(-Vector(pivot))
        )
        self.bpy.context.view_layer.update()
        for obj in objects:
            obj.matrix_world = transform @ obj.matrix_world
        old = self.placements
        self.placements = []
        for item in old:
            center = transform @ Vector((*item["center"], 0))
            w, d = item["footprint"]
            self.placement(
                item["name"],
                center.x,
                center.y,
                abs(math.cos(angle)) * w + abs(math.sin(angle)) * d,
                abs(math.sin(angle)) * w + abs(math.cos(angle)) * d,
            )

    def fit_camera(self):
        from bpy_extras.object_utils import world_to_camera_view
        from mathutils import Vector

        scene = self.bpy.context.scene
        self.bpy.context.view_layer.update()
        # Rugs may extend beyond the view; furniture must remain fully framed.
        points_world = [
            obj.matrix_world @ Vector(corner)
            for obj in self.furniture.objects
            if obj.type == "MESH" and "rug" not in obj.name.lower()
            for corner in obj.bound_box
        ]
        candidates = (
            (lens, cx, ax, az)
            for lens in (self.v["lens"], 22, 21, 20, 19, 18)
            for cx in (self.v["camera_x"], 0, 0.6, -0.6, 1.0, -1.0)
            for ax in (0, -0.6, 0.6)
            for az in (1.12, 0.95, 0.8, 0.65)
        )
        for lens, cx, aim_x, aim_height in candidates:
            scene.camera.location.x = cx
            scene.camera.data.lens = lens
            scene.camera.rotation_euler = (
                (Vector((aim_x, self.f["depth"] * 0.66, aim_height)) - scene.camera.location)
                .to_track_quat("-Z", "Y")
                .to_euler()
            )
            self.bpy.context.view_layer.update()
            points = [world_to_camera_view(scene, scene.camera, point) for point in points_world]
            if all(0.025 < p.x < 0.975 and 0.025 < p.y < 0.975 and p.z > 0 for p in points):
                return
        bad = [
            (o.name, list(world_to_camera_view(scene, scene.camera, o.matrix_world @ Vector(c))))
            for o in self.furniture.objects
            if o.type == "MESH"
            for c in o.bound_box
            if not (
                0.025
                < world_to_camera_view(scene, scene.camera, o.matrix_world @ Vector(c)).x
                < 0.975
                and 0.025
                < world_to_camera_view(scene, scene.camera, o.matrix_world @ Vector(c)).y
                < 0.975
            )
        ]
        raise ValueError(f"furnishing cannot fit camera: {bad[:6]}")

    def placement(self, name, x, y, width, depth):
        margin = 0.13
        if (
            abs(x) + width / 2 > self.f["width"] / 2 - margin
            or y - depth / 2 < 0.25
            or y + depth / 2 > self.f["depth"] - margin
        ):
            raise ValueError(f"furniture outside architectural clearance: {name}")
        # Reserve a 0.9 m deep clear entry zone in front of the side door.
        door_x = (-1 if self.f["door_side"] == "left" else 1) * (self.f["width"] / 2 - 0.45)
        if (
            abs(x - door_x) < (width + 0.9) / 2
            and abs(y - self.f["door_center"]) < (depth + 1.15) / 2
        ):
            raise ValueError(f"furniture blocks door clearance: {name}")
        self.placements.append(
            {"name": name, "center": [x, y], "footprint": [width, depth], "floor_z": 0}
        )

    def rug(self, x, y, width, depth):
        self.box(
            "Wool area rug", (x, y, 0.007), (width, depth, 0.014), self.materials["rug"], 0.006
        )
        for offset in (-width / 2 + 0.07, width / 2 - 0.07):
            self.box(
                "Woven rug border",
                (x + offset, y, 0.0145),
                (0.025, depth - 0.08, 0.001),
                self.materials["linen"],
                0,
            )

    def cloth(self, name, x, y, width, depth, height, mat, drape=0.0):
        bpy = self.bpy
        nx, ny = 56, 48
        vertices, faces = [], []
        for j in range(ny + 1):
            dy = depth * (j / ny - 0.5)
            for i in range(nx + 1):
                dx = width * (i / nx - 0.5)
                ripple = 0.009 * math.sin(19 * dx + 2 * math.cos(dy * 7)) * math.sin(
                    9 * dy
                ) + 0.004 * math.sin(36 * dy + dx * 5)
                drop = drape * max(0, abs(dx) - width * 0.43) + drape * max(0, -dy - depth * 0.41)
                vertices.append((x + dx, y + dy, height + ripple - drop))
                if i < nx and j < ny:
                    a = j * (nx + 1) + i
                    faces.append((a, a + 1, a + nx + 2, a + nx + 1))
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(vertices, [], faces)
        obj = bpy.data.objects.new(name, mesh)
        bpy.context.scene.collection.objects.link(obj)
        self.finish(obj, name, mat)
        for face in mesh.polygons:
            face.use_smooth = True
        solid = obj.modifiers.new("Cloth thickness", "SOLIDIFY")
        solid.thickness = 0.018
        subdiv = obj.modifiers.new("Soft cloth surface", "SUBSURF")
        subdiv.levels = 1
        return obj

    def soften_upholstery(self):
        bpy = self.bpy
        for obj in list(self.furniture.objects):
            if not any(word in obj.name.lower() for word in ("cushion", "pillow")):
                continue
            bpy.context.view_layer.objects.active = obj
            for mod in list(obj.modifiers):
                if mod.type == "BEVEL":
                    bpy.ops.object.modifier_apply(modifier=mod.name)
            subdivision = obj.modifiers.new("Upholstery microfold mesh", "SUBSURF")
            subdivision.subdivision_type = "SIMPLE"
            subdivision.levels = 3
            displacement = obj.modifiers.new("Upholstery irregularity", "DISPLACE")
            texture = bpy.data.textures.new(obj.name + " textile irregularity", type="CLOUDS")
            texture.noise_scale = 0.11
            texture.noise_depth = 2
            displacement.texture = texture
            displacement.strength = 0.007
            for face in obj.data.polygons:
                face.use_smooth = True

    def table(self, x, y, width, depth, height, name="table"):
        m = self.materials
        self.placement(name, x, y, width, depth)
        self.box(name + " tabletop", (x, y, height - 0.035), (width, depth, 0.07), m["wood"], 0.025)
        for dx in (-width / 2 + 0.13, width / 2 - 0.13):
            for dy in (-depth / 2 + 0.1, depth / 2 - 0.1):
                self.box(
                    name + " leg",
                    (x + dx, y + dy, (height - 0.07) / 2),
                    (0.065, 0.065, height - 0.07),
                    m["wood"],
                    0.009,
                )

    def chair(self, x, y, rotation=0, armchair=False):
        m = self.materials
        width, depth = (0.78, 0.78) if armchair else (0.49, 0.53)
        self.placement("Lounge chair" if armchair else "Dining chair", x, y, width, depth)
        if self.config.get("furnishing_mode") == "cc0_hybrid":
            self.asset(
                "modern_arm_chair_01" if armchair else "dining_chair_02",
                x,
                y,
                0.7 if armchair else 0.46,
                rotation,
            )
            return

        def transform(dx, dy, z):
            return (
                x + dx * math.cos(rotation) - dy * math.sin(rotation),
                y + dx * math.sin(rotation) + dy * math.cos(rotation),
                z,
            )

        for dx in (-width / 2 + 0.07, width / 2 - 0.07):
            for dy in (-depth / 2 + 0.07, depth / 2 - 0.07):
                self.box(
                    "Chair leg",
                    transform(dx, dy, 0.225),
                    (0.045, 0.045, 0.45),
                    m["wood"],
                    0.007,
                    rotation,
                )
        self.box(
            "Chair seat frame",
            transform(0, 0, 0.43),
            (width, depth, 0.065),
            m["wood"],
            0.035,
            rotation,
        )
        self.box(
            "Chair cushion",
            transform(0, -0.005, 0.49),
            (width - 0.05, depth - 0.03, 0.1),
            m["fabric"],
            0.045,
            rotation,
        )
        for dx in (-width / 2 + 0.04, width / 2 - 0.04):
            self.box(
                "Chair back post",
                transform(dx, depth / 2 - 0.05, 0.68),
                (0.04, 0.04, 0.5),
                m["wood"],
                0.007,
                rotation,
            )
        self.box(
            "Chair upholstered back",
            transform(0, depth / 2 - 0.06, 0.82),
            (width - 0.04, 0.085, 0.28),
            m["fabric"],
            0.04,
            rotation,
        )
        if armchair:
            for dx in (-width / 2 + 0.02, width / 2 - 0.02):
                self.box(
                    "Armrest",
                    transform(dx, 0, 0.66),
                    (0.07, depth, 0.065),
                    m["wood"],
                    0.025,
                    rotation,
                )

    def vase(self, x, y, z, flowers=False):
        m = self.materials
        self.sphere("Ceramic vase body", (x, y, z + 0.12), (0.11, 0.11, 0.12), m["ceramic"])
        self.cylinder("Vase neck", (x, y, z + 0.23), 0.045, 0.12, m["ceramic"])
        self.cylinder("Vase dark opening", (x, y, z + 0.292), 0.034, 0.002, m["soil"])
        if flowers:
            for i in range(7):
                angle = i * 2.399
                tip = (
                    x + 0.16 * math.cos(angle),
                    y + 0.16 * math.sin(angle),
                    z + 0.48 + i % 3 * 0.07,
                )
                self.rod("Botanical stem", (x, y, z + 0.26), tip, 0.003, m["leaf"])
                leaf = self.sphere("Botanical leaf", tip, (0.1, 0.035, 0.008), m["leaf"])
                leaf.rotation_euler = (0.3, 0.5, angle)

    def plant(self, x, y):
        m = self.materials
        self.placement("Potted plant", x, y, 0.65, 0.65)
        if self.config.get("furnishing_mode") == "cc0_hybrid":
            self.asset(
                "potted_plant_01" if self.f["index"] % 2 == 0 else "potted_plant_02", x, y, 0.65
            )
            return
        self.cylinder("Planter", (x, y, 0.2), 0.21, 0.4, m["ceramic"])
        self.cylinder("Soil", (x, y, 0.405), 0.19, 0.01, m["soil"])
        self.rod("Plant trunk", (x, y, 0.4), (x + 0.03, y, 1.55), 0.018, m["wood"])
        for i in range(28):
            angle = i * 2.399
            z = 0.65 + (i / 28) * 0.95
            radius = 0.2 + 0.08 * math.sin(i)
            tip = (x + radius * math.cos(angle), y + radius * math.sin(angle), z + 0.08)
            self.rod("Plant branch", (x, y, z), tip, 0.004, m["wood"])
            leaf = self.sphere("Plant leaf", tip, (0.17, 0.065, 0.009), m["leaf"])
            leaf.rotation_euler = (0.3, -0.35, angle)

    def lamp(self, x, y, base_z):
        m = self.materials
        self.cylinder("Lamp base", (x, y, base_z + 0.025), 0.12, 0.05, m["dark"])
        self.cylinder("Lamp stem", (x, y, base_z + 0.21), 0.018, 0.37, m["dark"])
        self.bpy.ops.mesh.primitive_cone_add(
            vertices=64, radius1=0.19, radius2=0.13, depth=0.24, location=(x, y, base_z + 0.45)
        )
        self.finish(self.bpy.context.object, "Non-emitting linen lampshade", m["linen"])

    def living(self):
        f, m = self.f, self.materials
        w, d = f["width"], f["depth"]
        sx, sy = (-0.4 if f["arrangement"] else 0.35), d - 0.8
        self.placement(
            "Sofa", sx, sy, 2.2 if self.config.get("furnishing_mode") == "cc0_hybrid" else 2.5, 0.98
        )
        for x in (-1.05, 1.05):
            for y in (-0.32, 0.32):
                self.box(
                    "Sofa timber foot",
                    (sx + x, sy + y, 0.11),
                    (0.075, 0.075, 0.22),
                    m["wood"],
                    0.01,
                )
        self.box("Sofa lower upholstered body", (sx, sy, 0.3), (2.5, 0.92, 0.28), m["fabric"], 0.08)
        self.box("Sofa back", (sx, sy + 0.36, 0.73), (2.5, 0.22, 0.68), m["fabric"], 0.085)
        for x in (-1.16, 1.16):
            self.box(
                "Sofa padded arm", (sx + x, sy - 0.015, 0.58), (0.2, 0.97, 0.44), m["fabric"], 0.085
            )
        for i in (-1, 0, 1):
            self.box(
                "Individual seat cushion",
                (sx + i * 0.71, sy - 0.08, 0.51),
                (0.69, 0.7, 0.18),
                m["fabric"],
                0.075,
            )
            cushion = self.box(
                "Sofa back cushion",
                (sx + i * 0.71, sy + 0.22, 0.82),
                (0.7, 0.19, 0.5),
                m["fabric"],
                0.08,
            )
            cushion.rotation_euler[0] = -0.13
        for x in (-0.88, 0.85):
            pillow = self.box(
                "Linen accent pillow",
                (sx + x, sy + 0.05, 0.83),
                (0.4, 0.16, 0.4),
                m["linen"],
                0.075,
            )
            pillow.rotation_euler = (-0.15, 0.08, x * 0.2)
        tx, ty = 0, d - 2.25
        if self.config.get("furnishing_mode") == "cc0_hybrid":
            for obj in list(self.furniture.objects):
                self.bpy.data.objects.remove(obj, do_unlink=True)
            self.asset("sofa_02", sx, sy, 2.15)
        if f["arrangement"]:
            self.transform_group(
                list(self.furniture.objects), (sx, sy, 0), (-w / 2 + 0.65, d - 1.45, 0), math.pi / 2
            )
            tx, ty = -w / 2 + 2.0, d - 2.1
        self.rug(tx, ty, 2.9 if f["arrangement"] else 3.45, 2.2 if f["arrangement"] else 2.7)
        self.table(tx, ty, 1.25, 0.68, 0.4, "Coffee table")
        self.vase(tx + 0.25, ty, 0.4)
        self.box(
            "Coffee table book",
            (tx - 0.28, ty - 0.05, 0.419),
            (0.29, 0.21, 0.035),
            m["linen"],
            0.003,
            0.12,
        )
        chair_x = w / 2 - 0.75 if f["arrangement"] else -w / 2 + 0.75
        self.chair(
            chair_x, d - (1.9 if f["arrangement"] else 2.5), -0.2 if chair_x > 0 else 0.2, True
        )
        self.plant(w / 2 - 0.48, d - 0.55)
        return "Furnish this living room with a tufted leather sofa, a timber coffee table with a book and ceramic vase, a wooden lounge chair with leather cushions, a wool area rug and a potted plant. Preserve the room architecture, doors, windows, camera and lighting."

    def bedroom(self):
        f, m = self.f, self.materials
        d = f["depth"]
        x, y = 0, d - 1.4
        self.rug(0, d - 1.8, 3.1, 3.1)
        self.placement("Queen bed", x, y, 1.9, 2.3)
        for dx in (-0.78, 0.78):
            for dy in (-0.9, 0.9):
                self.box("Bed foot", (dx, y + dy, 0.14), (0.09, 0.09, 0.28), m["wood"], 0.01)
        self.box("Timber bed frame", (x, y, 0.30), (1.82, 2.15, 0.22), m["wood"], 0.03)
        self.box(
            "Upholstered headboard", (0, d - 0.22, 0.89), (2.04, 0.14, 1.25), m["fabric"], 0.06
        )
        self.box("Mattress", (x, y, 0.52), (1.8, 2.12, 0.28), m["linen"], 0.10)
        self.cloth("Draped linen duvet", x, y - 0.24, 2.12, 1.96, 0.69, m["linen"], 1.4)
        for dx in (-0.44, 0.44):
            pillow = self.box(
                "Sleeping pillow", (dx, y + 0.65, 0.75), (0.73, 0.43, 0.2), m["linen"], 0.09
            )
            pillow.rotation_euler[2] = dx * 0.08
        self.cloth("Folded textured bed throw", 0, y - 0.63, 1.98, 0.53, 0.72, m["fabric"], 1.1)
        for dx in (-1.22, 1.22):
            self.table(dx, d - 0.67, 0.55, 0.52, 0.55, "Bedside table")
            self.box("Bedside drawer", (dx, d - 0.67, 0.43), (0.51, 0.48, 0.18), m["wood"], 0.012)
            self.sphere("Drawer knob", (dx, d - 0.92, 0.43), (0.017, 0.017, 0.017), m["dark"])
            self.lamp(dx, d - 0.67, 0.55)
        self.table(0, d - 3.0, 1.35, 0.43, 0.42, "End-of-bed bench")
        self.box("Bench cushion", (0, d - 3.0, 0.46), (1.31, 0.42, 0.08), m["fabric"], 0.035)
        if f["arrangement"]:
            self.transform_group(
                list(self.furniture.objects),
                (0, y, 0),
                (-f["width"] / 2 + 1.4, d - 1.75, 0),
                math.pi / 2,
            )
        self.plant(f["width"] / 2 - 0.6, d - 0.6)
        return "Furnish this bedroom with a timber queen bed, upholstered headboard, linen duvet and pillows, a folded colored throw, two bedside tables with unlit lamps, a cushioned bench, a wool rug and a potted plant. Preserve the room architecture, doors, windows, camera and lighting."

    def dining(self):
        f, m = self.f, self.materials
        d, y = f["depth"], f["depth"] * 0.62
        self.rug(0, y, 3.25, 2.7)
        self.table(0, y, 2.15, 0.95, 0.76, "Dining table")
        for x in (-0.77, 0, 0.77):
            self.chair(x, y + 0.91)
            self.chair(x, y - 0.91, math.pi)
        self.vase(0, y, 0.76)
        if f["arrangement"]:
            self.transform_group(list(self.furniture.objects), (0, y, 0), (0, y, 0), math.pi / 2)
        self.placement("Sideboard", 0, d - 0.43, 2.1, 0.5)
        self.box("Sideboard cabinet", (0, d - 0.43, 0.53), (2.1, 0.5, 0.76), m["wood"], 0.018)
        for x in (-0.9, 0.9):
            for dy in (-0.18, 0.18):
                self.box(
                    "Sideboard foot",
                    (x, d - 0.43 + dy, 0.075),
                    (0.055, 0.055, 0.15),
                    m["dark"],
                    0.006,
                )
        for x in (-0.7, 0, 0.7):
            self.box("Sideboard door", (x, d - 0.69, 0.53), (0.685, 0.025, 0.72), m["wood"], 0.008)
        self.vase(-0.65, d - 0.43, 0.91)
        self.plant(f["width"] / 2 - 0.6, d - 0.64)
        return "Furnish this dining room with a timber dining table, six upholstered wooden chairs, a ceramic vase, a low timber sideboard, a wool rug and a potted plant. Preserve the room architecture, doors, windows, camera and lighting."

    def build(self):
        self.make_materials()
        self.architecture()
        self.lighting_camera()
        self.current_furniture = True
        instruction = getattr(self, self.f["kind"])()
        self.soften_upholstery()
        self.validate_bounds()
        self.fit_camera()
        return instruction

    def validate_bounds(self):
        from mathutils import Vector

        self.bpy.context.view_layer.update()
        for obj in self.furniture.objects:
            if obj.type != "MESH":
                continue
            points = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
            if any(
                abs(p.x) > self.f["width"] / 2 - 0.005
                or p.y < 0.01
                or p.y > self.f["depth"] - 0.005
                or p.z < -0.002
                or p.z > self.f["height"] - 0.05
                for p in points
            ):
                raise ValueError(f"furniture intersects architectural envelope: {obj.name}")
        for i, a in enumerate(self.placements):
            for b in self.placements[i + 1 :]:
                overlap = [
                    (a["footprint"][axis] + b["footprint"][axis]) / 2
                    - abs(a["center"][axis] - b["center"][axis])
                    for axis in (0, 1)
                ]
                if min(overlap) > 0.04:
                    raise ValueError(
                        f"primary furniture footprints intersect: {a['name']} / {b['name']}"
                    )


def node_state(tree):
    nodes = []
    for node in sorted(tree.nodes, key=lambda n: n.name):
        inputs = {}
        for socket in node.inputs:
            if hasattr(socket, "default_value"):
                value = socket.default_value
                if not isinstance(value, (str, int, float, bool, type(None))):
                    value = list(value) if hasattr(value, "__len__") else str(value)
                inputs[socket.identifier] = value
        row = {"name": node.name, "type": node.type, "inputs": inputs}
        for attribute in (
            "operation",
            "sky_type",
            "sun_elevation",
            "sun_rotation",
            "sun_disc",
            "noise_dimensions",
        ):
            if hasattr(node, attribute):
                row[attribute] = getattr(node, attribute)
        if hasattr(node, "color_ramp"):
            row["ramp"] = [(e.position, list(e.color)) for e in node.color_ramp.elements]
        nodes.append(row)
    return {
        "nodes": nodes,
        "links": sorted(
            (
                link.from_node.name,
                link.from_socket.identifier,
                link.to_node.name,
                link.to_socket.identifier,
            )
            for link in tree.links
        ),
    }


def invariant_state(bpy, furniture):
    """Canonical state of camera, architectural mesh, materials and lighting, excluding furniture."""
    scene = bpy.context.scene
    objects = []
    excluded = set(furniture.all_objects)
    for obj in sorted(scene.objects, key=lambda o: o.name):
        if obj in excluded:
            continue
        row = {
            "name": obj.name,
            "type": obj.type,
            "matrix": [list(r) for r in obj.matrix_world],
            "visibility": {
                k: getattr(obj, k)
                for k in ("hide_render", "visible_camera", "visible_glossy", "visible_transmission")
            },
        }
        if obj.type == "MESH":
            row["vertices"] = [list(v.co) for v in obj.data.vertices]
            row["faces"] = [list(p.vertices) for p in obj.data.polygons]
            row["materials"] = {
                m.name: node_state(m.node_tree) for m in obj.data.materials if m and m.use_nodes
            }
            row["modifiers"] = [
                {
                    k: getattr(mod, k)
                    for k in ("type", "width", "segments", "levels")
                    if hasattr(mod, k)
                }
                for mod in obj.modifiers
            ]
        if obj.type == "LIGHT":
            row["light"] = {
                "type": obj.data.type,
                "energy": obj.data.energy,
                "color": list(obj.data.color),
                "size": obj.data.size,
            }
        objects.append(row)
    return {
        "objects": objects,
        "camera_lens": scene.camera.data.lens,
        "camera_sensor": scene.camera.data.sensor_width,
        "world": node_state(scene.world.node_tree),
        "render_seed": scene.cycles.seed,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "view_transform": scene.view_settings.view_transform,
        "exposure": scene.view_settings.exposure,
    }


def render_pair(bpy, config, family, variant, output, save_scene):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = config["device"]
    preferences.get_devices()
    gpu_devices = []
    for device in preferences.devices:
        device.use = device.type == config["device"]
        if device.use:
            gpu_devices.append(device.name)
    if not gpu_devices:
        raise RuntimeError("requested GPU backend unavailable; no silent CPU fallback")
    scene.cycles.device = "GPU"
    scene.cycles.samples = config["samples"]
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = config["adaptive_threshold"]
    scene.cycles.use_denoising = True
    scene.cycles.denoiser = config["denoiser"]
    scene.cycles.seed = variant["seed"]
    scene.cycles.use_animated_seed = False
    scene.cycles.max_bounces = 10
    scene.cycles.diffuse_bounces = 6
    scene.cycles.transmission_bounces = 8
    scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    builder = Builder(bpy, family, variant, config)
    instruction = builder.build()
    bpy.context.view_layer.update()
    output.mkdir(parents=True)
    before = invariant_state(bpy, builder.furniture)
    dump(output / "invariant_state.json", before)
    invariant = hashlib.sha256(json.dumps(before, sort_keys=True).encode()).hexdigest()
    times, hashes = {}, {}
    for label, hidden in (("source", True), ("target", False)):
        builder.furniture.hide_render = hidden
        bpy.context.view_layer.update()
        state = invariant_state(bpy, builder.furniture)
        current = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
        if current != invariant:
            raise RuntimeError("camera/architecture/light state changed between pair renders")
        scene.render.filepath = str((output / f"{label}.png").resolve())
        start = time.perf_counter()
        bpy.ops.render.render(write_still=True)
        times[label] = time.perf_counter() - start
        hashes[label] = digest(output / f"{label}.png")
    if save_scene:
        bpy.ops.file.pack_all()
        bpy.ops.wm.save_as_mainfile(filepath=str((output / "scene.blend").resolve()), compress=True)
    record = {
        "status": "rendered_pending_visual_review",
        "family": family,
        "variant": variant,
        "instruction": instruction,
        "settings": config,
        "blender_version": bpy.app.version_string,
        "blender_build": bpy.app.build_hash.decode(),
        "devices": gpu_devices,
        "generator_sha256": digest(__file__),
        "architecture_camera_lighting_sha256": invariant,
        "paired_invariant_verified": True,
        "camera": {
            "matrix_world": [list(r) for r in scene.camera.matrix_world],
            "lens": scene.camera.data.lens,
        },
        "placements": builder.placements,
        "geometry_checks": "architectural envelope, door clearance, primary footprint overlaps and furniture framing passed",
        "scene_sha256": digest(output / "scene.blend") if save_scene else None,
        "generation_seconds": times,
        "sha256": hashes,
        "original_assets_only": not builder.used_assets,
        "cc0_asset_ids": sorted(builder.used_assets),
        "asset_provenance_sha256": digest(ROOT / config["asset_root"] / "provenance.json")
        if builder.used_assets
        else None,
    }
    dump(output / "render.json", record)
    print(
        "ROOMCRAFT_PAIR "
        + json.dumps({"id": variant["id"], "seconds": times, "path": str(output)}),
        flush=True,
    )


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_v05.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--families", default="all", help="Comma-separated numeric family IDs or all"
    )
    parser.add_argument("--variants", default="all", help="Comma-separated variant IDs or all")
    parser.add_argument("--save-scenes", action="store_true")
    parser.add_argument(
        "--preflight", action="store_true", help="Build and check all scenes without rendering"
    )
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config.get("furnishing_mode") == "cc0_hybrid":
        asset_root = ROOT / config["asset_root"]
        lock = ROOT / "docs/sources/v05/asset_lock.json"
        if digest(asset_root / "provenance.json") != digest(lock):
            raise ValueError("asset provenance differs from reviewed lock")
        for asset in json.loads(lock.read_text(encoding="utf-8"))["assets"]:
            for item in asset["files"]:
                if digest(asset_root / item["path"]) != item["sha256"]:
                    raise ValueError(f"asset bytes changed: {item['path']}")
    if (
        bpy.app.version_string != config["blender_version"]
        or bpy.app.build_hash.decode() != config["blender_build"]
    ):
        raise ValueError("Blender version/build differs from reproducible config")
    families = (
        range(config["families"])
        if args.families == "all"
        else [int(i) for i in args.families.split(",")]
    )
    variants = (
        range(config["variants_per_family"])
        if args.variants == "all"
        else [int(i) for i in args.variants.split(",")]
    )
    for index in families:
        family = family_parameters(index, config["seed"])
        for number in variants:
            variant = variant_parameters(family, number, config["seed"])
            output = args.output / "pairs" / variant["id"]
            if args.preflight:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                scene = bpy.context.scene
                scene.render.resolution_x, scene.render.resolution_y = config["resolution"]
                builder = Builder(bpy, family, variant, config)
                builder.build()
                print(f"PREFLIGHT_PASS {variant['id']}", flush=True)
                continue
            if output.exists():
                previous = json.loads((output / "render.json").read_text(encoding="utf-8"))
                if (
                    previous["generator_sha256"] != digest(__file__)
                    or previous["settings"] != config
                ):
                    raise ValueError(
                        "existing render uses a different recipe; choose a new output directory"
                    )
                if any(
                    digest(output / f"{key}.png") != previous["sha256"][key]
                    for key in ("source", "target")
                ):
                    raise ValueError("existing images changed")
                print(f"SKIP verified completed {output}; no overwrites", flush=True)
                continue
            render_pair(bpy, config, family, variant, output, args.save_scenes)


if __name__ == "__main__":
    main()
