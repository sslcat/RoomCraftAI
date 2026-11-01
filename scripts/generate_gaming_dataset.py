"""Original gaming furniture extension of the preserved RoomCraft Blender renderer."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_room_dataset as rooms
from project_room_landmarks import project

ROOT = Path(__file__).resolve().parents[1]


def family_parameters(index, seed):
    f = rooms.family_parameters(index, seed)
    rng = random.Random(seed + index * 1009)
    kind = ("bedroom", "spare room", "home office", "compact apartment")[index % 4]
    widths = ((4.2, 4.8), (3.8, 4.5), (4.6, 5.4), (3.7, 4.2))[index % 4]
    depths = ((4.8, 5.6), (4.4, 5.0), (4.8, 5.8), (4.3, 4.8))[index % 4]
    f.update(
        family=f"gaming_{index:02d}",
        kind="gaming",
        room_use=kind,
        width=round(rng.uniform(*widths), 3),
        depth=round(rng.uniform(*depths), 3),
        desk_wall=("back", "left", "right")[index % 3],
        desk_shape="L-shaped" if index % 2 else "straight",
        door_center=1.25,
    )
    if f["desk_wall"] != "back":
        f["door_side"] = "right" if f["desk_wall"] == "left" else "left"
    f["window_width"] = min(f["window_width"], f["width"] * 0.45)
    f["window_offset"] *= 0.6
    return f


def variant_parameters(family, variant, seed):
    v = rooms.variant_parameters(family, variant, seed)
    v.update(
        id=f"{family['family']}_v{variant:02d}",
        style=("minimalist", "professional", "streamer", "RGB-heavy")[
            (family["index"] + variant) % 4
        ],
        monitors=1 + (family["index"] + variant) % 3,
        camera_y=0.65 + variant * 0.045,
        camera_height=1.6 + variant * 0.035,
        lens=25 + variant % 3,
    )
    if family["desk_wall"] == "back":
        v["camera_x"] = 0.55 + (variant - 2) * 0.1
    else:
        v["camera_y"] = 0.25 + variant * 0.04
        v["camera_x"] = (0.8 if family["desk_wall"] == "left" else -0.8) + (variant - 2) * 0.06
    if family["room_use"] == "compact apartment":
        v.update(camera_y=0.25 + variant * 0.04, lens=22)
    return v


class GamingBuilder(rooms.Builder):
    def lighting_camera(self):
        super().lighting_camera()
        if self.v["style"] == "RGB-heavy":
            self.bpy.context.scene.world.node_tree.nodes.get("Background").inputs[
                "Strength"
            ].default_value = 0.07
            for obj in self.bpy.context.scene.objects:
                if obj.type == "LIGHT":
                    obj.data.energy = 65 if "Exterior" in obj.name else 12

    def cable(self, name, points, material):
        curve = self.bpy.data.curves.new(name, "CURVE")
        curve.dimensions = "3D"
        curve.bevel_depth = 0.005
        curve.bevel_resolution = 3
        spline = curve.splines.new("BEZIER")
        spline.bezier_points.add(len(points) - 1)
        for p, co in zip(spline.bezier_points, points, strict=True):
            p.co = co
            p.handle_left_type = "AUTO"
            p.handle_right_type = "AUTO"
        obj = self.bpy.data.objects.new(name, curve)
        self.furniture.objects.link(obj)
        curve.materials.append(material)

    def architecture(self):
        super().architecture()
        if self.f["index"] % 4 == 3:
            sign = 1 if self.f["door_side"] == "left" else -1
            self.box(
                "Architectural service pier",
                (sign * (self.f["width"] / 2 - 0.13), 1.6, self.f["height"] / 2),
                (0.26, 0.42, self.f["height"]),
                self.materials["wall"],
                0.005,
            )

    def emission(self, name, color, power):
        mat = self.material(name, color, 0.3)
        shader = mat.node_tree.nodes.get("Principled BSDF")
        shader.inputs["Emission Color"].default_value = (*color, 1)
        shader.inputs["Emission Strength"].default_value = power
        return mat

    def gaming(self):
        f, v, m = self.f, self.v, self.materials
        y = f["depth"] - 0.66
        width = (1.6, 1.95, 2.3)[v["monitors"] - 1]
        style = v["style"]
        top = self.material(
            "Desk satin laminate",
            (0.12, 0.14, 0.16) if style in ("streamer", "RGB-heavy") else (0.48, 0.29, 0.14),
            0.36,
            "wood",
        )
        metal = self.material("Powder coated metal", (0.028, 0.035, 0.043), 0.38)
        rubber = self.material("Rubber and keycaps", (0.025, 0.028, 0.032), 0.65)
        fabric = self.material("Chair charcoal fabric", (0.055, 0.07, 0.085), 0.72, "fabric")
        accent = self.emission(
            "Original RGB accent",
            (0.08, 0.25, 1) if f["index"] % 2 else (0.65, 0.08, 0.8),
            3 if style == "RGB-heavy" else 0.4,
        )
        self.box("Desk top", (0, y, 0.75), (width, 0.80, 0.04), top, 0.015)
        for x in (-width / 2 + 0.1, width / 2 - 0.1):
            for dy in (-0.28, 0.28):
                self.box("Desk steel leg", (x, y + dy, 0.365), (0.045, 0.045, 0.73), metal, 0.006)
            self.box("Desk lower frame rail", (x, y, 0.08), (0.05, 0.62, 0.045), metal, 0.006)
        self.box("Desk rear brace", (0, y + 0.29, 0.56), (width - 0.15, 0.035, 0.06), metal, 0.004)
        if f["desk_shape"] == "L-shaped":
            x = -width / 2 + 0.3
            self.box("Desk return top", (x, y - 0.85, 0.75), (0.60, 0.90, 0.04), top, 0.014)
            for dx in (-0.22, 0.22):
                self.box("Return leg", (x + dx, y - 1.2, 0.365), (0.045, 0.045, 0.73), metal, 0.005)
            self.box("Return drawer cabinet", (x, y - 0.85, 0.41), (0.48, 0.64, 0.64), top, 0.012)
            for z in (0.23, 0.43, 0.63):
                self.box("Drawer face", (x, y - 1.176, z), (0.45, 0.018, 0.185), top, 0.005)
                self.box(
                    "Drawer pull", (x, y - 1.192, z + 0.04), (0.13, 0.015, 0.013), metal, 0.003
                )
        self.box(
            "Under-desk cable tray", (0, y + 0.23, 0.665), (width * 0.68, 0.15, 0.06), metal, 0.006
        )
        screen = self.emission("Original abstract monitor wallpaper", (0.025, 0.12, 0.21), 0.8)
        screen_accent = self.emission("Original wallpaper bars", (0.06, 0.34, 0.48), 0.7)
        monitor_centers = [(i - (v["monitors"] - 1) / 2) * 0.625 for i in range(v["monitors"])]
        for x in monitor_centers:
            self.box("Monitor bezel", (x, y + 0.16, 1.15), (0.61, 0.045, 0.355), metal, 0.012)
            self.box("Monitor screen", (x, y + 0.135, 1.15), (0.58, 0.003, 0.325), screen, 0.003)
            for band in range(3):
                self.box(
                    "Original wallpaper geometry",
                    (x - 0.17 + band * 0.14, y + 0.132, 1.11 + band * 0.035),
                    (0.08, 0.002, 0.10 + band * 0.035),
                    screen_accent,
                    0.004,
                )
            self.box("Monitor clamp", (x, y + 0.32, 0.77), (0.075, 0.08, 0.055), metal, 0.004)
            self.rod("Monitor arm upright", (x, y + 0.32, 0.78), (x, y + 0.32, 1.12), 0.018, metal)
            self.rod("Monitor arm joint", (x, y + 0.32, 1.12), (x, y + 0.19, 1.15), 0.022, metal)
            self.rod("Monitor cable", (x, y + 0.20, 1.12), (x, y + 0.30, 0.68), 0.006, rubber)
        self.box("Desk mat", (0, y - 0.16, 0.775), (0.90, 0.33, 0.008), rubber, 0.01)
        self.box("Keyboard chassis", (-0.12, y - 0.16, 0.79), (0.44, 0.15, 0.023), metal, 0.009)
        for row in range(5):
            for col in range(14):
                self.box(
                    "Unbranded keyboard key",
                    (-0.32 + col * 0.03, y - 0.215 + row * 0.026, 0.808),
                    (0.025, 0.021, 0.012),
                    rubber,
                    0.003,
                )
        self.sphere("Mouse", (0.31, y - 0.16, 0.805), (0.033, 0.055, 0.022), rubber)
        self.rod("Mouse cable", (0.31, y - 0.105, 0.785), (0.31, y + 0.25, 0.785), 0.0025, rubber)
        self.box(
            "Keyboard wrist support", (-0.12, y - 0.278, 0.786), (0.44, 0.035, 0.02), fabric, 0.009
        )
        # Tower has floor-contacting feet; front fans are attached to its enclosure.
        px = width / 2 + 0.25
        self.box("PC tower", (px, y, 0.265), (0.25, 0.44, 0.48), metal, 0.012)
        for dx in (-0.085, 0.085):
            for dy in (-0.15, 0.15):
                self.cylinder("PC foot", (px + dx, y + dy, 0.013), 0.025, 0.026, rubber)
        for z in (0.12, 0.265, 0.41):
            fan = self.cylinder("PC illuminated fan ring", (px, y - 0.226, z), 0.059, 0.009, accent)
            fan.rotation_euler[0] = math.pi / 2
            hub = self.cylinder("PC fan hub", (px, y - 0.234, z), 0.022, 0.008, metal)
            hub.rotation_euler[0] = math.pi / 2
            for blade in range(5):
                a = blade * math.tau / 5
                self.rod(
                    "PC fan blade",
                    (px + 0.023 * math.cos(a), y - 0.239, z + 0.023 * math.sin(a)),
                    (px + 0.052 * math.cos(a + 0.3), y - 0.239, z + 0.052 * math.sin(a + 0.3)),
                    0.006,
                    metal,
                )
        self.box(
            "Tower side panel",
            (px + 0.127, y, 0.27),
            (0.005, 0.35, 0.38),
            self.material("Smoked side panel", (0.07, 0.09, 0.12), 0.2),
            0.004,
        )
        self.cable(
            "PC cable routed to tray",
            [
                (px, y + 0.22, 0.15),
                (px - 0.04, y + 0.27, 0.24),
                (px - 0.15, y + 0.26, 0.56),
                (width * 0.3, y + 0.24, 0.64),
            ],
            rubber,
        )
        # Chair is pulled clear of the desk; swivel base and casters contact the floor.
        cx, cy = (
            (-0.5 if f["desk_shape"] == "straight" and f["desk_wall"] == "back" else 0.24),
            y - 1.32,
        )
        self.cylinder("Chair gas lift", (cx, cy, 0.265), 0.035, 0.38, metal)
        for i in range(5):
            a = i * math.tau / 5
            tx, ty = cx + 0.32 * math.cos(a), cy + 0.32 * math.sin(a)
            self.rod("Chair base spoke", (cx, cy, 0.12), (tx, ty, 0.065), 0.026, metal)
            self.sphere("Chair caster", (tx, ty, 0.035), (0.04, 0.04, 0.035), rubber)
        self.box("Chair seat", (cx, cy, 0.48), (0.52, 0.48, 0.10), fabric, 0.045)
        self.box("Chair upholstered back", (cx, cy - 0.23, 0.86), (0.47, 0.11, 0.72), fabric, 0.045)
        self.box("Chair headrest", (cx, cy - 0.245, 1.20), (0.29, 0.12, 0.16), fabric, 0.045)
        for dx in (-0.31, 0.31):
            self.rod("Chair arm support", (cx + dx, cy, 0.48), (cx + dx, cy, 0.68), 0.018, metal)
            self.box("Chair armrest", (cx + dx, cy, 0.70), (0.07, 0.31, 0.045), rubber, 0.017)
        # Desk accessories and shelf have explicit supporting surfaces.
        self.cylinder("Pen cup", (-width / 2 + 0.16, y + 0.18, 0.83), 0.04, 0.12, m["ceramic"])
        for i in range(4):
            self.rod(
                "Pen",
                (-width / 2 + 0.14 + i * 0.012, y + 0.18, 0.79),
                (-width / 2 + 0.14 + i * 0.012, y + 0.18, 0.96),
                0.003,
                metal,
            )
        if style in ("professional", "streamer"):
            sh = f["height"] - 0.22
            self.box("Wall shelf", (0, f["depth"] - 0.22, sh), (0.95, 0.25, 0.035), top, 0.008)
            for x in (-0.32, 0.32):
                self.rod(
                    "Shelf bracket",
                    (x, f["depth"] - 0.10, sh - 0.13),
                    (x, f["depth"] - 0.31, sh - 0.015),
                    0.012,
                    metal,
                )
            for i in range(5):
                self.box(
                    "Original blank book",
                    (-0.3 + i * 0.055, f["depth"] - 0.22, sh + 0.075),
                    (0.045, 0.16, 0.115),
                    self.material(f"Book cover {i}", (0.12 + i * 0.035, 0.18, 0.25), 0.6),
                    0.003,
                )
        if style in ("streamer", "RGB-heavy"):
            self.box(
                "Desk RGB light strip",
                (0, y + 0.35, 0.785),
                (width - 0.15, 0.015, 0.012),
                accent,
                0.003,
            )
            self.box(
                "Return ambient strip",
                (-width / 2 + 0.04, y - 0.05, 0.70),
                (0.012, 0.65, 0.012),
                accent,
                0.003,
            )
            light = self.bpy.data.lights.new("Setup RGB bounce", "AREA")
            light.energy = 12 if style == "RGB-heavy" else 2
            light.color = (0.12, 0.20, 1) if f["index"] % 2 else (0.6, 0.06, 1)
            light.shape = "RECTANGLE"
            light.size = width * 0.8
            light.size_y = 0.05
            obj = self.bpy.data.objects.new("Setup RGB bounce", light)
            self.furniture.objects.link(obj)
            obj.location = (0, y + 0.37, 0.79)
            obj.rotation_euler[0] = math.pi
            if style == "RGB-heavy":
                self.box(
                    "Under-desk RGB strip",
                    (0, y - 0.36, 0.727),
                    (width - 0.18, 0.012, 0.006),
                    accent,
                    0.002,
                )
                light = self.bpy.data.lights.new("Under-desk RGB bounce", "AREA")
                light.energy = 8
                light.color = (0.12, 0.20, 1) if f["index"] % 2 else (0.6, 0.06, 1)
                light.shape = "RECTANGLE"
                light.size = width * 0.8
                light.size_y = 0.05
                obj = self.bpy.data.objects.new("Under-desk RGB bounce", light)
                self.furniture.objects.link(obj)
                obj.location = (0, y - 0.33, 0.70)
        if style == "streamer":
            self.cylinder("Microphone weighted base", (-0.58, y - 0.20, 0.79), 0.065, 0.025, metal)
            self.rod(
                "Microphone stand", (-0.58, y - 0.20, 0.80), (-0.58, y - 0.20, 1.03), 0.012, metal
            )
            self.sphere(
                "Microphone windscreen", (-0.58, y - 0.20, 1.08), (0.038, 0.038, 0.07), rubber
            )
        self.placement("Desk", 0, y, width, 0.8)
        self.placement("Chair", cx, cy, 0.75, 0.75)
        self.placement("PC tower", px, y, 0.26, 0.46)
        if f["desk_shape"] == "L-shaped":
            self.placement("Desk return", -width / 2 + 0.3, y - 0.85, 0.6, 0.9)
        # Rigidly rotate the complete setup for side-wall arrangements.
        angle = (
            0
            if f["desk_wall"] == "back"
            else math.pi / 2
            if f["desk_wall"] == "left"
            else -math.pi / 2
        )
        if angle:
            destination = (
                (-f["width"] / 2 + 0.66) if angle > 0 else (f["width"] / 2 - 0.66),
                f["depth"] * 0.61,
                0,
            )
            self.transform_group(list(self.furniture.objects), (0, y, 0), destination, angle)
        return (
            f"Furnish this {f['room_use']} with a {style} gaming setup: a {f['desk_shape']} desk against the {f['desk_wall']} wall, {v['monitors']} monitor{'s' if v['monitors'] != 1 else ''} on supported monitor arms, a desktop PC tower, a swivel gaming chair pulled back from the desk, keyboard, mouse, desk mat, pen cup and concealed cable management."
            + (
                " Include a high wall shelf with books."
                if style in ("professional", "streamer")
                else ""
            )
            + (" Include a desk microphone." if style == "streamer" else "")
            + (
                " Add blue or violet RGB accent lighting."
                if style in ("streamer", "RGB-heavy")
                else " Keep lighting restrained."
            )
            + " Preserve the room walls, windows, doors and camera viewpoint."
        )

    def build(self):
        self.make_materials()
        self.architecture()
        self.lighting_camera()
        self.current_furniture = True
        instruction = self.gaming()
        self.validate_bounds()
        self.fit_camera()
        # Door clearance checked again after any rigid setup rotation.
        for p in self.placements:
            x, y = p["center"]
            w, d = p["footprint"]
            near = (
                x - w / 2 < -self.f["width"] / 2 + 0.95
                if self.f["door_side"] == "left"
                else x + w / 2 > self.f["width"] / 2 - 0.95
            )
            if near and abs(y - self.f["door_center"]) < d / 2 + 0.65:
                raise ValueError("setup blocks door clearance")
        return instruction


def main():
    import bpy

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/dataset_v10_gaming.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--families", default="all")
    parser.add_argument("--variants", default="all")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    cfg = json.loads(args.config.read_text())
    rooms.Builder = GamingBuilder
    if (
        bpy.app.version_string != cfg["blender_version"]
        or bpy.app.build_hash.decode() != cfg["blender_build"]
    ):
        raise ValueError("Blender build mismatch")
    families = (
        range(cfg["families"])
        if args.families == "all"
        else [int(x) for x in args.families.split(",")]
    )
    variants = (
        range(cfg["variants_per_family"])
        if args.variants == "all"
        else [int(x) for x in args.variants.split(",")]
    )
    for i in families:
        f = family_parameters(i, cfg["seed"])
        for j in variants:
            v = variant_parameters(f, j, cfg["seed"])
            folder = args.output / "pairs" / v["id"]
            if args.preflight:
                bpy.ops.wm.read_factory_settings(use_empty=True)
                bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y = cfg[
                    "resolution"
                ]
                GamingBuilder(bpy, f, v, cfg).build()
                print("GAMING_PREFLIGHT " + v["id"], flush=True)
                continue
            if folder.exists():
                r = json.loads((folder / "render.json").read_text())
                if (
                    r.get("gaming_generator_sha256") != rooms.digest(__file__)
                    or r["settings"] != cfg
                    or any(
                        rooms.digest(folder / (k + ".png")) != r["sha256"][k]
                        for k in ("source", "target")
                    )
                ):
                    raise ValueError("existing render mismatch; use new output")
                continue
            rooms.render_pair(bpy, cfg, f, v, folder, True)
            r = json.loads((folder / "render.json").read_text())
            r.update(
                gaming_generator_sha256=rooms.digest(__file__),
                landmark_generator_sha256=rooms.digest(ROOT / "scripts/project_room_landmarks.py"),
                lighting_note="Room lighting/camera/architecture fixed; intrinsic screen/RGB furniture emission appears only in target.",
            )
            landmarks = project(bpy, f)
            landmarks.update(source_sha256=r["sha256"]["source"], scene_id=v["id"])
            rooms.dump(folder / "landmarks.json", landmarks)
            r["landmarks_sha256"] = rooms.digest(folder / "landmarks.json")
            rooms.dump(folder / "render.json", r)


if __name__ == "__main__":
    main()
