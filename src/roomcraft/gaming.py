"""Deterministic, project-owned gaming instructions; no model or GPU imports."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

THEMES: dict[str, dict[str, Any]] = json.loads(Path(__file__).with_name("themes.json").read_text())
SETUPS = {
    "compact": ("Compact gaming corner", "compact", "single"),
    "single": ("Standard single-monitor setup", "straight", "single"),
    "dual": ("Dual-monitor setup", "straight", "dual_side"),
    "triple": ("Triple-monitor setup", "straight", "triple_arc"),
    "creator": ("Streaming / creator setup", "straight", "dual_side"),
    "ultrawide": ("Premium ultrawide setup", "l_shaped", "ultrawide"),
}
DESKS = {"straight": "straight desk", "l_shaped": "L-shaped desk", "compact": "compact desk"}
TOWERS = {
    "on_desk": "one realistically proportioned desktop PC tower on the desk, leaving keyboard and monitor space",
    "under_desk": "one ventilated desktop PC tower under the desk, clear of knees and chair movement",
    "stand": "one desktop PC tower on a low separate PC stand beside the desk, clear of walking paths",
}
MONITORS = {
    "single": (1, "one monitor"),
    "dual_side": (2, "two monitors side by side"),
    "dual_stacked": (2, "two monitors in a supported vertical stack"),
    "triple_arc": (3, "three monitors in a shallow arc"),
    "ultrawide": (1, "one premium ultrawide monitor"),
}
LIGHTING = {
    "none": "No RGB or colored lighting. Keep existing daylight and use neutral practical illumination only; ignore colored theme-light suggestions.",
    "ambient": "Use subtle warm or neutral ambient electric lighting, no saturated RGB glow; existing daylight remains dominant.",
    "accent": "Use restrained RGB accents behind screens and in the PC, with realistic falloff; keep room details readable.",
    "immersive": "Use immersive themed electric accent lighting with controlled brightness, realistic shadows and natural daylight integration.",
}
PRESERVE = (
    "Preserve the original walls, windows, doors, ceiling, flooring, camera viewpoint and room "
    "perspective. Do not add, remove or move architectural openings. Keep existing materials "
    "and light sources. Themes decorate the room through furniture and removable accessories, "
    "not structural renovation. Keep door swings and walking paths unobstructed."
)


@dataclass(frozen=True)
class GamingConfig:
    setup: str = "dual"
    theme: str = "minimalist_luxury"
    desk: str = "straight"
    tower: str = "stand"
    monitors: str = "dual_side"
    chair: bool = True
    peripherals: bool = True
    headset: bool = True
    speakers: bool = False
    cables: bool = True
    shelving: bool = False
    decorations: bool = True
    lighting: str = "ambient"
    custom_theme: str = ""

    purpose: str = "Gaming"
    workstation: str = "Standard desk"
    work_screens: str = "Single monitor"
    seating: str = "Reading armchair"
    reading: bool = True
    mixed_functions: str = "Work + relaxation"

    def validate(self) -> None:
        for name, options in (
            ("purpose", ("Gaming", "Work", "Relaxation", "Mixed-purpose")),
            ("workstation", ("Standard desk", "Sit-stand desk", "Compact laptop desk")),
            ("work_screens", ("Laptop only", "Single monitor", "Dual monitors")),
            ("seating", ("Reading armchair", "Compact sofa", "Chaise lounge")),
            ("mixed_functions", ("Work + relaxation", "Gaming + work", "Gaming + relaxation")),
            ("setup", SETUPS),
            ("theme", {**THEMES, "custom": {}}),
            ("desk", DESKS),
            ("tower", TOWERS),
            ("monitors", MONITORS),
            ("lighting", LIGHTING),
        ):
            if getattr(self, name) not in options:
                raise ValueError(f"Choose a valid {name.replace('_', ' ')} option.")
        for name in (
            "reading",
            "chair",
            "peripherals",
            "headset",
            "speakers",
            "cables",
            "shelving",
            "decorations",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be enabled or disabled.")
        if not isinstance(self.custom_theme, str) or len(self.custom_theme) > 500:
            raise ValueError("Keep the custom theme under 500 characters.")
        if self.theme == "custom" and not self.custom_theme.strip():
            raise ValueError("Describe your custom theme, or select one of the six themes.")


def compose_gaming(config: GamingConfig, extra: str = "") -> tuple[str, dict[str, Any]]:
    """Compile every setting into the actual Qwen instruction and a reproducible receipt."""
    config.validate()
    if not isinstance(extra, str) or len(extra) > 2000:
        raise ValueError("Keep custom instructions under 2,000 characters.")
    theme = THEMES.get(config.theme)
    parts = [
        "Create a photorealistic gaming setup inside this same photographed room.",
        PRESERVE,
        f"Setup preset: {SETUPS[config.setup][0]}; the equipment selections below take precedence over preset defaults.",
        f"Use a human-scale {DESKS[config.desk]} with all feet resting on the floor, sized conservatively for the visible space.",
        f"Include {MONITORS[config.monitors][1]} at comfortable viewing height, physically supported by stands or desk-clamped arms; no floating screens.",
        "Include " + TOWERS[config.tower] + ".",
        "Place an ergonomic gaming chair with plausible pull-out clearance."
        if config.chair
        else "Do not introduce a chair.",
        "Place a keyboard, mouse and mousepad naturally on the usable desktop."
        if config.peripherals
        else "Do not add keyboard, mouse or mousepad.",
        "Include a headset resting on a supported headset stand."
        if config.headset
        else "Omit headsets and headset stands.",
        "Include two compact desktop speakers without blocking screens."
        if config.speakers
        else "Omit speakers.",
        "Route visible cables through practical trays or clips; keep walking paths clear."
        if config.cables
        else "Keep realistic safe cable routing without decorative cable-management hardware.",
        "Add a small freestanding shelf only where circulation permits."
        if config.shelving
        else "Do not add shelving, even if the theme suggests it.",
    ]
    if config.purpose != "Gaming":
        parts = [
            f"Create a photorealistic {config.purpose.lower()} room inside this same photographed room.",
            PRESERVE,
        ]
        functions = (
            config.mixed_functions.lower()
            if config.purpose == "Mixed-purpose"
            else config.purpose.lower()
        )
        if "work" in functions:
            parts += [
                f"Include one {config.workstation.lower()} with {config.work_screens.lower()}, supported equipment, keyboard and mouse, one ergonomic task chair and tidy cables."
            ]
        if "gaming" in functions:
            # Reuse the established equipment composer without its independent theme branch.
            parts += parts_gaming_equipment(config)
        if "relaxation" in functions:
            parts += [
                f"Include one {config.seating.lower()} at realistic scale with a small side table.",
                "Add a supported electric reading lamp and a few books."
                if config.reading
                else "Omit reading lamps and books.",
            ]
        if config.purpose == "Mixed-purpose":
            parts += [
                f"Combine only {config.mixed_functions.lower()}; share the desk and chair between work and gaming where applicable. Prioritize circulation, omit excess furniture, do not overcrowd."
            ]
        parts += [
            "Add a small freestanding shelf only where circulation permits."
            if config.shelving
            else "Do not add shelving, even if the theme suggests it."
        ]
    if config.setup == "creator" and config.purpose == "Gaming":
        parts.append(
            "Add a desk-clamped microphone boom, small mounted camera and modest electric key light."
        )
    if theme:
        if config.shelving:
            parts.append("Shelving styling: " + theme["shelving"] + ".")
        parts.extend(
            [
                f"Theme: {theme['name']}. Materials: {theme['materials']}.",
                f"Furniture and accessory palette: {theme['palette_names']}.",
                f"Equipment styling: {theme['equipment']}.",
            ]
        )
        if (
            config.theme == "cat_kingdom"
            and config.headset
            and (
                config.purpose == "Gaming"
                or (config.purpose == "Mixed-purpose" and "Gaming" in config.mixed_functions)
            )
        ):
            parts.append("Use tasteful cat-ear details on the headset.")
        if (
            config.theme == "cat_kingdom"
            and config.peripherals
            and (
                config.purpose == "Gaming"
                or (config.purpose == "Mixed-purpose" and "Gaming" in config.mixed_functions)
            )
        ):
            parts.append("Use subtle paw-shaped details on the desk mat.")
        if config.decorations:
            parts.append("Removable decorative accents: " + theme["decorations"] + ".")
        parts.extend(
            [
                "Theme light inspiration, subordinate to the lighting choice: "
                + theme["lighting"]
                + ".",
                theme["constraints"] + ". Avoid: " + theme["negative_constraints"] + ".",
            ]
        )
    else:
        parts.append(
            "Custom decorative theme: "
            + config.custom_theme.strip()
            + ". Apply only to furniture and removable accessories."
        )
    if not config.decorations:
        parts.append(
            "Omit additional themed ornaments and decorations; express the theme through furniture and equipment finishes only."
        )
    if extra.strip():
        parts.append(
            "Additional wishes, subject to room-preservation constraints: " + extra.strip()
        )
    parts.extend(
        [
            LIGHTING[config.lighting],
            "Keep the room dry and electrical installations plausible. Match camera perspective and furniture scale with credible contact shadows. Do not imply measured fit from this photograph. "
            + PRESERVE,
        ]
    )
    receipt = {
        "composer_version": "1.1.0",
        "configuration": asdict(config),
        "monitor_count": (
            MONITORS[config.monitors][0]
            if config.purpose == "Gaming"
            or (config.purpose == "Mixed-purpose" and "Gaming" in config.mixed_functions)
            else {"Laptop only": 0, "Single monitor": 1, "Dual monitors": 2}[config.work_screens]
            if config.purpose == "Work"
            or (config.purpose == "Mixed-purpose" and "Work" in config.mixed_functions)
            else 0
        ),
        "theme_definition": theme
        if theme
        else {"id": "custom", "name": "Custom theme", "description": config.custom_theme.strip()},
        "additional_instructions": extra.strip(),
        "notice": "Conceptual visualization, not a measured installation plan. Gaming-specific weights have not been trained.",
    }
    return " ".join(parts), receipt


def parts_gaming_equipment(config: GamingConfig) -> list[str]:
    """Reuse gaming equipment for a shared mixed-purpose zone."""
    from dataclasses import replace

    prompt, _ = compose_gaming(replace(config, purpose="Gaming", theme="minimalist_luxury"))
    start = prompt.index("Setup preset:")
    end = prompt.index("Theme: Minimalist Luxury")
    return [prompt[start:end]]
