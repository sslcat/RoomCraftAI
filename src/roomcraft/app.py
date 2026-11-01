"""Local Gradio product UI. Model imports stay in the separate worker process."""

from __future__ import annotations

import atexit
import html
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_RUN_HISTORY", "False")

from roomcraft.app_service import PRESETS, ROOT, AppConfig, AppError, AppService, clean_upload
from roomcraft.gaming import DESKS, MONITORS, SETUPS, THEMES, GamingConfig, compose_gaming


def build_app(service: AppService) -> Any:
    import gradio as gr

    def display(record: dict[str, Any], token: str) -> tuple[Any, ...]:
        meta = record["metadata"]
        w, h = meta["working_size"]
        model = "RoomCraft LoRA" if meta["adapter_sha256"] else "Unchanged Qwen"
        if record["origin"] == "saved_example":
            message = f"**Saved example · {model}** · {w} × {h} native pixels. Previously generated during V0.7; this click did not run inference. Photo: upyernoz, CC BY 2.0."
        elif record["origin"] == "selective_composite":
            message = f"**Selective preservation preview** · {w} × {h}. Local composite; inspect boundaries and furniture. Original result remains in history."
        else:
            geometry = "RoomCraft RGA on" if meta.get("geometry_adapter") else "RGA off"
            message = f"**Your room is ready** · {model} · {geometry} · {w} × {h} native pixels · seed {meta['settings']['seed']} · {meta['total_seconds']:.1f}s. Inspect architecture and furniture before using the result."
        return (
            (record["before"], record["path"]),
            gr.update(value=record["path"], interactive=True),
            gr.update(value=record["metadata_path"], interactive=True),
            meta,
            service.gallery(token),
            message,
        )

    def failure(error: Exception, token: str) -> tuple[Any, ...]:
        message = (
            str(error)
            if isinstance(error, AppError)
            else "The worker could not complete this request. Check its installation or connection and try again."
        )
        try:
            history = service.gallery(token)
        except AppError:
            history = []
        return (
            None,
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
            None,
            history,
            "**Generation did not complete.** " + html.escape(message),
        )

    progress_indicator = gr.Progress()

    def generate(
        token: str,
        upload: str | None,
        category: str,
        preset: str,
        extra: str,
        model: str,
        seed: float,
        size: str,
        steps: float,
        cfg: float,
        rga: bool,
        setup: str,
        theme: str,
        desk: str,
        tower: str,
        monitors: str,
        chair: bool,
        peripherals: bool,
        headset: bool,
        speakers: bool,
        cables: bool,
        shelving: bool,
        decorations: bool,
        lighting: str,
        custom_theme: str,
        purpose: str,
        workstation: str,
        work_screens: str,
        seating: str,
        reading: bool,
        mixed_functions: str,
        progress: Any = progress_indicator,
    ) -> tuple[Any, ...]:
        try:
            progress(0.05, desc="Checking your room and settings")
            progress(
                None,
                desc="Loading Qwen and generating. The first model load can take several minutes.",
            )
            record = service.generate(
                token,
                upload,
                category,
                preset,
                extra or "",
                model,
                int(seed),
                int(size),
                int(steps),
                float(cfg),
                bool(rga),
                GamingConfig(
                    setup,
                    theme,
                    desk,
                    tower,
                    monitors,
                    chair,
                    peripherals,
                    headset,
                    speakers,
                    cables,
                    shelving,
                    decorations,
                    lighting,
                    custom_theme or "",
                    purpose,
                    workstation,
                    work_screens,
                    seating,
                    reading,
                    mixed_functions,
                ),
            )
            progress(1, desc="Image verified")
            return display(record, token)
        except Exception as exc:
            return failure(exc, token)

    def example(token: str, model: str) -> tuple[Any, ...]:
        try:
            return display(service.example(token, model), token)
        except Exception as exc:
            return failure(exc, token)

    def choose(token: str, event: gr.SelectData) -> tuple[Any, ...]:
        try:
            return display(service.select(token, int(event.index)), token)
        except Exception as exc:
            return failure(exc, token)

    # Gradio resolves event annotations in module globals, not this lazy-import scope.
    choose.__annotations__["event"] = gr.SelectData

    def clear(token: str) -> tuple[Any, ...]:
        service.clear(token)
        return (
            None,
            gr.update(value=None, interactive=False),
            gr.update(value=None, interactive=False),
            None,
            [],
            "Session cleared. Upload a photograph to begin.",
            None,
            gr.update(value=None, visible=False),
        )

    def preview_input(token: str, upload: str | None) -> tuple[Any, str]:
        try:
            service.records(token)
            if not upload:
                return gr.update(value=None, visible=False), "Upload a room photograph to begin."
            preview = service.directory / token / "upload-preview.png"
            clean_upload(Path(upload), preview)
            return gr.update(
                value=str(preview), visible=True
            ), "Photograph checked. Choose a preset and visualize your room."
        except AppError as exc:
            return gr.update(value=None, visible=False), "**Invalid photograph.** " + html.escape(
                str(exc)
            )

    def presets(category: str) -> Any:
        values = PRESETS[category]
        return gr.update(choices=[(v["name"], v["id"]) for v in values], value=values[0]["id"])

    def theme_card(value: str) -> str:
        theme = THEMES.get(value)
        if theme is None:
            return '<div class="rc-theme-card"><strong>Your own world</strong><p>Describe a decorative theme below. The room architecture stays the starting point.</p></div>'
        swatches = "".join(f'<i style="background:{c}"></i>' for c in theme["palette"])
        return f'<div class="rc-theme-card"><span class="rc-swatches">{swatches}</span><strong>{html.escape(theme["name"])}</strong><p>{html.escape(theme["description"])}</p></div>'

    def prompt_preview(category: str, preset: str, extra: str, *values: Any) -> str:
        try:
            if category == "Gaming setups":
                normalized = list(values)
                normalized[13] = normalized[13] or ""
                return compose_gaming(GamingConfig(*normalized), extra or "")[0]
            from roomcraft.app_service import instruction

            return instruction(category, preset, extra or "")
        except (ValueError, AppError) as exc:
            return "Check your configuration: " + str(exc)

    with gr.Blocks(
        title="RoomCraft Studio",
        analytics_enabled=False,
        delete_cache=(300, 3600),
    ) as app:
        session = gr.State(value=lambda: service.session(), time_to_live=3600)
        gr.HTML(
            """<header class="rc-header"><div class="rc-brand"><svg width="38" height="38" viewBox="0 0 40 40" aria-hidden="true"><path d="M6 14 20 6l14 8v16l-14 7-14-7Z" fill="none" stroke="#6550aa" stroke-width="2.4"/><path d="M6 14l14 8 14-8M20 22v15M12 27v-7l16-1v9" fill="none" stroke="#6550aa" stroke-width="2"/></svg><span>roomcraft<span class="rc-ai"> AI</span></span></div><span class="rc-pill">ROOMCRAFT STUDIO · V1.1.0</span></header><section class="rc-hero"><p class="rc-eyebrow">YOUR ROOM. A WORLD OF POSSIBILITIES.</p><h1>Build your next<br><span>room, reimagined.</span></h1><p>A photograph of your room. Space for work, play and relaxation. A theme worth coming home to.</p><div class="rc-steps"><span>01 Upload your room</span><span>02 Choose your world</span><span>03 Visualize &amp; compare</span></div></section>"""
        )
        if service.config.remote_session:
            connection = "**Compute: approved Runpod worker** · Generation uses your explicitly started worker and its spending deadline. This page does not create paid resources."
        else:
            connection = "**Compute: local Qwen worker** · Requires an 80 GB class GPU. On a smaller GPU, explore a saved result or connect an approved worker using the launch guide."
        gr.Markdown(connection, elem_classes="rc-compute")
        with gr.Row(equal_height=False, elem_id="studio-workflow"):
            with gr.Column(scale=5, min_width=300, elem_classes="rc-controls"):
                gr.Markdown("### 1 · Your room")
                upload = gr.File(
                    label="Empty-room photograph",
                    file_types=[".jpg", ".jpeg", ".png", ".webp"],
                    type="filepath",
                    elem_id="room-upload",
                )
                upload_preview = gr.Image(
                    label="Your photograph",
                    type="filepath",
                    format="png",
                    height=190,
                    interactive=False,
                    visible=False,
                    buttons=[],
                )
                gr.Markdown("### 2 · Design your setup")
                category = gr.Dropdown(
                    [("RoomCraft Studio" if k == "Gaming setups" else k, k) for k in PRESETS],
                    value="Gaming setups",
                    label="Design category",
                )
                purpose = gr.Dropdown(
                    ["Gaming", "Work", "Relaxation", "Mixed-purpose"],
                    value="Gaming",
                    label="Room purpose",
                )
                with gr.Group(visible=False) as work_group:
                    workstation = gr.Dropdown(
                        ["Standard desk", "Sit-stand desk", "Compact laptop desk"],
                        value="Standard desk",
                        label="Workstation",
                    )
                    work_screens = gr.Dropdown(
                        ["Laptop only", "Single monitor", "Dual monitors"],
                        value="Single monitor",
                        label="Work displays",
                    )
                with gr.Group(visible=False) as relaxation_group:
                    seating = gr.Dropdown(
                        ["Reading armchair", "Compact sofa", "Chaise lounge"],
                        value="Reading armchair",
                        label="Relaxation seating",
                    )
                    reading = gr.Checkbox(True, label="Reading lamp & books")
                mixed_functions = gr.Dropdown(
                    ["Work + relaxation", "Gaming + work", "Gaming + relaxation"],
                    value="Work + relaxation",
                    label="Combined functions",
                    visible=False,
                )
                first = PRESETS["Gaming setups"]
                preset = gr.Dropdown(
                    [(v["name"], v["id"]) for v in first],
                    value=first[0]["id"],
                    label="Preset",
                    visible=False,
                )
                with gr.Group(visible=True, elem_id="gaming-controls") as gaming_group:
                    setup = gr.Dropdown(
                        [(v[0], k) for k, v in SETUPS.items()], value="dual", label="Gaming setup"
                    )
                    theme = gr.Dropdown(
                        [(v["name"], k) for k, v in THEMES.items()] + [("Custom theme", "custom")],
                        value="minimalist_luxury",
                        label="Creative theme",
                    )
                    theme_summary = gr.HTML(theme_card("minimalist_luxury"))
                    custom_theme = gr.Textbox(
                        value="",
                        label="Describe your theme",
                        placeholder="A retro space observatory with removable celestial decor…",
                        max_lines=3,
                        visible=False,
                    )
                    lighting = gr.Dropdown(
                        [
                            ("No RGB", "none"),
                            ("Subtle ambient lighting", "ambient"),
                            ("RGB accent lighting", "accent"),
                            ("Immersive themed lighting", "immersive"),
                        ],
                        value="ambient",
                        label="Lighting",
                    )
                    with gr.Accordion("Equipment & placement", open=False) as equipment_section:
                        gr.Markdown(
                            "Changing a setup restores its suggested desk and monitors. Fine-tune them here.",
                            elem_classes="rc-muted",
                        )
                        desk = gr.Dropdown(
                            [(v.capitalize(), k) for k, v in DESKS.items()],
                            value="straight",
                            label="Desk type",
                        )
                        tower = gr.Dropdown(
                            [
                                ("On desk", "on_desk"),
                                ("Under desk", "under_desk"),
                                ("Separate PC stand", "stand"),
                            ],
                            value="stand",
                            label="PC tower placement",
                        )
                        monitors = gr.Dropdown(
                            [(v[1].capitalize(), k) for k, v in MONITORS.items()],
                            value="dual_side",
                            label="Monitor count & arrangement",
                        )
                        with gr.Row():
                            chair = gr.Checkbox(True, label="Gaming chair")
                            peripherals = gr.Checkbox(True, label="Keyboard, mouse & mousepad")
                        with gr.Row():
                            headset = gr.Checkbox(True, label="Headset & stand")
                            speakers = gr.Checkbox(False, label="Desktop speakers")
                        with gr.Row():
                            cables = gr.Checkbox(True, label="Cable management")
                            shelving = gr.Checkbox(False, label="Freestanding shelving")
                        decorations = gr.Checkbox(True, label="Theme decorations")
                extra = gr.Textbox(
                    label="Anything else?",
                    placeholder="Keep the closet clear, choose a light oak finish…",
                    lines=2,
                    max_lines=4,
                )
                model = gr.Radio(
                    choices=[("Unchanged Qwen", "base"), ("Qwen + RoomCraft LoRA", "roomcraft")],
                    value="base",
                    label="Image model",
                )
                gr.Markdown(
                    "RoomCraft LoRA is a general staging research adapter. Gaming presets use instructions; a gaming-specific adapter has not been trained.",
                    elem_classes="rc-muted",
                )
                rga = gr.Checkbox(
                    value=False,
                    interactive=False,
                    label="Preserve Room Geometry — RoomCraft RGA",
                    info="On by default with RoomCraft LoRA; switch off for its baseline. One positive real-photo comparison; room geometry can still drift. Unchanged Qwen uses neither adapter.",
                    elem_id="roomcraft-rga",
                )
                with gr.Accordion("Advanced inference settings", open=False) as advanced_section:
                    size = gr.Radio(
                        ["512", "768", "1024"], value="1024", label="Native output long side"
                    )
                    seed = gr.Number(
                        value=42, precision=0, minimum=0, maximum=2**53 - 1, label="Seed"
                    )
                    steps = gr.Slider(1, 60, value=40, step=1, label="Inference steps")
                    cfg = gr.Slider(1, 10, value=4, step=0.1, label="Instruction guidance")
                generate_button = gr.Button(
                    "Visualize my room", variant="primary", size="lg", elem_id="generate-room"
                )
                gr.Markdown(
                    "Conceptual visualization, not a measured installation plan. Furniture fit and architecture require human review.",
                    elem_classes="rc-muted",
                )
                with gr.Accordion(
                    "Preview the exact instruction", open=False
                ) as instruction_section:
                    composed = gr.Textbox(
                        value=compose_gaming(GamingConfig())[0],
                        label="Instruction sent to Qwen",
                        lines=7,
                        interactive=False,
                        elem_id="composed-prompt",
                    )
            with gr.Column(scale=7, min_width=300, elem_classes="rc-result"):
                gr.Markdown("### 3 · See the possibilities")
                comparison = gr.ImageSlider(
                    label="Before / after · drag to compare",
                    type="filepath",
                    format="png",
                    height=490,
                    interactive=False,
                    elem_id="room-comparison",
                )
                status = gr.Markdown(
                    "Upload a room, choose a setup and theme, then visualize. Or explore a real saved staging example below; it is not a themed gaming demonstration.",
                    elem_id="generation-status",
                )
                with gr.Row():
                    download = gr.DownloadButton(
                        "Download PNG", interactive=False, variant="primary", elem_id="download-png"
                    )
                    settings_download = gr.DownloadButton("Download settings", interactive=False)
                example_button = gr.Button("Explore a saved real-room result", variant="secondary")
                gr.Markdown(
                    "Saved example: an actual V0.7 model output, not a new generation. Uses your selected base/LoRA model. [Photo and image credits](https://commons.wikimedia.org/wiki/File:Our_Hotel_Suite-1_(4983234010).jpg): upyernoz, CC BY 2.0; resized and AI edited.",
                    elem_classes="rc-muted",
                )
                with gr.Accordion("Image details", open=False):
                    details = gr.JSON(label="Model, seed, native dimensions and runtime")
        with gr.Accordion("Selective preservation · experimental", open=False):
            gr.Markdown(
                "Optional local compositing of the displayed result. Upload a mask matching your source photo: white restores original architecture, black keeps generated content. Keep furniture black. An optional foreground mask (white furniture) excludes objects from restoration. Inspect seams and lighting; this is not model geometry control."
            )
            protect_mask = gr.File(
                label="Architecture protection mask", file_types=[".png"], type="filepath"
            )
            foreground_mask = gr.File(
                label="Foreground exclusion mask (optional)", file_types=[".png"], type="filepath"
            )
            feather = gr.Slider(0, 24, value=8, step=1, label="Boundary feather (pixels)")
            preserve_button = gr.Button("Apply selective preservation")
        with gr.Row():
            gr.Markdown("### Recent in this session")
            clear_button = gr.Button("Clear session", size="sm", scale=0)
            reset_button = gr.Button("Reset studio", size="sm", scale=0)
        gallery = gr.Gallery(
            columns=4,
            rows=2,
            height=220,
            label="Recent images",
            show_label=False,
            interactive=False,
            allow_preview=False,
            format="png",
            elem_id="recent-images",
            object_fit="contain",
        )
        gr.Markdown(
            "Images can change architecture, lighting and materials. Review every result. This studio keeps up to eight results in the current session; download anything you want to keep. Photos and results are temporary local files. An approved Runpod worker receives the photo only when you generate.",
            elem_classes="rc-footer",
        )
        outputs = [comparison, download, settings_download, details, gallery, status]

        def preserve_result(
            token: str,
            meta: dict[str, Any] | None,
            mask: str | None,
            foreground: str | None,
            radius: float,
        ) -> tuple[Any, ...]:
            try:
                return display(service.preserve(token, meta, mask, foreground, int(radius)), token)
            except Exception as exc:
                return failure(exc, token)

        preserve_button.click(
            preserve_result,
            [session, details, protect_mask, foreground_mask, feather],
            outputs,
            api_name=False,
            concurrency_id="image-worker",
            concurrency_limit=1,
        )
        category.change(presets, [category], [preset], api_name=False, queue=False)
        category.change(
            lambda value: (
                gr.update(visible=value == "Gaming setups"),
                gr.update(visible=value != "Gaming setups"),
            ),
            [category],
            [gaming_group, preset],
            api_name=False,
            queue=False,
        )
        setup.change(
            lambda value: (
                SETUPS[value][0] + ". Adjust equipment below to personalize it.",
                gr.update(value=SETUPS[value][1]),
                gr.update(value=SETUPS[value][2]),
            ),
            [setup],
            [status, desk, monitors],
            api_name=False,
            queue=False,
        )
        theme.change(
            lambda value: (theme_card(value), gr.update(visible=value == "custom")),
            [theme],
            [theme_summary, custom_theme],
            api_name=False,
            queue=False,
        )
        gaming_inputs = [
            setup,
            theme,
            desk,
            tower,
            monitors,
            chair,
            peripherals,
            headset,
            speakers,
            cables,
            shelving,
            decorations,
            lighting,
            custom_theme,
        ]
        purpose_inputs = [purpose, workstation, work_screens, seating, reading, mixed_functions]
        gaming_inputs += purpose_inputs

        def purpose_visibility(mode: str, purpose_value: str, functions: str) -> tuple[Any, ...]:
            studio_mode = mode == "Gaming setups"
            active = (
                functions.lower() if purpose_value == "Mixed-purpose" else purpose_value.lower()
            )
            return (
                gr.update(visible=studio_mode),
                gr.update(visible=studio_mode and "work" in active),
                gr.update(visible=studio_mode and "relaxation" in active),
                gr.update(visible=studio_mode and purpose_value == "Mixed-purpose"),
                gr.update(visible=studio_mode and "gaming" in active),
                gr.update(visible=studio_mode and "gaming" in active),
            )

        gr.on(
            triggers=[category.change, purpose.change, mixed_functions.change],
            fn=purpose_visibility,
            inputs=[category, purpose, mixed_functions],
            outputs=[
                purpose,
                work_group,
                relaxation_group,
                mixed_functions,
                setup,
                equipment_section,
            ],
            queue=False,
            api_name=False,
        )
        preview_inputs = [category, preset, extra, *gaming_inputs]
        gr.on(
            triggers=[control.change for control in preview_inputs],
            fn=prompt_preview,
            inputs=preview_inputs,
            outputs=[composed],
            api_name=False,
            queue=False,
            trigger_mode="always_last",
        )
        upload.change(
            preview_input, [session, upload], [upload_preview, status], api_name=False, queue=False
        )
        model.change(
            lambda value: gr.update(value=value == "roomcraft", interactive=value == "roomcraft"),
            [model],
            [rga],
            api_name=False,
            queue=False,
        )
        generate_button.click(
            generate,
            [
                session,
                upload,
                category,
                preset,
                extra,
                model,
                seed,
                size,
                steps,
                cfg,
                rga,
                *gaming_inputs,
            ],
            outputs,
            api_name="generate",
            concurrency_limit=1,
            concurrency_id="image-worker",
        )
        example_button.click(
            example,
            [session, model],
            outputs,
            api_name="saved_example",
            concurrency_limit=1,
            concurrency_id="image-worker",
        )
        gallery.select(
            choose,
            [session],
            outputs,
            api_name=False,
            concurrency_limit=1,
            concurrency_id="image-worker",
        )
        clear_button.click(
            clear,
            [session],
            outputs + [upload, upload_preview],
            api_name="clear_session",
            concurrency_limit=1,
            concurrency_id="image-worker",
        )

        def reset(token: str) -> tuple[Any, ...]:
            return clear(token) + (
                "Gaming setups",
                gr.update(
                    choices=[(v["name"], v["id"]) for v in first],
                    value=first[0]["id"],
                    visible=False,
                ),
                "",
                "base",
                42,
                "1024",
                40,
                4,
                gr.update(value=False, interactive=False),
                "dual",
                "minimalist_luxury",
                "straight",
                "stand",
                "dual_side",
                True,
                True,
                True,
                False,
                True,
                False,
                True,
                "ambient",
                gr.update(value="", visible=False),
                "Gaming",
                "Standard desk",
                "Single monitor",
                "Reading armchair",
                True,
                "Work + relaxation",
                gr.update(visible=True),
                theme_card("minimalist_luxury"),
                compose_gaming(GamingConfig())[0],
                gr.update(open=False),
                gr.update(open=False),
                gr.update(open=False),
                gr.update(visible=False),
                gr.update(visible=False),
                None,
                None,
                8,
            )

        reset_button.click(
            reset,
            [session],
            outputs
            + [upload, upload_preview]
            + [
                category,
                preset,
                extra,
                model,
                seed,
                size,
                steps,
                cfg,
                rga,
                *gaming_inputs,
                gaming_group,
                theme_summary,
                composed,
                equipment_section,
                advanced_section,
                instruction_section,
                work_group,
                relaxation_group,
                protect_mask,
                foreground_mask,
                feather,
            ],
            api_name=False,
            concurrency_limit=1,
            concurrency_id="image-worker",
        )
    app.queue(max_size=4, default_concurrency_limit=1)
    return app


def launch(config: AppConfig, port: int = 7860, open_browser: bool = False) -> None:
    import json

    import psutil

    cache = ROOT / ".cache/app"
    cache.mkdir(parents=True, exist_ok=True)
    for old in cache.glob("studio-*"):
        owner = old / "owner.json"
        if old.is_symlink() or old.resolve().parent != cache.resolve() or not owner.is_file():
            continue
        try:
            record = json.loads(owner.read_text())
            alive = (
                psutil.pid_exists(record["pid"])
                and abs(psutil.Process(record["pid"]).create_time() - record["created"]) < 1
            )
        except (OSError, ValueError, KeyError, psutil.Error):
            alive = True  # Unknown ownership is not permission to remove a directory.
        if not alive:
            shutil.rmtree(old)
    run_dir = Path(tempfile.mkdtemp(prefix="studio-", dir=cache))
    (run_dir / "owner.json").write_text(
        json.dumps({"pid": os.getpid(), "created": psutil.Process().create_time()})
    )
    os.environ["GRADIO_TEMP_DIR"] = str(run_dir / "gradio")
    import gradio as gr

    service = AppService(run_dir / "sessions", config)
    stopped = threading.Event()

    def expire() -> None:
        while not stopped.wait(300):
            service.cleanup()

    threading.Thread(target=expire, daemon=True).start()
    atexit.register(shutil.rmtree, run_dir, True)
    app = build_app(service)
    try:
        app.launch(
            server_name="127.0.0.1",
            server_port=port,
            share=False,
            inbrowser=open_browser,
            allowed_paths=[str(service.directory)],
            blocked_paths=[
                str(ROOT / p)
                for p in (
                    ".git",
                    ".aws",
                    ".codex",
                    "runs",
                    "outputs",
                    "data",
                    "models",
                    "checkpoints",
                )
            ],
            max_file_size="20mb",
            run_history=False,
            footer_links=[],
            strict_cors=True,
            theme=gr.themes.Soft(
                primary_hue="violet",
                neutral_hue="slate",
                font=["Segoe UI", "sans-serif"],
                font_mono=["Consolas", "monospace"],
            ),
            css_paths=[Path(__file__).with_name("app.css")],
        )
    finally:
        stopped.set()
        app.close()
        shutil.rmtree(run_dir, ignore_errors=True)
        if config.remote_session is not None:
            # Forced process exit is covered by the separate idle/deadline guard.
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/app_runpod.py"),
                    "stop",
                    "--state",
                    str(config.remote_session),
                ],
                timeout=60,
                check=False,
            )
