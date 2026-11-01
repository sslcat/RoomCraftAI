# Using Gaming Room Studio V1.1

[Theme and equipment guide](GAMING_STUDIO.md).

```bash
uv sync --project environments/app --frozen
uv run --project environments/app python scripts/launch_app.py
```

Open `http://127.0.0.1:7860`. Use `--port 7861` for another loopback port or `--open` to open a browser. There is no public share link, remote model API fallback, login, payments or database.

## Your room to a visualization

1. Upload a JPEG, PNG or static WebP under 20 MB and 24 megapixels. Actual bytes are checked, EXIF orientation is applied and metadata is stripped before inference.
2. Choose a gaming setup, creative theme and lighting. Fine-tune Equipment & placement if desired, or select an interior category/preset. Add optional custom instructions. Preview the exact instruction in its own section.
3. Choose unchanged Qwen or RoomCraft LoRA. The latter requires the verified private checkpoint; missing files never trigger base-model substitution.
4. Click **Visualize my room**. The first full model load can take several minutes. Only one inference runs at a time.
5. Drag the before/after divider, inspect architecture/furniture, and download PNG and settings. Only the comparison source preview is resized to the output dimensions. Generated images are not upscaled to impersonate native high resolution.

Defaults are native long side 1024, 40 steps, CFG 4 and seed 42. Dimensions round to multiples of 64. Equal seeds do not imply identical noise across resolutions. Preserve Room Geometry — RoomCraft RGA turns on by default with RoomCraft LoRA and has a visible OFF switch beside model selection. Unchanged Qwen uses neither adapter. One matched real-photo example improved preservation; consistent benefit remains unproven. See [RGA validation](RGA_VALIDATION.md).

Up to eight results stay in the current session. Selecting a thumbnail restores its image and metadata. Reset studio also restores design/model/settings defaults and closes advanced sections. Clear session retains design choices and removes the session's original result files; Gradio's upload/download copies expire separately within an hour or at process exit. Download anything you want to keep. Identifiable stale studio caches are cleaned after an abnormal exit when the owning process is absent.

## Model worker

```bash
uv sync --project environments/qwen --frozen
uv run --project environments/qwen roomcraft edit-preflight
```

Expect roughly 54 GiB of weights, at least 60 GiB free VRAM for the current conservative resident mode, and at least 64 GiB available host RAM. An 80 GB GPU with 128 GB host RAM is recommended. The local 12 GB GPU fails before weight downloading. The app introduces no hidden quantization/offload fallback.

The launcher chooses the Qwen environment's Python for Windows/Linux. Override with `--worker-python PATH`, `--lora PATH` or `--rga PATH`. There is no public adapter URL yet. For a separately approved allocation, pass `--runpod-session .cache/YOUR_FRESH_SESSION`; see [Runpod](RUNPOD.md).

## Saved examples and acceptance

**Explore a saved real-room result** loads a genuine previous V0.7 base/LoRA result. Status and metadata explicitly say `saved_example`; this click does not run inference.

With a running UI:

```bash
uv run --project environments/app python scripts/verify_app.py
# Only with a separately approved GPU worker:
uv run --project environments/app python scripts/verify_app.py --live --output .cache/app-live-acceptance
```

Saved mode checks upload/preset controls, real historical image hashes, slider movement, both model identities, downloads and clearing. Live mode invokes the actual Generate button for both models and checks matched sources/settings and native output integrity. No inference response is mocked.

Invalid images, missing checkpoints, insufficient memory, expired sessions, worker timeouts and unavailable Runpod connections produce errors. Failed local job photos/partial outputs are removed. Connection failure may leave an isolated cloud job directory requiring operator cleanup; process exit is not proof of Pod termination.


## Historical V1.0 live acceptance

This is the previous real-model acceptance, not a fresh themed-gaming test. V1.1 free browser acceptance is recorded in [the separate receipt](security/gaming_acceptance.json).

On 2026-10-31 a real browser uploaded the licensed hotel photograph, selected Living rooms / Quiet reading corner, generated with each model, moved the comparison divider, downloaded PNG/settings and selected session history. Both outputs were native 768x1024 with seed 42, 40 steps, CFG 4 and RGA disabled. There were no browser errors. [Receipt](security/app_acceptance.json).

| Model | Backend total, including loading | Sampling time | Peak allocated VRAM |
|---|---:|---:|---:|
| Unchanged Qwen | 317.12 s | 38.43 s | 56.93 GiB |
| RoomCraft LoRA | 121.23 s | 39.82 s | 57.02 GiB |

The complete browser journey took 519.44 seconds. These backend totals exclude some browser/transport/process overhead and differ in cache state; they do not establish a speed advantage. The live session was retained for evidence, so clear was tested separately in saved mode. Both remote temporary jobs were cleaned. Automatic Pod deletion was independently verified through HTTP 404 after 10 minutes 50 seconds. Estimated compute/container plus storage allowance was $0.77, pending settled billing. No new training occurred.

Actual outputs show plausible chair/table/lamp placement, with changed wallpaper, curtain and lighting details. Exact architecture preservation and LoRA quality superiority are not established by this case. The README contact sheet shows these fresh results; the bundled saved-example button still uses historical V0.7 outputs.


## Final RoomCraft Studio polish

Independent room-purpose/theme controls and optional local selective-preservation derivatives are documented in [FINAL_POLISH.md](FINAL_POLISH.md). Original model checkpoints and research methods remain unchanged. The matched Underwater comparison now demonstrates one positive example with active calibrated V0.9 RGA: improved wallpaper and framing, with remaining shifts. This does not establish consistent improvement. See [verified RGA comparison](RGA_VALIDATION.md).


## V1.1 RC3 geometry integration

**Preserve Room Geometry — RoomCraft RGA** is visible beside model selection and is automatically enabled when RoomCraft LoRA is selected. Switch it off for LoRA-only inference. Unchanged Qwen remains the initial/reset model choice, with RGA disabled. Selecting LoRA again restores its default RGA-on state. The pinned base, LoRA and calibrated RGA checkpoints and inference mathematics are unchanged. Downloaded metadata records actual geometry conditioning; saved examples retain their historical settings rather than inheriting the current checkbox.

The matched Underwater comparison shows better wallpaper, camera/room framing and visible preservation with active RGA. This is one positive example, not proof of reliable preservation across rooms. Monitor count, small geometric shifts, materials, occlusion and furniture clearance remain limitations. [Full evidence and gallery](RGA_VALIDATION.md). No new GPU generation occurred for this release audit.
