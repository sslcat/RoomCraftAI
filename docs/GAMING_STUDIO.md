# Gaming Room Studio V1.1

Upload a real empty-room photo, choose a setup, pick a theme, then visualize, compare and download. A sensible dual-monitor / Minimalist Luxury / subtle ambient configuration is ready by default. Equipment and inference settings are in separate accordions. The existing living-room, bedroom and modern-luxury presets remain available through Design category.

## Setups and themes

Six setups provide suggested desks and monitors: compact corner, single monitor, dual monitor, triple monitor, streaming/creator and premium ultrawide. Changing a setup resets its suggested desk and monitor arrangement; subsequent equipment edits take precedence. A creator setup adds a supported microphone boom, camera and modest key light.

| Theme | Furniture and accessories |
|---|---|
| Pirate's Cove | Dark timber, brass, nautical artwork, treasure-inspired storage and electric lantern styling |
| Cat Kingdom | Cozy pastel furniture, cat-ear headset details, paw motifs and optional playful shelving |
| Underwater Realm | Ocean tones, dry aquatic decor, electric jellyfish lamps and optional coral-inspired shelving |
| Cyberpunk Neon | Matte dark surfaces, cyan/violet accents and clean modern equipment |
| Enchanted Forest | Natural wood, small greenery, electric mushroom lamps and warm fairy lights |
| Minimalist Luxury | Restrained premium materials, elegant equipment and understated illumination |

A custom theme accepts up to 500 characters. Additional wishes accept up to 2,000. No RGB and subtle ambient choices override colored-light suggestions. Shelving, decorations, chair, headset/stand, speakers, peripherals and cable-management hardware can be selected independently. Monitor arrangement specifies the count: single, side-by-side dual, stacked dual, triple arc or ultrawide. Tower placement is on desk, under desk or a separate stand.

Themes are furniture/accessory styling, not instructions to reconstruct the room. Underwater styling explicitly forbids actual water or submerged electrical equipment. The prompt requests supported monitors, grounded furniture, chair clearance, realistic scale/contact shadows and unobstructed doors. **These are instructions, not guaranteed constraints. A single photograph does not provide verified physical dimensions. Results are conceptual visualizations, not measured installation plans.**

## Implementation and metadata

`src/roomcraft/themes.json` is the six-theme registry. Each definition contains an ID/name, description, hex and named palettes, materials, decorations, shelf styling, lighting, equipment styling and positive/negative constraints. These are original written definitions, not downloaded third-party assets.

`GamingConfig` and `compose_gaming` in `src/roomcraft/gaming.py` deterministically compile the selections into the actual instruction passed through `AppService.generate` to the unchanged Qwen editing worker. Every control changes prompt content. Preview the exact instruction before generation. The registry does not select or train a different neural model.

Downloaded metadata includes `application.version`, `application.studio.configuration`, the exact `theme_definition` snapshot, composer version, derived monitor count and additional wishes. Existing top-level fields retain the actual prompt, pinned model revision, adapter SHA, seed, settings, native dimensions, runtime, memory and PNG SHA. The UI appends the recipe only after verifying worker identity, source, prompt, settings and output hash. Base/LoRA/RGA identity checks remain enforced. The remote code snapshot now includes the theme registry.

**Reset studio** clears original session files, upload, comparison, downloads and history, restores all defaults, and closes equipment/advanced/instruction sections. Clear session preserves design choices. Downloaded files remain under the user's control; Gradio's temporary copies expire separately. Session history is temporary and limited to eight entries.

## Verified scope

- V1.1 browser tests check all six themes, custom theme, six setups, equipment, lighting, model/RGA controls, invalid empty configuration, reset, upload preview and saved-artifact downloads. Desktop 1440px and narrow 390/320px layouts are checked and screenshots inspected.
- Unit transport tests verify that composed instructions/configuration traverse local and Runpod requests and metadata for base, LoRA and RGA. Their historical image fixtures are test doubles, not fresh image-quality evidence.
- V1.0 genuinely generated both base and LoRA images through Gradio at native 768x1024, with verified downloads and Pod deletion. That historical acceptance is preserved separately.
- Three themed gaming examples were generated previously, followed by the successful matched Underwater RGA comparison. The saved-example button still loads a labeled historical V0.7 reading corner, not a new themed output. RGA now defaults on with RoomCraft LoRA; [one positive real-photo result](RGA_VALIDATION.md) does not establish consistent quality. Monitor count and geometry can still be wrong. No gaming-specific weights were trained.

## Run and verify

```bash
uv sync --project environments/app --frozen
uv run --project environments/app python scripts/launch_app.py
# Against the running loopback studio, no paid calls:
uv run --project environments/app python scripts/verify_gaming_app.py
uv run --project environments/app python scripts/verify_app.py
uv run --project environments/app python -m pytest tests/test_app.py tests/test_gaming.py -q
```

For full local inference, install the separate pinned Qwen environment and pass its memory preflight. An 80 GB-class GPU is recommended; the 12 GB development GPU is insufficient. The UI never provisions compute. See [Runpod workflow](RUNPOD.md), [security audit](SECURITY_AUDIT.md) and [model card](MODEL_CARD.md).

## Genuine image evidence

The previously approved Underwater Realm generation and matched RGA off/on pair are preserved in the [verified demonstration](RGA_VALIDATION.md). No additional paid test is needed or authorized for RC3. Other purpose/theme combinations are browser/transport tested, not a new comprehensive GPU quality benchmark.
