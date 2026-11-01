# Adapter distribution decision

Keep weights out of Git. The V0.6 LoRA is **94,277,832 bytes**; calibrated RGA is **2,915,656 bytes**. A separate model host is preferable for immutable versions, checksums, model cards and notices. No hosting repository or public release was created.

After separate owner approval, release only verified adapter safetensors, matching config, model card and license/attribution. Exclude optimizer state, caches, private photos, full foundation weights and operational logs. These are project-format adapters; do not advertise PEFT/ComfyUI interchange without a tested conversion.

The proposed separate weight license is Apache-2.0 retaining Qwen attribution, subject to final owner review. Original synthetic contributions and reviewed CC0 assets support the pilot's provenance; this conveys no rights guarantee about unknown foundation pretraining data. [Exact checkpoint license record](licenses/qwen_license_record.json).

There is **no public download URL yet**. The app requires the private matching checkpoint and refuses absent/mismatched bytes. Do not invent a Hub repository or silently fall back to base inference.

| Adapter | SHA256 |
|---|---|
| V0.6 LoRA | `1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6` |
| Calibrated RGA | `31625ddd4f82f10c8ff08947ffe68aa8d08682813c2b47e11cab15321ae56fca` |

The UI labels the calibrated checkpoint **Preserve Room Geometry — RoomCraft RGA** and defaults it on with RoomCraft LoRA. One positive matched real-photo example is documented in [RGA validation](RGA_VALIDATION.md); consistent benefit remains unproven. A gaming-specific trained adapter must not be advertised. Private checkpoint delivery remains a limitation for public end-to-end LoRA/RGA use.
