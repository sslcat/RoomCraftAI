# RoomCraft model and application card

**Status:** V1.1 Gaming Room Studio release candidate. Themes/equipment are deterministic instructions, not new weights. Three themed outputs and a matched Underwater RGA comparison have been generated previously; this final audit uses CPU/browser checks without new GPU inference. One positive geometry example does not establish consistent improvement. RoomCraft adapters were trained for research and remain unpublished. No gaming-specific adapter has been trained. Exact architectural preservation is not guaranteed.

| Component | Identity |
|---|---|
| Foundation | `Qwen/Qwen-Image-Edit-2511@6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9` |
| V0.6 LoRA SHA256 | `1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6` |
| Calibrated RGA SHA256 | `31625ddd4f82f10c8ff08947ffe68aa8d08682813c2b47e11cab15321ae56fca` |
| Format | Project-format safetensors/config; third-party LoRA interchange is not established |

Qwen declares Apache-2.0; original code is MIT. Data/image/dependency terms remain separate. No neural weights are bundled.

## Training and engineering

V0.6 trained for 100 optimizer updates, rank/alpha 8/8, 23,543,808 FP32 trainable parameters, BF16 frozen base/autocast, accumulation 4, AdamW LR 1e-4 and gradient clipping. There are 100 licensed aligned Blender pairs, split 70/15/15 by scene family, with 512x384 targets. Test families do not enter fitting or validation loss. The project owns its paired conditional flow loop, projection selection, gradient reporting and checkpoint/resume contracts. Qwen's architecture, VAE/VLM and preprocessing are upstream. [Equations and tensor shapes](QWEN_TRAINING_V04.md).

RGA is an original 728,512-parameter encoder for Canny/optional supplied cues and XY coordinates, injecting target-token residuals through Qwen's existing block interface. Full-model training and checkpoint replay occurred in V0.9; calibration fixed an initially ineffective BF16 path. Consistent architectural benefit was not established. Established residual-conditioning concepts are attributed, not claimed as inventions.

The gaming generator produced 100 reviewed original synthetic pairs across 20 families, with 60/20/20 family-disjoint splits. Those pairs have not trained gaming weights.

The final application generated fresh unchanged-Qwen and RoomCraft-LoRA results through Gradio on one licensed real photograph. Native 768x1024 outputs, matched settings, slider, downloads and history passed. This establishes functional integration, not an adapter quality advantage. [Acceptance receipt](security/app_acceptance.json).

## Evaluation and limitations

Previous real-photo comparisons used native 512/768/1024 and matched base/LoRA settings. The later 36-image intervention experiment found appearance changes without demonstrated useful geometry control. A 64-pixel cue shift produced upper-window translation proxies of 0, 0 and -1 pixels in three synthetic rooms. Apartment doorway/closet framing still drifted. [Verified findings](RESEARCH_RESULTS.md).

The data is small, synthetic and correlated. Outputs can move openings, change flooring/materials, distort perspective, merge furniture or block circulation. Synthetic appearance bias and real-photo domain shift remain concerns. Loss and edge/pixel scores are not objective realism benchmarks. Inspect every image; outputs are not construction measurements or proof that furniture fits.

## Deployment and trust

The app records revision, adapter identity/version, seed, settings, runtime, memory and actual output dimensions. RGA defaults on with RoomCraft LoRA, with an accessible OFF baseline; unchanged Qwen remains adapter-free. Missing adapter files cause an error without substituting base Qwen. No public adapter download currently exists.

Measured native-1024 resident inference peak is about 57.3 GiB allocated; an 80 GB GPU and 128 GB host RAM are recommended. The tested 12 GB GPU fails preflight. Cloud execution requires fresh budget approval and actual Pod-termination verification. The app is loopback-only without authentication, not a public hosted service. [Security audit](SECURITY_AUDIT.md).


## Final RoomCraft Studio polish

Independent room-purpose/theme controls and optional local selective-preservation derivatives are documented in [FINAL_POLISH.md](FINAL_POLISH.md). Original model checkpoints and research methods remain unchanged. The matched Underwater comparison now demonstrates one positive example with active calibrated V0.9 RGA: improved wallpaper and framing, with remaining shifts. This does not establish consistent improvement. See [verified RGA comparison](RGA_VALIDATION.md).


## V1.1 RC3 geometry integration

**Preserve Room Geometry — RoomCraft RGA** is visible beside model selection and is automatically enabled when RoomCraft LoRA is selected. Switch it off for LoRA-only inference. Unchanged Qwen remains the initial/reset model choice, with RGA disabled. Selecting LoRA again restores its default RGA-on state. The pinned base, LoRA and calibrated RGA checkpoints and inference mathematics are unchanged. Downloaded metadata records actual geometry conditioning; saved examples retain their historical settings rather than inheriting the current checkbox.

The matched Underwater comparison shows better wallpaper, camera/room framing and visible preservation with active RGA. This is one positive example, not proof of reliable preservation across rooms. Monitor count, small geometric shifts, materials, occlusion and furniture clearance remain limitations. [Full evidence and gallery](RGA_VALIDATION.md). No new GPU generation occurred for this release audit.
