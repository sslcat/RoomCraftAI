# RoomCraft Qwen-Image-Edit-2511 LoRA — V0.6 private pilot

**Status:** genuinely trained, unpublished research adapter, 2026-10-30. Load through RoomCraft's existing Qwen backend and `load_training_adapter`; third-party LoRA format compatibility has not been established.

- Base: `Qwen/Qwen-Image-Edit-2511`, revision `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`.
- Adapter: `outputs/v06/trained/pilot/step-000100/adapter.safetensors`.
- SHA256: `1be7a8089b2fe037e7458a1b20bff5b76fe133dafe6aa4d3a3691a9772a442e6`.
- Objective: project-owned target-only paired rectified-flow velocity MSE with clean source-image/VLM conditioning. Mathematical details: [Qwen training documentation](QWEN_TRAINING_V04.md).
- Training: rank/alpha 8/8; 23,543,808 FP32 trainable parameters, BF16 frozen base/autocast; AdamW LR 1e-4, weight decay 0.01, clip norm 1, accumulation 4, 100 updates/400 microbatches, seed 42, gradient checkpointing. Independent smoke save/reload/resume completed first.
- Data: 100 original licensed Blender pairs; 70 train, 15 validation, 15 test, with disjoint scene families. Target 512x384; source VAE reference approximately 1024²; original V0.6 conditioning preserved. Manifest SHA256: `668b368b3dabeac29b9407b61d67f973755d03d9743f6bdb7a397b480411fd31`.
- Environment: Python 3.12.15, PyTorch 2.8.0+cu128, Diffusers 0.37.1, Transformers 4.57.6; exact lockfiles retained.
- Hardware: one H100 SXM 80 GB; peak training allocation 41.48 GiB. Full resident inference needs more memory; local RTX 4070 SUPER is insufficient for the current unquantized path.

Intended use is reproducible virtual-staging research. Three held-out synthetic examples show closer window/framing preservation than unchanged Qwen under matched settings. Both models still violate some instructions and can alter materials, lighting and geometry. This is a small single-seed qualitative AI review, with no independent human benchmark, real-photo generalization proof or perfect-preservation claim. The synthetic dataset is correlated and shares furniture/layout templates across splits. Controlled seeds do not guarantee bitwise GPU replay.

Final fixed validation MSE is 0.0775585; it is a training diagnostic. Complete logs, actual images, checkpoint/resume verification, costs and source attribution are in the execution report (`../ROOMCRAFT_V06_TRAINING_REPORT.md`, local research artifact excluded from the source candidate). All model outputs are retained, including failures. The adapter remains private; no public inference service is exposed.

The foundation has the archived Apache-2.0 grant. Original dataset contributions use MIT to the extent rights subsist; listed Poly Haven assets are CC0. Retain notices and **Powered by Poly Haven** credit; authors/provenance are recorded with the dataset. Review the adapter's distribution license and all notice obligations before an owner-approved release. Archived website HTML is private rights evidence, not automatically redistributable CC0 content. No full foundation checkpoint is included in the adapter archive.
