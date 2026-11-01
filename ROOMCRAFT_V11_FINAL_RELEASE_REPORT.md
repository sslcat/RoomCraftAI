# RoomCraft Studio V1.1 RC3 — final release report

## Release result

The source candidate integrates the unchanged, calibrated V0.9 RGA as the default-on geometry option when RoomCraft LoRA is selected. The visible **Preserve Room Geometry — RoomCraft RGA** checkbox provides the OFF baseline. Initial/reset unchanged-Qwen selection is retained and uses neither adapter. No checkpoint tensors, inference mathematics, paid resources or unrelated projects were changed.

The verified Underwater example demonstrates better wallpaper, room framing and visible architectural preservation with active RGA. This is one positive real-photo/seed result; previous broader studies remain mixed. Monitor-count accuracy, small geometric shifts, material/color changes, occlusion and physical furniture fit remain limitations. [Full evidence and attributed native gallery](docs/RGA_VALIDATION.md).

## Verification

The complete base and Qwen CPU suites, UI acceptance tests, lint, formatting, type checks and compilation were run. Browser verification covers all four purposes × six themes, default-on/OFF RGA, unchanged-Qwen/reset behavior, equipment/lighting, native saved downloads, slider comparison, session history/clear and 1440/390/320-pixel layouts. Genuine RGA-off/on outputs are replayed through acceptance and wrong-switch metadata is rejected. Model-scale activation is established by the preserved actual Underwater run; this release audit creates no GPU resource and makes no claim of fresh quality testing across every mode/theme.

The final machine-readable verification receipt beside the ZIP records exact packaged test counts, clean frozen UI install and launch, hashes, privacy/link checks and preserved archive/checkpoint identities. Public browser receipts are in `docs/security/rc3_*_acceptance.json`; detailed local logs remain excluded from the release.

Packaged verification completed: **158 Qwen tests passed; 144 base tests passed with 14 expected skips; 46 UI/transport tests passed in a fresh ZIP installation**. All three browser journeys passed both in the workspace and the clean extracted installation, including 24 purpose/theme combinations. Lint/format checks cover 101 packaged Python files; both environment type checks cover 30 source modules. Compilation, frozen lock checks, archive hashes, and all public Markdown/gallery links are verified by the final audit receipt. The model-scale conditioning evidence is from the preserved genuine comparison, not a new release-audit GPU run.

## Security and license disposition

- Fresh installed-distribution audit: UI 0 known advisory entries; Qwen 88 entries (66 distinct IDs), SDXL 92 entries (70 distinct IDs), across four packages each. Worker advisories remain explicitly open with IDs/fix versions in [audit evidence](docs/security/dependency_audit.json). No finding was suppressed or described as fixed without remediation.
- The UI intentionally uses patched Pillow 13 despite Gradio's `<13` metadata bound. This one compatibility exception remains documented; actual upload/render/download/browser checks pass. Qwen and SDXL dependency checks pass.
- Original code is MIT. Qwen declares Apache-2.0 at the pinned official model card; SDXL has separate model terms. Reviewed Poly Haven training assets are CC0; their binaries/full datasets are excluded. The demo photo, AI derivatives and screenshots credit upyernoz under CC BY 2.0, with source/license/modification notices. No private owner photo, unknown-license dataset or unauthorized external theme art is included. [License ledger](docs/security/license_audit.json), [notices](NOTICE.md).
- The release allowlist excludes credentials, keys, caches, owner profiles, private account/worker receipts, weights, optimizer state, complete datasets, third-party asset binaries and archived website HTML. Only selected inference identity/settings/activation data is public. A text privacy scan, type/size gates and SHA256 manifest verify the final source artifact. The raw workspace is not the publication artifact.

## Remaining limitations and publication blockers

**No unresolved functional blocker to the documented local UI/source release remains if the final artifact verification receipt passes.** The code and explicitly labeled saved-example flow install and launch without private weights.

**Complete public LoRA/RGA inference remains blocked on separately approved adapter distribution and license/hosting delivery.** The trained V0.6 LoRA and calibrated V0.9 RGA are private, excluded from this ZIP, and have no public download URL. Qwen alone can use its public pinned foundation on suitable hardware. Missing/mismatched RoomCraft adapters fail without fallback.

**Untrusted public inference hosting is not security-cleared.** Open pinned model-worker advisories and the UI dependency-bound exception remain disclosed. The unauthenticated loopback application must not be presented as a hardened public service. Consistent geometric preservation and all-purpose image quality are also not established.

Only the reviewed candidate may be considered for later publication. RC1/RC2 archives and trained checkpoint/config hashes are preserved. This task performs no commit, push, GitHub release, model hosting, paid allocation, retraining, or changes to Brasik/Rulbi.
