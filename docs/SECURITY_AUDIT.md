# V1.1 RC3 release security and license audit

This is a bounded local-application audit, not a security certification. Package advisory data was queried with pip-audit 2.10.1 against the actual installed UI, SDXL and Qwen environments. [Machine-readable findings](security/dependency_audit.json) retain advisory IDs, aliases and fix versions. Counts are tool entries; duplicate aliases/entries are not silently counted as distinct issues.

## UI remediation and compatibility exception

Initial UI findings included Pillow 12.3.0 and pytest 8.4.2. The UI now pins **Pillow 13.0.0** and **pytest 9.0.3**. A fresh installed-distribution audit reports **zero known vulnerabilities** in that environment, including its dev tools. This says nothing about unknown vulnerabilities, the browser binary or the legacy model environments.

Gradio 6.31.0 currently declares `Pillow<13`; its release predates these Pillow fixes. The app's lock deliberately overrides that upper bound. `uv pip check` therefore reports **one known metadata incompatibility** in the UI. Actual JPEG/PNG/WebP validation, EXIF handling, saved-image rendering, comparison dragging, native PNG/settings downloads and both model example identities are tested with Pillow 13. Fresh base/LoRA generation, comparison and downloads through the live application also passed on the bounded H100 worker. The exception is documented rather than hiding the dependency check. Remove it when an upstream compatible Gradio release is verified. It is not an assertion that every Gradio feature supports Pillow 13.

The root package metadata now accepts both existing SDXL/Qwen framework ranges, while their locks still pin their original runtime versions. This fixes the old Qwen `pip check` mismatch caused by SDXL-only package metadata. SDXL and Qwen dependency checks pass. Original pre-release research locks are retained in `docs/reproducibility/`; they are historical records, not the recommended UI installation.

## Retained research-worker advisories

Fresh scans report **88 entries across four Qwen packages**, and **92 across four SDXL packages**, involving Accelerate, Diffusers, Pillow and Transformers. See the JSON for exact IDs/fix versions/distinct counts. These findings remain open; the workers are **not vulnerability-free**.

Blindly moving the trained stack to new major framework versions would require another compatibility/inference evaluation. This MVP preserves its pinned model runtime and records the limitation. The supported UI path decodes untrusted photos in patched Pillow first, strips metadata and sends its own bounded RGB PNG to an isolated worker. Model/config/checkpoint paths are operator-controlled, the foundation revision is pinned, safetensors are required for inference, and adapter hashes are verified. No user upload can select arbitrary model code, a weight URL or a training-state file. These controls reduce reachable input surfaces; they are **not proof that every advisory is unreachable**.

Use only trusted sources/checkpoints and do not expose the unauthenticated loopback app as a public service. Upgrading the model environment is a separate, testable maintenance task, not a completed fix in this release.

## Source candidate boundaries

The allowlisted candidate excludes `.cache`, credentials/keys, owner profiles, private account receipts, original room collections, model caches, neural weights, optimizer state, third-party asset binaries and archived website HTML. A credential/private-path scan runs over candidate text. File size/type checks and per-file SHA256 are recorded in `release_manifest.json`. Historical allocation entrypoints are omitted from the public lifecycle library; its tested identity/auth/guard/transport/deletion functions remain.

The raw research workspace is **not** the publication artifact. Publish only the reviewed candidate after final owner approval. Regex scanning is not a guarantee that no possible secret exists; review the manifest and notices too.

## License findings

- Original source is MIT; dependency and model terms remain separate.
- Exact Qwen revision: official model card declares Apache-2.0; no standalone LICENSE at that snapshot. Canonical terms and declaration provenance are retained under `docs/licenses`.
- Bundled hotel photo, AI derivatives and screenshots credit upyernoz under CC BY 2.0, with source/license/modification notices. No private owner photo is included.
- Procedural fixtures are original CC0. Dataset generators/asset lock preserve original/MIT and reviewed CC0 provenance, but dataset and asset binaries are excluded.
- Adapters are excluded. Separate hosting/license review requires final owner approval; no public download is invented.

This supports review of a source release candidate with disclosed limitations. It does not approve publishing model weights, claim commercial reliability, or replace final owner publication approval.


## V1.1 refresh

Audits were rerun against all three installed environments on 2026-10-31: UI zero known advisories; Qwen 88 entries and SDXL 92 entries across four packages each. The Pillow bounds exception remains. Package/lock changes update only RoomCraft's product version; model runtime packages are unchanged. Themes are original text/hex-color records with no external assets. Custom text is passed as JSON to the existing worker, never interpolated into shell commands. New controls and reset were browser-tested without paying for compute. At this earlier checkpoint themed quality acceptance was pending; subsequent [matched Underwater evidence](RGA_VALIDATION.md) is now available. Earlier V1.0 acceptance remains historical.


## Final Studio polish review

No runtime dependency versions changed. The 2026-10-31 advisory inventory above remains the disclosed audit, not a new zero-vulnerability claim. Added masks use the existing image byte/type/size validation and require source dimensions; they stay local to session jobs. Composites preserve raw hashes and are separately labeled. No public hosting or authentication change. Existing Pillow compatibility override, model-worker advisories and adapter-license decisions remain unresolved owner-review items.


## V1.1 RC3 final release audit

Live pip-audit queries against the installed distributions were repeated for RC3: **UI: 0 known advisory entries; Qwen: 88 entries across 4 packages; SDXL: 92 entries across 4 packages**. Exact advisory IDs, aliases, affected versions and advertised fix versions are retained in [dependency_audit.json](security/dependency_audit.json). Entries are not necessarily unique CVEs. No finding was suppressed to achieve a clean result.

Disposition: UI vulnerabilities previously addressed by the patched Pillow/pytest pins remain resolved in this inventory. The single Gradio `Pillow<13` metadata conflict remains an explicit compatibility exception; native uploads, downloads, comparison/history and browser journeys pass with Pillow 13. Model-worker findings remain open in Accelerate, Diffusers, Pillow and Transformers. The trained runtime is pinned for reproducibility and has not been silently upgraded without full-model compatibility evaluation. Restrict it to trusted operator-selected models/checkpoints and sanitized bounded PNGs from the patched UI; these controls do not prove every advisory unreachable. **This candidate is not suitable to expose as an untrusted public inference service.** Advisory remediation is a remaining blocker for that deployment profile, not claimed completed.

The reviewed source archive is a local application/source release with these explicit limitations, not a vulnerability-free binary or complete model distribution. Full public LoRA/RGA use also remains blocked on separately approved adapter hosting/license delivery. The final release performs no paid allocation, training, publication or changes to unrelated resources. Raw comparison/job receipts were sanitized into selected settings/hashes/activation evidence; no account IDs, credentials, SSH paths or private telemetry are bundled. Rights notices cover every included room image/derivative. See [RGA evidence](RGA_VALIDATION.md).
