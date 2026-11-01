# Security and privacy

The studio binds **127.0.0.1 only**, disables public Gradio sharing/analytics/browser run history, and has no authentication. Do not expose it publicly or add a public proxy without a separate security design.

Uploads are restricted to JPEG, PNG and static WebP, 20 MB and 24 megapixels. Actual bytes are checked, orientation corrected, metadata removed and RGB PNG produced before the model worker. Instructions are JSON, never shell interpolation. Model/adapter identity, returned settings, dimensions and PNG hashes are verified; there is no silent adapter fallback.

Sessions use random server-side identifiers and dedicated temporary directories, with eight recent images. History is not persistent across app restarts. Clear removes original session files; cached upload/download copies expire separately within an hour or at process exit. Browser/OS downloaded files remain under the user's control. Stale-cache cleanup targets only identifiable old RoomCraft studio directories.

Runpod uses SSH and isolated UUID jobs on dedicated RoomCraft storage. Cleanup is attempted after recovery; connection failures may leave temporary data on retained storage for operator review. The app does not send photos to an inference API or provision compute. Operator startup requires fresh authorization, pricing, exact resource checks and a deadline guard. The guard depends on the Windows machine/network; it is not a cloud-independent overnight guarantee. Verify actual Pod deletion with fresh authenticated HTTP 404. Storage continues billing while retained.

Credentials stay in memory, resolved from the operator's process or existing Windows Credential Manager entry. SSH keys remain outside the repository. The ignored owner profile stores nonsecret target/path/resource bindings only.

See [the dependency audit](docs/SECURITY_AUDIT.md): the UI uses patched Pillow 13 with a documented and browser-tested Gradio bounds override. Historical model environments retain upstream advisories and are not represented as vulnerability-free. Use trusted pinned safetensors/configs, isolate research workers and never expose training-state loading to untrusted files.

The source candidate excludes credentials, private receipts, complete datasets and weights. Public examples have explicit image terms. Rerun the release builder/audit before owner-approved publication. Report security issues privately to the maintainer; no fictional security contact is supplied before a public repository exists.
