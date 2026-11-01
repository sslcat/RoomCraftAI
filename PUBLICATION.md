# RoomCraft AI v1.1.0 — source release provenance

This initial public release is promoted from the audited V1.1 RC3 source candidate. The immutable local RC3 archive SHA256 is `30fd00b582301a54b104faf51d2d2b05215ccea47f7d003daae66d56d182dc6c`. RC1, RC2, RC3 and the original research workspace/history remain preserved locally. Historical report filenames and RC3-era screenshots are retained as provenance, not descriptions of additional unpublished features.

Only release version metadata, the UI version label, this provenance notice, the release policy version, and a historical planning example's non-authorizing flag differ from the reviewed candidate. Text line endings are normalized to LF so Git blobs and release-archive manifest hashes agree across platforms. Python ASTs and parsed JSON/TOML/locks were checked unchanged by this normalization. Runtime dependencies, model revisions, inference mathematics and trained adapter identities are unchanged. The public Git history begins with one clean release commit; prior private workspace history is not redistributed or rewritten. `release_manifest.json` records every approved source file, its hash and its RC3 origin hash.

## Included

RoomCraft Studio, four room purposes, six themes plus custom themes, Qwen integration, original LoRA/RGA engineering and training mathematics, reproducibility code, original procedural test fixtures, dataset/license provenance, attributed demonstration images, and security/quality disclosures. The matched Underwater RGA result is one positive real-photo, single-seed observation, with remaining monitor-count/material/geometric limitations. It is not a universal preservation guarantee.

## Excluded and limitations

**No trained LoRA/RGA adapters, foundation weights, optimizer states, full datasets, private room photos, account records, credentials, caches, temporary GPU artifacts or third-party asset binaries are distributed.** The real-demo photo and derivatives have CC BY 2.0 attribution/modification notices. Adapter-enabled inference requires separately obtained compatible private weights and configs; there is no public weight download in this initial release. Saved examples and public foundation-Qwen inference have distinct documented paths.

Pinned Qwen/SDXL workers retain disclosed dependency advisories. The UI has a tested Gradio/Pillow metadata-bound exception. This local unauthenticated application is not a hardened public inference service. See [security audit](docs/SECURITY_AUDIT.md), [adapter distribution](docs/ADAPTER_DISTRIBUTION.md), [original RC3 report](ROOMCRAFT_V11_FINAL_RELEASE_REPORT.md), and [licenses](NOTICE.md).

Historical cloud planning files are examples and confer no budget/deployment authorization. Publication creates no GPU resource and retrains no model. Release assets contain only the independently reviewed source archive, manifest, verification record and checksums.
