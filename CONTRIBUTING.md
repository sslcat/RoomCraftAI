# Contributing

Use Python 3.12 and the separate locked UI, SDXL and Qwen environments. CPU/tiny-transformer tests verify contracts, not image quality.

```bash
uv sync --frozen
uv run python scripts/make_fixture.py
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run --project environments/qwen python -m pytest -q
uv run --project environments/app python -m pytest tests/test_app.py -q
```

For browser verification, install Playwright Chromium in `.cache/playwright`, launch the UI and run `scripts/verify_app.py`. Saved mode remains labeled as historical-output verification. Never mock an inference response and claim full-model generation. New GPU tests need fresh budget/lifetime approval and authenticated termination evidence.

Install the browser once from the repository root. PowerShell:

```powershell
$env:PLAYWRIGHT_BROWSERS_PATH = "$PWD/.cache/playwright"
uv run --project environments/app python -m playwright install chromium
```

On Linux/macOS use `PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright" uv run --project environments/app python -m playwright install chromium` instead. The browser-verification scripts use that project-local directory by default; application users do not need Playwright to launch the studio.

Keep exact model identity, provenance, family-disjoint splits, tensor contracts and baseline behavior. Distinguish synthetic/real images, automated proxies and attributed visual judgments. Do not claim gaming or geometry improvement without evidence.

Exclude credentials, private photos, caches, large weights, optimizer state, account-bound receipts and unlicensed assets from Git. Attribute photos in screenshots. Review the curated release manifest instead of pushing a private research workspace wholesale.


## Final Studio browser checks (no GPU)

Start the UI with the README quick start. In a second terminal:

```bash
uv run --project environments/app python scripts/verify_studio.py
uv run --project environments/app python scripts/verify_gaming_app.py
uv run --project environments/app python scripts/verify_app.py
uv run --project environments/app python -m pytest tests/test_app.py tests/test_gaming.py tests/test_studio.py tests/test_rga_release.py -q
```

The first journey covers all 24 purpose/theme combinations and default-on RGA with LoRA, explicit OFF, unchanged Qwen and reset. The other journeys cover equipment, layouts, native downloads, slider, history and session clearing using labeled genuine saved artifacts. None provisions compute or invokes fresh generation. `test_rga_release.py` replays the actual verified off/on artifacts through acceptance and rejects switch/metadata mismatches. Existing RGA tests cover strict loading and non-mutating residual diagnostics. Never use `--live` without separately authorized compute.
