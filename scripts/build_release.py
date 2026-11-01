"""Build an audited source-only candidate; never commit, push, upload or deploy."""

import ast
import fnmatch
import hashlib
import json
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def public_lifecycle(source):
    """Keep the exact tested library functions; omit archival allocation entrypoints."""
    tree = ast.parse(source)
    names = {"save", "api", "session", "identity", "terminate", "guard", "transport"}
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            nodes.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in ("ROOT", "STATE") for t in node.targets
        ):
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
    actual = {n.name for n in nodes if isinstance(n, ast.FunctionDef)}
    if actual != names:
        raise ValueError("Lifecycle library shape changed; review release policy")
    code = (
        '"""Shared exact-ID lifecycle library; historical allocation entrypoints excluded."""\n\n'
        + ast.unparse(ast.Module(body=nodes, type_ignores=[]))
        + "\n"
    )
    fixed = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            "--fix",
            "--stdin-filename",
            "scripts/roomcraft_runpod.py",
            "-",
        ],
        input=code,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    formatted = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "format",
            "--stdin-filename",
            "scripts/roomcraft_runpod.py",
            "-",
        ],
        input=fixed,
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    published = ast.parse(formatted)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            other = next(
                n for n in published.body if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            if ast.dump(node, include_attributes=False) != ast.dump(
                other, include_attributes=False
            ):
                raise ValueError("Release processing changed lifecycle behavior")
    return formatted


def scan(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    patterns = {
        "private_key": r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
        "api_token": r"\b(?:hf_|rpa_|sk-)[A-Za-z0-9_-]{25,}\b",
        "aws_access_key": r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b",
        "personal_windows_home": r"[A-Z]:[\\/]+Users[\\/]+[A-Za-z][^\s\"<>]*",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, text)]


def main():
    policy = json.loads((ROOT / "configs/release_policy.json").read_text())
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = "RoomCraftAI-v" + policy["version"]
    stage = dist / name
    marker = stage / ".roomcraft-candidate.json"
    if stage.exists():
        if (
            stage.is_symlink()
            or stage.resolve() != (dist.resolve() / name)
            or not marker.is_file()
            or json.loads(marker.read_text()).get("format") != "roomcraft-source-candidate-v1"
        ):
            raise ValueError("Refusing to replace an unrecognized directory")
        shutil.rmtree(stage)
    stage.mkdir()
    selected = set()
    for pattern in policy["include"]:
        selected.update(p for p in ROOT.glob(pattern) if p.is_file())
    records, issues = [], []
    denied = {
        ".safetensors",
        ".ckpt",
        ".pt",
        ".pth",
        ".bin",
        ".pem",
        ".key",
        ".blend",
        ".exe",
        ".dll",
    }
    for source in sorted(selected):
        rel = source.relative_to(ROOT).as_posix()
        if any(fnmatch.fnmatch(rel, x) for x in policy["exclude"]):
            continue
        if (
            source.is_symlink()
            or source.suffix.lower() in denied
            or source.stat().st_size > policy["maximum_file_bytes"]
        ):
            raise ValueError(f"Prohibited source artifact: {rel}")
        target = stage / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        transform = rel == policy["lifecycle_library_only"]
        if transform:
            target.write_text(public_lifecycle(source.read_text()), encoding="utf-8")
        else:
            shutil.copyfile(source, target)
        if target.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            findings = scan(target)
            if findings:
                issues.append({"path": rel, "rules": findings})
        license_id = (
            "CC-BY-2.0"
            if rel.startswith("docs/assets/") and target.suffix == ".png"
            else "CC0-1.0"
            if rel.startswith("tests/fixtures/")
            else "SEE-NOTICE"
            if rel.startswith("docs/licenses/") or rel.startswith("docs/sources/")
            else "MIT-original-contribution"
        )
        records.append(
            {
                "path": rel,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "source_sha256": digest(source),
                "transform": "AST-library-only" if transform else None,
                "license_scope": license_id,
            }
        )
    if issues:
        (dist / "release-audit-issues.json").write_text(json.dumps(issues, indent=2))
        raise ValueError(
            "Candidate failed credential/privacy scan; inspect filenames/rules in dist/release-audit-issues.json"
        )
    # No source-data/checkpoint directories can enter through these allowlisted patterns.
    marker.write_text(
        json.dumps({"format": "roomcraft-source-candidate-v1", "version": policy["version"]})
    )
    manifest = {
        "format": "roomcraft-release-manifest-v1",
        "version": policy["version"],
        "source_files": records,
        "files_count": len(records),
        "credential_privacy_scan": "passed",
        "weights_included": False,
        "datasets_included": False,
        "publication": "not published; final owner approval required",
        "dependency_caveat": "See docs/SECURITY_AUDIT.md; UI Pillow override and historical model-worker advisories are disclosed.",
        "policy_sha256": digest(ROOT / "configs/release_policy.json"),
    }
    (stage / "release_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = dist / (name + ".zip")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zip:
        for p in sorted(stage.rglob("*")):
            if p.is_file() and p != marker:
                entry = zipfile.ZipInfo(
                    name + "/" + p.relative_to(stage).as_posix(), date_time=(2026, 10, 31, 0, 0, 0)
                )
                entry.compress_type = zipfile.ZIP_DEFLATED
                zip.writestr(entry, p.read_bytes())
    receipt = {
        "version": policy["version"],
        "archive": archive.name,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": digest(archive),
        "manifest_sha256": digest(stage / "release_manifest.json"),
        "source_files": len(records),
        "published": False,
    }
    (dist / "release_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
