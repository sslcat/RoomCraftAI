"""Exercise required POSIX operations on the exact isolated project filesystem."""

import argparse
import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path


def verify(root):
    import fcntl

    root = root.resolve()
    if root != Path("/workspace/RoomCraftAI"):
        raise ValueError("probe must stay inside the owned RoomCraft project")
    folder = Path(tempfile.mkdtemp(prefix=".roomcraft-posix-probe-", dir=root))
    try:
        a, b = folder / "source", folder / "renamed"
        with a.open("xb") as stream:
            stream.write(b"RoomCraft atomic checkpoint probe")
            stream.flush()
            os.fsync(stream.fileno())
        a.rename(b)
        if b.read_bytes() != b"RoomCraft atomic checkpoint probe" or a.exists():
            raise RuntimeError("rename/read-back failed")
        os.chmod(b, 0o600)
        permissions = stat.S_IMODE(b.stat().st_mode) == 0o600
        os.link(b, folder / "hardlink")
        os.symlink("renamed", folder / "symlink")
        if (folder / "hardlink").read_bytes() != b.read_bytes() or (
            folder / "symlink"
        ).read_bytes() != b.read_bytes():
            raise RuntimeError("link semantics failed")
        with b.open("r+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            code = "import fcntl,sys; f=open(sys.argv[1],'r+b');\ntry: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(0)\nelse: sys.exit(2)"
            child = subprocess.run([sys.executable, "-c", code, str(b)], timeout=20)
            if child.returncode != 0:
                raise RuntimeError("cross-process exclusive lock not enforced")
        return {
            "status": "required_checkpoint_operations_passed",
            "root": str(root),
            "posix_permission_bits_retained": permissions,
            "tested": [
                "fsync",
                "same-directory rename/readback",
                "chmod",
                "hardlink",
                "symlink",
                "cross-process flock exclusion",
            ],
            "limitation": "Functional check, not a power-loss durability proof. Permission-bit emulation may differ; no credentials or private SSH keys may be written to this public-data volume.",
        }
    finally:
        for name in ("source", "renamed", "hardlink", "symlink"):
            (folder / name).unlink(missing_ok=True)
        folder.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.root)))


if __name__ == "__main__":
    main()
