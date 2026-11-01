"""Make review sheets or export explicitly approved pairs for existing V0.4 sealing."""

import argparse
from pathlib import Path

from roomcraft.common import write_json
from roomcraft.render_dataset import make_gallery, reviewed_ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("gallery", "ledger"))
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.action == "gallery":
        print(f"Audited {len(make_gallery(args.root))} rendered pairs; review still required.")
    else:
        target = args.root / "reviewed.json"
        if target.exists():
            raise ValueError("refusing to overwrite reviewed ledger")
        data = reviewed_ledger(args.root, args.root / "quality_reviews.json")
        write_json(target, data)
        print(f"Exported {len(data['pairs'])} approved pairs for V0.4 sealing.")


if __name__ == "__main__":
    main()
