"""Seal reviewed, explicitly listed source/target pairs; never guess pair correspondence."""

import argparse
import json
from pathlib import Path

from roomcraft.common import sha256, write_json
from roomcraft.data import local_path
from roomcraft.qwen_data import validate_pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reviewed", type=Path, help="Reviewed pair ledger; same schema, hashes may be omitted"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.parent.resolve() != args.reviewed.parent.resolve():
        raise ValueError("write a new manifest beside the reviewed ledger")
    data = json.loads(args.reviewed.read_text(encoding="utf-8"))
    for row in data["pairs"]:
        row["sha256"] = {
            key: sha256(local_path(args.reviewed.parent, row[key]))
            for key in ("source", "target", "evidence", "alignment_evidence")
        }
    write_json(args.output, data)
    validate_pairs(args.output, require_splits=False)
    print(f"Sealed {len(data['pairs'])} explicitly reviewed pairs. Assign splits next.")


if __name__ == "__main__":
    main()
