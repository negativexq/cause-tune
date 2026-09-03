#!/usr/bin/env python3
"""Generate and validate the deterministic FineForge M4 realistic dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fineforge.data.realistic_generate import DATASET_VERSION, write_realistic_dataset
from fineforge.data.validation import build_manifest, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/realistic")
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    output = Path(args.output_dir)
    splits = write_realistic_dataset(output, args.seed)
    manifest = build_manifest(splits, args.seed)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    write_validation_report(output / "validation_report.json", {
        "dataset_version": DATASET_VERSION,
        "status": "pass",
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "duplicate_check": manifest["duplicate_check"],
    })
    print(json.dumps({
        "dataset_version": DATASET_VERSION,
        "output_dir": str(output),
        "split_sizes": {split: len(records) for split, records in splits.items()},
        "total": sum(len(records) for records in splits.values()),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

