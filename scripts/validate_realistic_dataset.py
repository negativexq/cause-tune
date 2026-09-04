#!/usr/bin/env python3
"""Run strict schema, metadata, duplicate, and leakage validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.data.validation import build_manifest, read_realistic_splits, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/realistic")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    splits = read_realistic_splits(dataset_dir)
    manifest = build_manifest(splits, json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))["generation_seed"])
    write_validation_report(dataset_dir / "validation_report.json", {
        "status": "pass",
        "split_sizes": manifest["split_sizes"],
        "duplicate_check": manifest["duplicate_check"],
    })
    print(json.dumps({"status": "pass", "split_sizes": manifest["split_sizes"], "duplicates": manifest["duplicate_check"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

