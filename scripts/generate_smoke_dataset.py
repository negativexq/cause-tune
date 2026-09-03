#!/usr/bin/env python3
"""Generate and split the deterministic local 200-example smoke dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fineforge.config import load_config
from fineforge.data.generate import generate_smoke_records
from fineforge.data.schema import validate_records, write_jsonl
from fineforge.data.split import split_ids, split_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_smoke.json")
    parser.add_argument("--output", help="override config.dataset_path")
    args = parser.parse_args()

    config = load_config(args.config)
    records = validate_records(generate_smoke_records(config.seed))
    if len(records) != config.max_examples:
        raise ValueError(
            f"generated {len(records)} records, expected max_examples={config.max_examples}"
        )

    output_path = Path(args.output or config.dataset_path)
    write_jsonl(output_path, records)
    splits = split_records(
        records,
        seed=config.seed,
        train_ratio=config.train_ratio,
        validation_ratio=config.validation_ratio,
        test_ratio=config.test_ratio,
    )
    manifest_path = output_path.with_suffix(".splits.json")
    manifest = {
        "seed": config.seed,
        "dataset_path": str(output_path),
        "counts": {name: len(items) for name, items in splits.items()},
        "example_ids": split_ids(splits),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(config.human_readable())
    print(f"Wrote dataset: {output_path} ({len(records)} records)")
    print(f"Wrote split manifest: {manifest_path}")
    print("Split counts:", manifest["counts"])


if __name__ == "__main__":
    main()

