#!/usr/bin/env python3
"""Report M4 dataset statistics, optionally including Qwen tokenizer lengths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.data.statistics import dataset_statistics
from causetune.data.validation import read_realistic_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/realistic")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--no-tokenizer", action="store_true")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    splits = read_realistic_splits(dataset_dir)
    tokenizer = None
    if not args.no_tokenizer:
        from causetune.model import load_tokenizer_for_model
        tokenizer = load_tokenizer_for_model(args.model_id)
    report = dataset_statistics(splits, tokenizer)
    path = dataset_dir / "statistics.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(path), "total_examples": report["global"]["total_examples"]}, indent=2))


if __name__ == "__main__":
    main()
