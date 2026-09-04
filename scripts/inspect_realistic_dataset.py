#!/usr/bin/env python3
"""Write a compact stratified human-inspection artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.data.schema import INTENTS
from causetune.data.validation import read_realistic_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/realistic")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    splits = read_realistic_splits(dataset_dir)
    selected: list[dict] = []
    for intent in INTENTS:
        for difficulty in ("easy", "hard"):
            candidates = [
                row
                for split in ("train", "hard_test")
                for row in splits[split]
                if row["expected_response"]["intent"] == intent and row["difficulty"] == difficulty
            ]
            selected.extend(candidates[:2])
    selected.extend(splits["ood_test"][:8])
    selected.extend([row for row in splits["hard_test"] if "multi_issue" in row["phenomena"]][:10])
    selected.sort(key=lambda row: row["example_id"])
    out_jsonl = dataset_dir / "inspection_sample.jsonl"
    with out_jsonl.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps({
                "example_id": row["example_id"],
                "split": row["split"],
                "intent": row["expected_response"]["intent"],
                "difficulty": row["difficulty"],
                "phenomena": row["phenomena"],
                "confusable_with": row["confusable_with"],
                "label_rule": row["label_rule"],
                "scenario_family": row["scenario_family"],
                "user": row["messages"][0]["content"],
                "assistant": row["messages"][1]["content"],
            }, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(out_jsonl), "examples": len(selected)}, indent=2))


if __name__ == "__main__":
    main()
