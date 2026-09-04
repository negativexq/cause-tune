#!/usr/bin/env python3
"""Print a compact stratified human-inspection sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.incident_benchmark import read_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis")
    parser.add_argument("--output", default="data/incident_diagnosis/inspection_sample.jsonl")
    args = parser.parse_args()
    inputs, ground_truth, _ = read_benchmark(args.dataset_dir)
    truth_by_id = {row["incident_id"]: row for row in ground_truth}
    selected = []
    for split, count in (("standard", 2), ("hard", 3), ("transfer", 2)):
        selected.extend(inputs[split][:count])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as handle:
        for incident in selected:
            handle.write(json.dumps({
                **incident,
                "ground_truth": truth_by_id[incident["incident_id"]],
            }, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
    print(json.dumps({"status": "pass", "sample_count": len(selected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
