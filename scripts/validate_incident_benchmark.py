#!/usr/bin/env python3
"""Validate the frozen Experiment 02A benchmark without loading model weights."""

from __future__ import annotations

import argparse
import json

from causetune.incident_benchmark import read_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis")
    args = parser.parse_args()
    inputs, ground_truth, manifest = read_benchmark(args.dataset_dir)
    print(json.dumps({
        "status": "pass",
        "incident_count": sum(len(rows) for rows in inputs.values()),
        "ground_truth_count": len(ground_truth),
        "manifest": manifest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
