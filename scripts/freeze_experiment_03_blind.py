#!/usr/bin/env python3
"""Generate and freeze the independent Experiment 03 blind benchmark."""

from __future__ import annotations

import argparse
import json

from causetune.benchmark_blind_v2 import write_blind_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/incident_diagnosis_blind_v2")
    parser.add_argument("--seed", type=int, default=20260912)
    parser.add_argument(
        "--source",
        action="append",
        default=["data/incident_diagnosis_training", "data/incident_diagnosis"],
        help="old train/validation/benchmark source to audit for exact overlap",
    )
    args = parser.parse_args()
    manifest = write_blind_benchmark(args.output, seed=args.seed, contamination_sources=tuple(args.source))
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
