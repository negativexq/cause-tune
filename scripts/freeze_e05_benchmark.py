#!/usr/bin/env python3
"""Generate, audit, and freeze the fresh Experiment 05 challenge."""

from __future__ import annotations

import argparse
import json

from causetune.benchmark_e05 import DEFAULT_SEED, write_e05_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/incident_diagnosis_e05")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--source", action="append", dest="sources", default=[])
    args = parser.parse_args()
    sources = tuple(args.sources) or (
        "data/incident_diagnosis_training",
        "data/incident_diagnosis",
        "data/incident_diagnosis_blind_v2",
    )
    manifest = write_e05_benchmark(args.output, seed=args.seed, contamination_sources=sources)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
