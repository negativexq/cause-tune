#!/usr/bin/env python3
"""Generate the frozen Experiment 02A incident benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.incident_benchmark import DEFAULT_SEED, write_benchmark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/incident_diagnosis")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    manifest = write_benchmark(args.output_dir, args.seed)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
