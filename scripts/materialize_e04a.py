#!/usr/bin/env python3
"""Materialize and fingerprint the frozen Experiment 04A subsets."""

from __future__ import annotations

import argparse
from pathlib import Path

from causetune.e04a_data import materialize_data_efficiency_subsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/incident_diagnosis_training")
    parser.add_argument("--output", default="data/experiment_04a/subsets")
    parser.add_argument("--seed", type=int, default=20260941)
    args = parser.parse_args()
    result = materialize_data_efficiency_subsets(args.source, args.output, seed=args.seed)
    print(result["source_dataset_hash"])
    for subset in result["subsets"]:
        print(f"{subset['fraction']:.2f} {subset['example_count']} {subset['subset_hash']}")


if __name__ == "__main__":
    main()
