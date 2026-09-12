#!/usr/bin/env python3
"""Write predeclared Experiment 04 focused-study contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.focused_study import data_efficiency_study, learning_rate_study, lora_capacity_study, write_study_contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/experiment_04/study_contracts")
    parser.add_argument("--benchmark-manifest", default="data/incident_diagnosis_blind_v2/manifest.json")
    parser.add_argument("--train", default="data/incident_diagnosis_training/train.jsonl")
    parser.add_argument("--validation", default="data/incident_diagnosis_training/validation.jsonl")
    args = parser.parse_args()
    benchmark = json.loads(Path(args.benchmark_manifest).read_text(encoding="utf-8"))
    kwargs = {
        "train_path": args.train,
        "validation_path": args.validation,
        "benchmark_fingerprint": benchmark["fingerprint"],
    }
    contracts = (data_efficiency_study(**kwargs), lora_capacity_study(**kwargs), learning_rate_study(**kwargs))
    output = Path(args.output)
    for contract in contracts:
        write_study_contract(contract, output / f"{contract['study_id']}.json")
    print(json.dumps({"status": "PASS", "studies": [contract["study_id"] for contract in contracts]}, sort_keys=True))


if __name__ == "__main__":
    main()
