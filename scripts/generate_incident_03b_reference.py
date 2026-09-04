#!/usr/bin/env python3
"""Generate and audit the small CPU-only Experiment 03B reference corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.incident_telemetry.reference import REFERENCE_SEED, write_reference_corpus
from causetune.incident_telemetry.models import (
    OUTPUT_CONTRACT_VERSION,
    SCENARIO_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
)
from causetune.incident_telemetry.ontology import ONTOLOGY_VERSION
from causetune.incident_telemetry.splits import SPLIT_MANIFEST_VERSION


def _load_frozen_contract(path: Path) -> dict[str, object]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "experiment": "03A",
        "status": "CONTRACT_FROZEN",
        "ontology_version": ONTOLOGY_VERSION,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "split_manifest_version": SPLIT_MANIFEST_VERSION,
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    policy = contract.get("split_policy", {})
    if mismatches or not isinstance(policy, dict) or policy.get("split_before_rendering") is not True:
        raise ValueError(f"frozen 03A contract mismatch: {mismatches or 'split policy'}")
    forbidden = ("provider_calls_allowed", "dataset_generation_allowed", "training_allowed", "benchmark_evaluation_allowed")
    if any(contract.get(key) is not False for key in forbidden):
        raise ValueError("03A frozen contract permits a forbidden operation")
    return contract


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/incident_telemetry_03b")
    parser.add_argument("--seed", type=int, default=REFERENCE_SEED)
    parser.add_argument("--config", type=Path, default=Path("configs/incident_telemetry_03a.json"))
    args = parser.parse_args()
    _load_frozen_contract(args.config)
    summary = write_reference_corpus(args.output_dir, args.seed)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
