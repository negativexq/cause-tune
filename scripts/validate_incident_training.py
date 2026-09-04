"""Validate frozen 02B.1 files without loading a tokenizer or model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.incident_contract import contract_fingerprint
from causetune.incident_training import (
    contamination_report,
    training_fingerprint,
    validate_training_manifest,
    validate_training_split,
)


def _read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_training")
    parser.add_argument("--benchmark-dir", default="data/incident_diagnosis")
    parser.add_argument("--evaluation-config", default="configs/incident_diagnosis_eval.json")
    args = parser.parse_args()
    directory = Path(args.dataset_dir)
    train = _read(directory / "train.jsonl")
    validation = _read(directory / "validation.jsonl")
    train_truth = _read(directory / "ground_truth_train.jsonl")
    validation_truth = _read(directory / "ground_truth_validation.jsonl")
    train_report = validate_training_split(train, train_truth, expected_count=2400, expected_per_family=200)
    validation_report = validate_training_split(validation, validation_truth, expected_count=288, expected_per_family=24)
    contamination = contamination_report(train, validation, args.benchmark_dir)
    if contamination["status"] != "pass":
        raise ValueError(f"contamination checks failed: {contamination}")
    eval_config = json.loads(Path(args.evaluation_config).read_text(encoding="utf-8"))
    resolved = json.loads((directory / "resolved_training_manifest.json").read_text(encoding="utf-8"))
    validate_training_manifest(resolved, expected_benchmark_fingerprint=contamination["benchmark_fingerprint"], expected_evaluation_fingerprint=contract_fingerprint(eval_config))
    if resolved["train_fingerprint"] != training_fingerprint(train, train_truth) or resolved["validation_fingerprint"] != training_fingerprint(validation, validation_truth):
        raise ValueError("dataset fingerprint mismatch")
    print(json.dumps({"status": "pass", "train": train_report, "validation": validation_report, "contamination": contamination}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
