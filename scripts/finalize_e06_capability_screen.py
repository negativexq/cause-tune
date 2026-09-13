#!/usr/bin/env python3
"""Finalize the persisted E06 capability-screen retry without model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from causetune.evidence import sha256_path
from causetune.verify import score_incident_predictions


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/experiment_06/capability_gap-retry-01")
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e06_screen")
    parser.add_argument("--compatibility-record", default="results/experiment_06/compatibility_recovery.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    dataset = Path(args.dataset_dir)
    evaluation = _read_json(output / "evaluation.json")
    predictions = output / "predictions.jsonl"
    raw = output / "raw_outputs.jsonl"
    reproduced = score_incident_predictions(predictions)
    if reproduced != evaluation["metrics"]:
        raise ValueError("E06 capability screen retry does not reproduce offline")
    if sum(1 for line in raw.read_text(encoding="utf-8").splitlines() if line.strip()) != 48:
        raise ValueError("E06 capability screen retry raw output count is not 48")
    manifest = _read_json(dataset / "manifest.json")
    _write_json(output / "retry_provenance.json", {
        "schema_version": 1,
        "status": "PASS",
        "attempt": 1,
        "previous_attempt": "results/experiment_06/capability_gap/technical_failure.json",
        "compatibility_record": args.compatibility_record,
        "benchmark_fingerprint": manifest["fingerprint"],
        "semantic_generation_count": 1,
        "raw_outputs_persisted_before_scoring": True,
        "offline_reproduction": "PASS",
        "model_loading": "native Transformers with trust_remote_code=false",
    })
    artifacts = {
        path.relative_to(output).as_posix(): sha256_path(path)
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    _write_json(output / "artifact_hashes.json", {
        "schema_version": 1,
        "benchmark_fingerprint": manifest["fingerprint"],
        "artifacts": artifacts,
    })
    print(json.dumps({"status": "PASS", "attempt": 1, "offline_reproduction": "PASS", "semantic_generation_count": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
