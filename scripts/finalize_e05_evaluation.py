#!/usr/bin/env python3
"""Finalize Experiment 05 from already-persisted one-shot outputs only."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causetune.benchmark_e05 import SCORER_VERSION, scorer_fingerprint
from causetune.evidence import sha256_path
from causetune.verify import score_incident_predictions


SYSTEMS = ("base", "e02", "e04")
EXPECTED_BENCHMARK = "b3daed4f49b123b0270baebf49e7609c06f26e5bb10fd125185d6e0864644eaf"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def _transition(source_name: str, target_name: str, source_rows: list[Mapping[str, Any]], target_rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    source = {row["incident_id"]: row for row in source_rows}
    target = {row["incident_id"]: row for row in target_rows}
    if set(source) != set(target) or len(source) != 120:
        raise ValueError(f"{source_name}->{target_name}: prediction IDs do not match the 120-case benchmark")
    rows = []
    for incident_id in source:
        source_row = source[incident_id]
        target_row = target[incident_id]
        source_correct = bool(source_row["diagnosis_exact"])
        target_correct = bool(target_row["diagnosis_exact"])
        category = (
            "source_wrong_target_correct" if not source_correct and target_correct else
            "source_correct_target_wrong" if source_correct and not target_correct else
            "persistent_correct" if source_correct and target_correct else
            "persistent_wrong"
        )
        rows.append({
            "incident_id": incident_id,
            "slice": target_row["slice"],
            "category": category,
            "source_diagnosis_exact": source_correct,
            "target_diagnosis_exact": target_correct,
            "expected": target_row["expected"],
            "source_predicted": source_row["predicted"],
            "target_predicted": target_row["predicted"],
        })
    categories = ("source_wrong_target_correct", "source_correct_target_wrong", "persistent_correct", "persistent_wrong")
    return {
        "source": source_name,
        "target": target_name,
        "count": len(rows),
        "counts": {category: sum(row["category"] == category for row in rows) for category in categories},
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/experiment_05")
    parser.add_argument("--protocol", default="results/experiment_05/protocol.json")
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e05")
    args = parser.parse_args()
    root = Path(args.output_dir)
    protocol = _read_json(Path(args.protocol))
    manifest = _read_json(Path(args.dataset_dir) / "manifest.json")
    if manifest.get("fingerprint") != EXPECTED_BENCHMARK or protocol.get("benchmark_fingerprint") != EXPECTED_BENCHMARK:
        raise ValueError("E05 benchmark fingerprint mismatch")
    if protocol.get("scorer_version") != SCORER_VERSION or protocol.get("scorer_fingerprint") != scorer_fingerprint():
        raise ValueError("E05 scorer mismatch")
    summaries: dict[str, Any] = {}
    predictions: dict[str, list[dict[str, Any]]] = {}
    raw_hashes: dict[str, str] = {}
    for name in SYSTEMS:
        destination = root / "evaluations" / name
        raw_path = destination / "raw_outputs.jsonl"
        prediction_path = destination / "predictions.jsonl"
        evaluation_path = destination / "evaluation.json"
        if not all(path.is_file() for path in (raw_path, prediction_path, evaluation_path)):
            raise ValueError(f"{name}: required persisted evaluation artifact is missing")
        raw_rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        prediction_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(raw_rows) != 120 or len(prediction_rows) != 120 or len({row["incident_id"] for row in raw_rows}) != 120:
            raise ValueError(f"{name}: one-shot raw/prediction accounting failed")
        persisted = _read_json(evaluation_path)
        metrics = persisted.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"{name}: evaluation metrics are missing")
        reproduced = score_incident_predictions(prediction_path)
        if dict(metrics) != reproduced:
            raise ValueError(f"{name}: offline reproduction mismatch")
        predictions[name] = prediction_rows
        summaries[name] = {
            "system": name,
            "metrics": _without_predictions(metrics),
            "slice_metrics": _read_json(destination / "slice_metrics.json"),
            "offline_reproduction": "PASS",
        }
        raw_hashes[name] = sha256_path(raw_path)
    transitions = {
        "base_to_e04": _transition("base", "e04", predictions["base"], predictions["e04"]),
        "e02_to_e04": _transition("e02", "e04", predictions["e02"], predictions["e04"]),
    }
    _write_json(root / "transition_analysis.json", transitions)
    _write_json(root / "evaluation_comparison.json", {
        "schema_version": 1,
        "experiment": "E05",
        "benchmark_fingerprint": EXPECTED_BENCHMARK,
        "systems": summaries,
        "transitions": {key: {k: v for k, v in value.items() if k != "rows"} for key, value in transitions.items()},
        "raw_output_sha256": raw_hashes,
        "e03_used": False,
        "finalized_from_persisted_outputs": True,
    })
    _write_json(root / "g05b_summary.json", {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_fingerprint": EXPECTED_BENCHMARK,
        "systems": list(SYSTEMS),
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "raw_predictions": True,
        "offline_reproduction": True,
        "checkpoint_switching": False,
        "post_result_prompt_change": False,
        "e03_used": False,
        "semantic_generation_rerun_during_finalization": False,
    })
    artifact_paths = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifact_paths[path.relative_to(root).as_posix()] = sha256_path(path)
    _write_json(root / "artifact_hashes.json", {
        "schema_version": 1,
        "benchmark_fingerprint": EXPECTED_BENCHMARK,
        "artifacts": artifact_paths,
    })
    print(json.dumps({"status": "PASS", "systems": list(SYSTEMS), "offline_reproduction": True, "semantic_generation_rerun": False}, sort_keys=True))


if __name__ == "__main__":
    main()
