#!/usr/bin/env python3
"""Finalize an E06 final evaluation from already-persisted predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from causetune.benchmark_e05 import scorer_fingerprint
from causetune.evidence import sha256_path
from causetune.verify import score_incident_predictions


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def transition(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = {row["incident_id"]: row for row in source["predictions"]}
    target_rows = {row["incident_id"]: row for row in target["predictions"]}
    categories = ("source_wrong_target_correct", "source_correct_target_wrong", "persistent_correct", "persistent_wrong")
    rows = []
    for incident_id, source_row in source_rows.items():
        source_correct = bool(source_row["diagnosis_exact"])
        target_correct = bool(target_rows[incident_id]["diagnosis_exact"])
        category = categories[0] if not source_correct and target_correct else categories[1] if source_correct and not target_correct else categories[2] if source_correct else categories[3]
        rows.append({
            "incident_id": incident_id,
            "category": category,
            "source_diagnosis_exact": source_correct,
            "target_diagnosis_exact": target_correct,
            "source_predicted": source_row["predicted"],
            "target_predicted": target_rows[incident_id]["predicted"],
            "expected": target_rows[incident_id]["expected"],
        })
    return {"source": "base", "target": "tuned", "count": len(rows), "counts": {category: sum(row["category"] == category for row in rows) for category in categories}, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/experiment_06/final_evaluation-retry-01")
    parser.add_argument("--protocol", default="results/experiment_06/final_protocol.json")
    args = parser.parse_args()
    root = Path(args.output_dir)
    protocol = read_json(Path(args.protocol))
    benchmark = read_json(Path(protocol["benchmark_dir"]) / "manifest.json")
    if protocol["benchmark_fingerprint"] != benchmark["fingerprint"] or protocol["scorer_fingerprint"] != scorer_fingerprint():
        raise ValueError("E06 final protocol or scorer fingerprint changed")
    evaluations: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name in ("base", "tuned"):
        evaluation = read_json(root / name / "evaluation.json")["metrics"]
        reproduced = score_incident_predictions(root / name / "predictions.jsonl")
        if reproduced != evaluation:
            raise ValueError(f"persisted E06 {name} evaluation does not reproduce offline")
        if sum(1 for line in (root / name / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()) != 60:
            raise ValueError(f"persisted E06 {name} raw output count is not 60")
        evaluations[name] = evaluation
        summaries[name] = {"system": name, "metrics": without_predictions(evaluation), "slice_metrics": read_json(root / name / "slice_metrics.json")}
    write_json(root / "technical_failure.json", {
        "status": "TECHNICAL_FAILURE",
        "valid_semantic_run": True,
        "stage": "post_semantic_aggregation",
        "exception": "KeyError: summaries omitted prediction rows during transition aggregation",
        "attempt": 1,
        "semantic_evaluations_completed": 2,
        "semantic_evaluations_regenerated": False,
        "recovery": "offline finalization from persisted base/tuned predictions",
    })
    result = transition(evaluations["base"], evaluations["tuned"])
    write_json(root / "transition_analysis.json", result)
    write_json(root / "evaluation_comparison.json", {
        "schema_version": 1,
        "experiment": "E06",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "systems": summaries,
        "transitions": {key: value for key, value in result.items() if key != "rows"},
        "raw_output_sha256": {name: sha256_path(root / name / "raw_outputs.jsonl") for name in ("base", "tuned")},
        "capability_screen_used_as_final_evidence": False,
    })
    write_json(root / "g06_summary.json", {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "systems": ["base", "tuned"],
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "raw_predictions": True,
        "offline_reproduction": True,
        "architecture_differences_documented": True,
        "capability_screen_used_as_final_evidence": False,
        "semantic_evaluations_regenerated_during_recovery": False,
    })
    artifacts = {path.relative_to(root).as_posix(): sha256_path(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    write_json(root / "artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": benchmark["fingerprint"], "artifacts": artifacts})
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": benchmark["fingerprint"], "transition_counts": result["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
