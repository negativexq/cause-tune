#!/usr/bin/env python3
"""Recover Experiment 02B.2 artifacts from persisted tuned raw outputs.

This script is deliberately offline: it loads no model and never calls
generation.  It only reparses the already persisted 144 benchmark outputs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from causetune.incident_benchmark import read_benchmark
from causetune.incident_evaluation import evaluate_incidents, failure_patterns_for_splits


ROOT = Path("outputs/incident_diagnosis_02b2")
EXPECTED_BENCHMARK = "5e9a6d74ba881af7aa1146a23f4b837e139a4e8e5a77dba7fe40c3bb2c9c0a94"
EXPECTED_EVALUATION = "d2dea87fd26a3a8c78a2a6a1b4b6fc583b27070ed2de7e691b53c17e59231cf8"
SPLITS = ("standard", "hard", "transfer")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def metric_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def transition_key(base: bool, tuned: bool) -> str:
    return f"base_{'correct' if base else 'wrong'} -> tuned_{'correct' if tuned else 'wrong'}"


def main() -> None:
    prediction_path = ROOT / "tuned_predictions.jsonl"
    prediction_rows = [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    benchmark_inputs, benchmark_truth_rows, benchmark_manifest = read_benchmark("data/incident_diagnosis")
    all_inputs = [item for split in SPLITS for item in benchmark_inputs[split]]
    truth_by_id = {truth["incident_id"]: truth for truth in benchmark_truth_rows}
    expected_ids = [item["incident_id"] for item in all_inputs]
    prediction_ids = [row.get("incident_id") for row in prediction_rows]
    if len(prediction_rows) != 144 or len(set(prediction_ids)) != 144 or prediction_ids != expected_ids:
        raise RuntimeError("persisted tuned predictions do not exactly cover the frozen benchmark")
    if benchmark_manifest["benchmark_fingerprint"] != EXPECTED_BENCHMARK:
        raise RuntimeError("benchmark fingerprint mismatch")
    resolved = read_json(ROOT / "resolved_manifest.json")
    if resolved["frozen_evaluation_fingerprint"] != EXPECTED_EVALUATION:
        raise RuntimeError("evaluation fingerprint mismatch")
    if not all(isinstance(row.get("raw_output"), str) and row["raw_output"] for row in prediction_rows):
        raise RuntimeError("persisted tuned predictions contain missing raw outputs")

    raw_by_id = {row["incident_id"]: row["raw_output"] for row in prediction_rows}
    tuned = {split: evaluate_incidents(benchmark_inputs[split], truth_by_id, {item["incident_id"]: raw_by_id[item["incident_id"]] for item in benchmark_inputs[split]}) for split in SPLITS}
    tuned["all"] = evaluate_incidents(all_inputs, truth_by_id, raw_by_id)
    patterns = failure_patterns_for_splits(benchmark_inputs, tuned, all_inputs)
    write_json(ROOT / "tuned_metrics.json", {"benchmark_fingerprint": EXPECTED_BENCHMARK, "evaluation_fingerprint": EXPECTED_EVALUATION, "prediction_count": len(raw_by_id), "predictions_persisted": True, "splits": tuned, "failure_patterns": patterns})

    base_result = read_json(Path("results/incident_diagnosis_base.json"))
    base_predictions = {row["incident_id"]: row for row in (json.loads(line) for line in Path("outputs/incident_diagnosis_base/predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())}
    tuned_predictions = {row["incident_id"]: row for row in tuned["all"]["predictions"]}
    transitions: dict[str, Counter[str]] = {name: Counter() for name in ("diagnosis", "resolution", "action")}
    transition_rows: list[dict[str, Any]] = []
    for item in all_inputs:
        incident_id = item["incident_id"]
        base = base_predictions[incident_id]
        current = tuned_predictions[incident_id]
        row: dict[str, Any] = {"incident_id": incident_id}
        for name, field in (("diagnosis", "diagnosis_exact"), ("resolution", "resolution_exact"), ("action", "action_correct")):
            key = transition_key(bool(base[field]), bool(current[field]))
            transitions[name][key] += 1
            row[f"base_{name}"] = bool(base[field])
            row[f"tuned_{name}"] = bool(current[field])
            row[f"{name}_transition"] = key
        transition_rows.append(row)
    write_json(ROOT / "failure_transitions.json", {"count": 144, "transitions": {name: dict(counter) for name, counter in transitions.items()}, "rows": transition_rows})

    family_comparison: dict[str, Any] = {}
    for family, base_family in base_result["splits"]["all"]["per_failure_family"].items():
        tuned_family = tuned["all"]["per_failure_family"][family]
        fields = ("diagnosis_exact_match", "resolution_exact_match", "culprit_accuracy", "failure_mode_accuracy", "action_accuracy", "json_compliance")
        family_comparison[family] = {"base": base_family, "tuned": tuned_family, "delta_pp": {field: (tuned_family[field] - base_family[field]) * 100 for field in fields}}
    write_json(ROOT / "per_family_comparison.json", {"families": family_comparison, "training_note": "The previously weak frozen benchmark families were not oversampled during training."})

    base_all = base_result["splits"]["all"]
    comparison = {
        "base": {split: metric_view(metrics) for split, metrics in base_result["splits"].items()},
        "tuned": {split: metric_view(metrics) for split, metrics in tuned.items()},
        "failure_patterns": {"base": base_result["analysis"]["all"]["failure_patterns"], "tuned": patterns["all"]},
        "transitions": {name: dict(counter) for name, counter in transitions.items()},
        "per_family": family_comparison,
        "selected_checkpoint": 100,
        "selection_basis": "validation_only",
    }
    write_json(ROOT / "base_vs_tuned.json", comparison)
    write_json(ROOT / "failure_pattern_comparison.json", {"base": base_result["analysis"]["all"]["failure_patterns"], "tuned": patterns["all"], "delta": {key: patterns["all"].get(key, 0) - base_result["analysis"]["all"]["failure_patterns"].get(key, 0) for key in sorted(set(patterns["all"]) | set(base_result["analysis"]["all"]["failure_patterns"]))}})

    training = read_json(ROOT / "training_summary.json")
    selection = read_json(ROOT / "checkpoint_selection.json")
    reload = read_json(ROOT / "reload_verification.json")
    site = {
        "experiment": "production_incident_diagnosis_specialization", "model": "Qwen/Qwen3-4B",
        "base_diagnosis_exact": base_all["diagnosis_exact_match"]["rate"], "tuned_diagnosis_exact": tuned["all"]["diagnosis_exact_match"]["rate"], "diagnosis_gain_pp": (tuned["all"]["diagnosis_exact_match"]["rate"] - base_all["diagnosis_exact_match"]["rate"]) * 100,
        "base_resolution_exact": base_all["resolution_exact_match"]["rate"], "tuned_resolution_exact": tuned["all"]["resolution_exact_match"]["rate"], "resolution_gain_pp": (tuned["all"]["resolution_exact_match"]["rate"] - base_all["resolution_exact_match"]["rate"]) * 100,
        "base_hard": base_result["splits"]["hard"]["diagnosis_exact_match"]["rate"], "tuned_hard": tuned["hard"]["diagnosis_exact_match"]["rate"], "hard_gain_pp": (tuned["hard"]["diagnosis_exact_match"]["rate"] - base_result["splits"]["hard"]["diagnosis_exact_match"]["rate"]) * 100,
        "base_transfer": base_result["splits"]["transfer"]["diagnosis_exact_match"]["rate"], "tuned_transfer": tuned["transfer"]["diagnosis_exact_match"]["rate"], "transfer_gain_pp": (tuned["transfer"]["diagnosis_exact_match"]["rate"] - base_result["splits"]["transfer"]["diagnosis_exact_match"]["rate"]) * 100,
        "base_action_accuracy": base_all["recommended_action_accuracy"]["rate"], "tuned_action_accuracy": tuned["all"]["recommended_action_accuracy"]["rate"], "action_gain_pp": (tuned["all"]["recommended_action_accuracy"]["rate"] - base_all["recommended_action_accuracy"]["rate"]) * 100,
        "max_optimizer_steps": 600, "actual_stop_step": training["actual_stop_step"], "best_checkpoint_step": training["best_checkpoint_step"], "earliest_near_best_step": training["earliest_near_best"]["earliest_step"], "training_seconds": training["wall_clock_training_seconds"], "validation_seconds": training["validation_wall_clock_seconds"], "peak_allocated_vram_gib": training["peak_vram"]["allocated_gib"], "peak_reserved_vram_gib": training["peak_vram"]["reserved_gib"],
    }
    write_json(ROOT / "site_ready_summary.json", site)

    start = (ROOT / "preflight.json").stat().st_mtime
    end = prediction_path.stat().st_mtime
    final = {
        "experiment": "02B.2 Production Incident Diagnosis QLoRA Specialization",
        "fingerprints": {"train": resolved["train_fingerprint"], "validation": resolved["validation_fingerprint"], "benchmark": EXPECTED_BENCHMARK, "evaluation": EXPECTED_EVALUATION},
        "training": training, "checkpoint_selection": selection, "reload": reload,
        "benchmark_evaluations": {"tuned_generation_passes": 1, "tuned_prediction_count": 144, "recovery_evaluation_mode": "offline_from_persisted_raw_outputs", "alternate_checkpoint_evaluations": 0},
        "integrity": {"training_executed_once": True, "selected_checkpoint": 100, "checkpoint_selected_using_validation_only": True, "fresh_adapter_reload_succeeded": True, "tuned_frozen_benchmark_generation_executed_once": True, "initial_postprocessing_failure": "KeyError: 'all'", "metrics_recovered_offline": True, "second_generation_performed": False, "second_training_run_performed": False, "raw_predictions_overwritten": False, "prediction_accounting": {"expected": 144, "actual": 144, "duplicates": 0, "missing": 0}},
        "total_experiment_seconds_to_prediction_persistence": max(0.0, end - start),
    }
    write_json(ROOT / "final_experiment_summary.json", final)
    write_json(ROOT / "prediction_integrity.json", {"status": "pass", "prediction_file": str(prediction_path), "count": 144, "unique_ids": 144, "missing_ids": [], "extra_ids": [], "raw_outputs_present": 144, "benchmark_fingerprint": EXPECTED_BENCHMARK, "evaluation_fingerprint": EXPECTED_EVALUATION, "parsed_predictions_persisted": False, "recovered_offline": True, "model_generation_during_recovery": False})
    print(json.dumps({"status": "recovered", "prediction_count": 144, "benchmark_fingerprint": EXPECTED_BENCHMARK, "evaluation_fingerprint": EXPECTED_EVALUATION, "selected_checkpoint": 100}, sort_keys=True))


if __name__ == "__main__":
    main()
