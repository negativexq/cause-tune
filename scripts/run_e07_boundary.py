#!/usr/bin/env python3
"""Run the frozen, evaluation-only Experiment 07 boundary challenge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import torch

from causetune.evidence import sha256_path
from causetune.incident_benchmark import packet_evidence_ids
from causetune.incident_taxonomy import ACTION_SET, FAILURE_FAMILY_SET, SLICES
from causetune.model import load_adapter, load_frozen_quantized_base, load_tokenizer_for_model
try:
    from run_experiment_05 import _generate, _write_json, _write_jsonl
except ModuleNotFoundError:
    from scripts.run_experiment_05 import _generate, _write_json, _write_jsonl


EXPECTED_MODEL = "Qwen/Qwen3-4B"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
SYSTEMS = ("base", "e02", "e04")
STATUSES = {"diagnose", "insufficient_evidence", "ambiguous_evidence", "out_of_taxonomy"}
E07_KEYS = {"status", "culprit_service", "failure_mode", "recommended_action", "evidence_ids", "required_evidence"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _parse_output(raw: str, present_components: list[str], available: set[str]) -> tuple[dict[str, Any] | None, str, bool, str | None]:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid JSON", False, None
    if not isinstance(value, dict):
        return None, "invalid schema", True, None
    if set(value) == E07_KEYS:
        status = value.get("status")
        if status not in STATUSES:
            return None, "unknown status", True, None
        required = value.get("required_evidence")
        evidence = value.get("evidence_ids")
        if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
            return None, "invalid required_evidence", True, None
        if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            return None, "invalid evidence_ids", True, None
        if len(evidence) != len(set(evidence)) or not set(evidence).issubset(available):
            return None, "invalid evidence reference", True, None
        if status == "diagnose":
            if not isinstance(value.get("culprit_service"), str) or value["culprit_service"] not in set(present_components):
                return None, "unknown culprit", True, None
            if value.get("failure_mode") not in FAILURE_FAMILY_SET:
                return None, "unknown failure mode", True, None
            if value.get("recommended_action") not in ACTION_SET:
                return None, "unknown action", True, None
        elif value.get("culprit_service") is not None or value.get("failure_mode") is not None or value.get("recommended_action") is not None or evidence:
            return None, "non-diagnosis contains diagnosis fields", True, None
        return dict(value), "valid E07 JSON", True, "e07"
    legacy_keys = {"culprit_service", "failure_mode", "recommended_action", "evidence_ids"}
    if set(value) == legacy_keys:
        evidence = value.get("evidence_ids")
        if isinstance(evidence, list) and all(isinstance(item, str) for item in evidence) and len(evidence) == len(set(evidence)) and set(evidence).issubset(available):
            return {"status": "diagnose", **value, "required_evidence": []}, "legacy diagnosis contract", True, "legacy"
    return None, "unexpected keys", True, None


def _score(records: list[dict[str, Any]], truths: Mapping[str, Mapping[str, Any]], raw: Mapping[str, str]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        incident_id = record["incident_id"]
        truth = truths[incident_id]
        parsed, category, json_valid, contract_mode = _parse_output(raw[incident_id], record["metadata"]["present_components"], packet_evidence_ids(record["incident_packet"]))
        expected_status = truth["expected_status"]
        predicted_status = parsed.get("status") if parsed else None
        expected_diagnosis = expected_status == "diagnose"
        predicted_diagnosis = predicted_status == "diagnose"
        semantic_correct = bool(
            parsed
            and expected_diagnosis
            and predicted_status == "diagnose"
            and parsed.get("culprit_service") == truth.get("culprit_service")
            and parsed.get("failure_mode") == truth.get("failure_mode")
            and parsed.get("recommended_action") == truth.get("recommended_action")
        )
        rows.append({
            "incident_id": incident_id,
            "boundary_category": record["metadata"]["boundary_category"],
            "expected_status": expected_status,
            "predicted_status": predicted_status,
            "contract_mode": contract_mode,
            "parse_category": category,
            "json_valid": json_valid,
            "schema_valid": parsed is not None and contract_mode == "e07",
            "predicted_abstention": predicted_status in {"insufficient_evidence", "ambiguous_evidence", "out_of_taxonomy"},
            "expected_abstention": not expected_diagnosis,
            "status_correct": bool(predicted_status == expected_status),
            "sufficient_case_correct": semantic_correct,
            "false_confident_diagnosis": bool(not expected_diagnosis and predicted_diagnosis),
            "false_abstention": bool(expected_diagnosis and not predicted_diagnosis),
            "expected": {
                "status": expected_status,
                "culprit_service": truth.get("culprit_service"),
                "failure_mode": truth.get("failure_mode"),
                "recommended_action": truth.get("recommended_action"),
            },
            "predicted": parsed,
            "raw_output": raw[incident_id],
        })
    total = len(rows)
    sufficient = [row for row in rows if row["expected_status"] == "diagnose"]
    insufficient = [row for row in rows if row["expected_status"] != "diagnose"]
    predicted_abstain = sum(row["predicted_abstention"] for row in rows)
    true_abstain = sum(row["expected_abstention"] for row in rows)
    abstention_tp = sum(row["predicted_abstention"] and row["expected_abstention"] for row in rows)
    precision = abstention_tp / predicted_abstain if predicted_abstain else 0.0
    recall = abstention_tp / true_abstain if true_abstain else 0.0
    by_category: dict[str, dict[str, Any]] = {}
    for category in sorted({row["boundary_category"] for row in rows}):
        selected = [row for row in rows if row["boundary_category"] == category]
        by_category[category] = {
            "count": len(selected),
            "status_accuracy": sum(row["status_correct"] for row in selected) / len(selected),
            "schema_valid_rate": sum(row["schema_valid"] for row in selected) / len(selected),
            "false_confident_diagnosis_count": sum(row["false_confident_diagnosis"] for row in selected),
            "false_abstention_count": sum(row["false_abstention"] for row in selected),
        }
    return {
        "schema_version": 1,
        "count": total,
        "sufficient_case_count": len(sufficient),
        "insufficient_or_ambiguous_case_count": len(insufficient),
        "abstention_precision": precision,
        "abstention_recall": recall,
        "abstention_f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_confident_diagnosis_rate": sum(row["false_confident_diagnosis"] for row in insufficient) / len(insufficient) if insufficient else 0.0,
        "false_confident_diagnosis_count": sum(row["false_confident_diagnosis"] for row in insufficient),
        "false_abstention_rate": sum(row["false_abstention"] for row in sufficient) / len(sufficient) if sufficient else 0.0,
        "false_abstention_count": sum(row["false_abstention"] for row in sufficient),
        "normal_sufficient_case_accuracy": sum(row["sufficient_case_correct"] for row in sufficient) / len(sufficient) if sufficient else 0.0,
        "status_accuracy": sum(row["status_correct"] for row in rows) / total if total else 0.0,
        "json_valid_rate": sum(row["json_valid"] for row in rows) / total if total else 0.0,
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / total if total else 0.0,
        "parse_categories": dict(Counter(row["parse_category"] for row in rows)),
        "by_boundary_category": by_category,
        "predictions": rows,
    }


def _without_predictions(value: Mapping[str, Any]) -> dict[str, Any]:
    return {key: item for key, item in value.items() if key != "predictions"}


def _system_model(system: Mapping[str, Any], quantization: Mapping[str, Any]) -> Any:
    model = load_frozen_quantized_base(system["model_id"], revision=system["revision"], trust_remote_code=False, **quantization)
    if system.get("adapter"):
        model = load_adapter(model, system["adapter"])
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e07_boundary")
    parser.add_argument("--protocol", default="results/experiment_07/protocol.json")
    parser.add_argument("--e05-protocol", default="results/experiment_05/protocol.json")
    parser.add_argument("--output-dir", default="results/experiment_07/evaluations")
    args = parser.parse_args()
    dataset = Path(args.dataset_dir)
    protocol_path = Path(args.protocol)
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to reuse immutable E07 evaluation directory: {root}")
    protocol = _read_json(protocol_path)
    manifest = _read_json(dataset / "manifest.json")
    prompt_path = Path(protocol["prompt_path"])
    if protocol.get("benchmark_fingerprint") != manifest.get("fingerprint") or protocol.get("case_count") != 60:
        raise ValueError("E07 benchmark contract mismatch")
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != protocol.get("prompt_sha256"):
        raise ValueError("E07 prompt hash mismatch")
    if protocol.get("decoding") != {"max_new_tokens": 128, "batch_size": 4, "do_sample": False}:
        raise ValueError("E07 decoding contract mismatch")
    e05 = _read_json(Path(args.e05_protocol))
    source_systems = {item["name"]: item for item in e05["systems"]}
    if list(protocol.get("systems")) != list(SYSTEMS):
        raise ValueError("E07 system order mismatch")
    for name in SYSTEMS:
        system = source_systems[name]
        if system["model_id"] != EXPECTED_MODEL or system["revision"] != EXPECTED_REVISION:
            raise ValueError(f"{name}: model identity mismatch")
    records = [record for split in SLICES for record in _read_jsonl(dataset / f"{split}.jsonl")]
    truths = {row["incident_id"]: row for row in _read_jsonl(dataset / "ground_truth.jsonl")}
    if len(records) != 60 or len(truths) != 60:
        raise ValueError("E07 requires exactly 60 cases")
    quantization = _read_json(Path("configs/incident_diagnosis_eval.json"))["quantization"]
    root.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, Any] = {}
    raw_hashes: dict[str, str] = {}
    try:
        for name in SYSTEMS:
            system = source_systems[name]
            destination = root / name
            print(f"loading={name}", flush=True)
            tokenizer = load_tokenizer_for_model(system["model_id"], revision=system["revision"], trust_remote_code=False)
            model = _system_model(system, quantization)
            raw = _generate(model, tokenizer, records, prompt_path.read_text(encoding="utf-8"), max_new_tokens=128, batch_size=4)
            _write_jsonl(destination / "raw_outputs.jsonl", [{"incident_id": row["incident_id"], "raw_output": raw[row["incident_id"]]} for row in records])
            raw_hashes[name] = sha256_path(destination / "raw_outputs.jsonl")
            evaluation = _score(records, truths, raw)
            _write_jsonl(destination / "predictions.jsonl", evaluation["predictions"])
            _write_json(destination / "evaluation.json", {"schema_version": 1, "system": name, "metrics": _without_predictions(evaluation)})
            reproduced = _score(records, truths, {row["incident_id"]: row["raw_output"] for row in _read_jsonl(destination / "raw_outputs.jsonl")})
            _write_json(destination / "offline_reproduction.json", {"status": "PASS", "metrics": _without_predictions(reproduced)})
            if _without_predictions(reproduced) != _without_predictions(evaluation):
                raise ValueError(f"E07 offline reproduction mismatch for {name}")
            summaries[name] = {"system": name, "metrics": _without_predictions(evaluation)}
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as exc:
        _write_json(root / "technical_failure.json", {
            "status": "TECHNICAL_FAILURE",
            "valid_semantic_run": False,
            "stage": "model_load_or_generation_or_persistence",
            "exception": f"{type(exc).__name__}: {exc}",
            "git_sha": _git_sha(),
            "benchmark_fingerprint": manifest["fingerprint"],
            "protocol_path": str(protocol_path),
        })
        raise
    _write_json(root / "evaluation_comparison.json", {
        "schema_version": 1,
        "experiment": "E07",
        "benchmark_fingerprint": manifest["fingerprint"],
        "systems": summaries,
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "e05_regression_untouched": True,
        "raw_output_sha256": raw_hashes,
    })
    _write_json(root / "g07_summary.json", {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_fingerprint": manifest["fingerprint"],
        "systems": list(SYSTEMS),
        "one_shot_per_system": True,
        "same_frozen_benchmark": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "excluded_from_training": True,
        "raw_predictions": True,
        "offline_reproduction": True,
        "false_confidence_analysis": True,
        "llm_judge_required": False,
    })
    artifacts = {path.relative_to(root).as_posix(): sha256_path(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    _write_json(root / "artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": manifest["fingerprint"], "artifacts": artifacts})
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": manifest["fingerprint"], "systems": list(summaries)}, sort_keys=True))


if __name__ == "__main__":
    main()
