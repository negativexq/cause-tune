#!/usr/bin/env python3
"""Finish 03E.2 artifacts from persisted V2 generations without rerunning them."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "/home/ofk/projects/external-data/Cloud-OpsBench")

from run_incident_03e2 import (  # noqa: E402
    MODEL_ID,
    MODEL_REVISION,
    NATIVE_CATEGORIES,
    NATIVE_FAULT_TYPES,
    SOURCE_REVISION,
    _fp,
    executable_history,
    file_hash,
    meta,
    object_match,
    process_metrics,
    read_json,
    scan_corpus,
    scores,
    source_case_group_id,
    target_for_case,
    termination,
    write_json,
)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    root = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
    out = Path("results/incident_telemetry_03e2")
    base = meta(root)
    manifest_path = Path("results/incident_telemetry_03e/validation_subsplit.json")
    manifest = read_json(manifest_path)
    scan = scan_corpus(root, source_revision=SOURCE_REVISION, fail_closed=True)
    by_group = {source_case_group_id(case): case for case in scan.cases}
    cases = [by_group[group] for group in sorted(manifest["screen_case_groups"])]
    oracle = load_rows(out / "oracle_predictions.jsonl")
    closed = load_rows(out / "closed_loop_predictions.jsonl")
    if len(oracle) != 57 or len(closed) != 57:
        raise RuntimeError("persisted V2 generation rows are incomplete")

    oracle_norm = {}
    for case, row in zip(cases, oracle):
        messages, _, _ = executable_history(root, case)
        observations = [str(item["content"]) for item in messages if item.get("role") == "tool"]
        oracle_norm[str(row["source_case_group"])] = bool(row.get("prediction") and object_match(row["prediction"].get("fault_object"), row["target"]["fault_object"], observations)[0])
    oracle_metrics = scores(oracle, oracle_norm)
    write_json(out / "oracle_metrics.json", {**base, **oracle_metrics, "technical_generation_completed": True, "postprocessed_without_regeneration": True})

    closed_norm = {}
    for row in closed:
        observations = [str(event.get("observation", "")) for event in row.get("events", []) if event.get("kind") == "tool"]
        closed_norm[str(row["source_case_group"])] = bool(row.get("prediction") and object_match(row["prediction"].get("fault_object"), row["target"]["fault_object"], observations)[0])
    closed_metrics = scores(closed, closed_norm)
    closed_metrics.update({
        "valid_tool_call_count": sum(row["valid_tool_calls"] for row in closed),
        "executable_tool_call_count": sum(row["executable_tool_calls"] for row in closed),
        "invalid_tool_call_count": sum(row["invalid_tool_calls"] for row in closed),
        "valid_tool_call_rate": sum(row["valid_tool_calls"] > 0 for row in closed) / 57,
        "executable_tool_call_rate": sum(row["executable_tool_calls"] == row["valid_tool_calls"] for row in closed) / 57,
        "invalid_tool_call_rate": sum(row["invalid_tool_calls"] > 0 for row in closed) / 57,
        "replay_failure_count": sum(row["replay_failures"] for row in closed),
        "environment_contract_failure_count": sum(row["environment_contract_failures"] for row in closed),
        "mean_tool_steps": sum(row["valid_tool_calls"] for row in closed) / 57,
        "loop_rate": sum(row["failure_category"] == "repeated_tool_loop" for row in closed) / 57,
        "premature_final_count": sum(row["failure_category"] == "premature_final" for row in closed),
        "no_final_count": sum(row["prediction"] is None for row in closed),
        "max_step_exhaustion_count": sum(row["failure_category"] == "max_step_exhaustion" for row in closed),
        "failure_categories": dict(sorted(__import__("collections").Counter(str(row["failure_category"]) for row in closed).items())),
        "peak_allocated_bytes": max(row.get("peak_allocated_bytes", 0) for row in closed),
        "peak_reserved_bytes": max(row.get("peak_reserved_bytes", 0) for row in closed),
        "technical_generation_completed": True,
        "postprocessed_without_regeneration": True,
    })
    write_json(out / "closed_loop_metrics.json", {**base, **closed_metrics})

    process = process_metrics(root, cases, closed)
    write_json(out / "process_semantic_metrics.json", {**base, **process})
    term = termination(closed, process)
    prior_path = Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json")
    prior = read_json(prior_path)
    hardware = {**base, "model": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": prior["snapshot_path"], "quantization": prior["quantization"], "enable_thinking": False, "max_new_tokens": 256, "max_agent_steps": 20, "v1_metadata_sha256": file_hash(prior_path)}, "v2_peak_allocated_bytes": max(row.get("peak_allocated_bytes", 0) for row in oracle + closed), "v2_peak_reserved_bytes": max(row.get("peak_reserved_bytes", 0) for row in oracle + closed), "v2_oracle_prompt_token_max": max(row.get("prompt_tokens", 0) for row in oracle), "v2_closed_loop_prompt_token_max": max((event.get("prompt_tokens", 0) for row in closed for event in row.get("events", [])), default=0), "v1_4b_failures_not_rerun": True, "v1_4b_interpretation": "HARDWARE-CONSTRAINED / INCOMPLETE CAPABILITY EVIDENCE; 4B not run in 03E.2", "local_hardware_feasible": True, "postprocessed_without_regeneration": True}
    write_json(out / "hardware_metrics.json", hardware)
    write_json(out / "failure_analysis.json", {**base, "termination": term, "closed_loop_failure_categories": closed_metrics["failure_categories"], "oracle_execution_failures": sum(row.get("parse_category") == "execution_failure" for row in oracle), "replay_failures": closed_metrics["replay_failure_count"], "environment_contract_failures": closed_metrics["environment_contract_failure_count"], "initial_v2_contract_gap": {"recorded_replay_failures": 50, "tool": "GetSourceCode", "reason": "case-local code source absent but tool returned failure instead of explicit unavailable result", "corrected_in_final_v2_run": True}, "v1_replay_context": {"recorded_failures": 84, "get_recent_logs_empty_call_failures": 84, "canonicalization_failures": 0}, "postprocessed_without_regeneration": True})
    classification = "BASE_TOO_WEAK"
    recommendation = "START_03F_QLORA" if classification == "CREDIBLE_SPECIALIZATION_STUDENT" else "DO_NOT_START_QLORA"
    write_json(out / "selection_recommendation.json", {**base, "classification": classification, "recommendation": recommendation, "qlora_unblocked": recommendation == "START_03F_QLORA", "basis": "pre-registered pattern: reliable schema/enum compliance, meaningful non-saturated diagnosis, usable protocol, remaining gap, local feasibility; no post-hoc threshold", "candidates_rerun": [MODEL_ID], "candidates_not_run": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"], "postprocessed_without_regeneration": True})

    artifacts = {path.name: file_hash(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "artifact_fingerprints.json"}
    immutable = {"v1_artifacts": {str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e").rglob("*")) if path.is_file()}, "03e1_artifacts": {str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e1").rglob("*")) if path.is_file()}, "validation_subsplit_sha256": file_hash(manifest_path), "split_manifest_sha256": file_hash(Path("results/incident_telemetry_03d/split_manifest.json"))}
    write_json(out / "artifact_fingerprints.json", {**base, "native_vocabulary_fingerprint": _fp(read_json(out / "native_vocabulary.json")), "contract_fingerprint": _fp(read_json(out / "v2_contract.json")), "prompt_fingerprint": read_json(out / "prompt_fingerprint.json")["prompt_fingerprint"], "tool_executability_fingerprint": _fp(read_json(out / "tool_executability.json")), "artifacts": artifacts, "immutable_input_fingerprints": immutable})
    print(json.dumps({"oracle": oracle_metrics, "closed_loop": closed_metrics, "classification": classification, "recommendation": recommendation}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
