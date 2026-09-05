#!/usr/bin/env python3
"""Finalize the offline 03E screening record from completed candidate artifacts.

This script does not load model weights or generate text.  It records the
already completed screen, computes comparison/selection metadata, and measures
tokenizer-only context thresholds from the frozen TRAIN/VALIDATION_SCREEN
representations.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from causetune.cloudopsbench import scan_corpus
from causetune.cloudopsbench.screening import (
    REPRESENTATION_FINGERPRINT,
    SOURCE_REVISION,
    SPLIT_FINGERPRINT,
    TOOL_SCHEMAS,
    _messages_for_initial,
    source_case_group_id,
)
from causetune.cloudopsbench.training_contract import normalize_trajectory


ROOT = Path("results/incident_telemetry_03e")
SOURCE_ROOT = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
MODELS = {
    "Qwen/Qwen3.5-0.8B": "Qwen__Qwen3.5-0.8B",
    "Qwen/Qwen3.5-2B": "Qwen__Qwen3.5-2B",
    "Qwen/Qwen3.5-4B": "Qwen__Qwen3.5-4B",
}
REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def apply_template(tokenizer: Any, messages: list[dict[str, Any]]) -> Any:
    return tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)


def tensor_parameter_count(snapshot: Path) -> int | None:
    try:
        from safetensors import safe_open
        from math import prod
        total = 0
        for shard in sorted(snapshot.glob("*.safetensors")):
            with safe_open(str(shard), framework="pt", device="cpu") as handle:
                total += sum(prod(handle.get_slice(name).get_shape()) for name in handle.keys())
        return total
    except Exception:
        return None


def collect_context_lengths(tokenizer: Any, cases: list[Any], *, skip_golden_fallbacks: bool) -> list[int]:
    values: list[int] = []
    for case in cases:
        try:
            normalized = normalize_trajectory(SOURCE_ROOT, case, "golden_path1")
            if skip_golden_fallbacks and any(step["replay_status"] == "RESOLVED_FROM_GOLDEN_TRACE" for step in normalized["replay_steps"]):
                continue
            messages = _messages_for_initial(case)
            for step in normalized["replay_steps"]:
                messages.extend((
                    {"role": "assistant", "content": f"<tool_call><function={step['tool_id']}></function></tool_call>"},
                    {"role": "tool", "content": step["observation"]["text"]},
                ))
            values.append(len(apply_template(tokenizer, messages + [{"role": "user", "content": "Return the final structured diagnosis JSON now."}])["input_ids"][0]))
        except Exception:
            continue
    return values


def threshold_counts(values: list[int]) -> dict[str, int]:
    return {f"over_{threshold}_tokens": sum(value > threshold for value in values) for threshold in (4096, 8192, 16384, 32768, 65536)}


def update_model_metadata() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for model_id, directory in MODELS.items():
        path = ROOT / directory / "model_metadata.json"
        metadata = read_json(path)
        snapshot = Path(metadata["snapshot_path"])
        count = tensor_parameter_count(snapshot)
        metadata.update({"license": "Apache-2.0", "parameter_count": count, "parameter_count_source": "safetensors header shapes", "selection_candidate": True})
        write_json(path, metadata)
        result[model_id] = metadata
    return result


def candidate_summary(model_id: str, metadata: dict[str, Any], directory: Path) -> dict[str, Any]:
    closed = read_json(directory / "closed_loop_metrics.json")
    oracle = read_json(directory / "oracle_evidence_metrics.json")
    action = read_json(directory / "next_action_metrics.json")
    tokenization = read_json(directory / "tokenization_metrics.json")
    hardware = read_json(directory / "hardware_metrics.json")
    failure_analysis = read_json(directory / "closed_loop_failure_analysis.json")
    oracle_rows = [json.loads(line) for line in (directory / "oracle_evidence_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    action_rows = [json.loads(line) for line in (directory / "next_action_predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    failure_analysis = {
        "closed_loop": failure_analysis,
        "oracle_parse_categories": {category: sum(1 for row in oracle_rows if row.get("parse_category") == category) for category in sorted({row.get("parse_category") for row in oracle_rows})},
        "teacher_forced": {"records": len(action_rows), "tool_name_exact": sum(bool(row.get("tool_name_exact")) for row in action_rows), "tool_name_valid": sum(bool(row.get("tool_name_valid")) for row in action_rows), "argument_schema_valid": sum(bool(row.get("arguments_schema_valid")) for row in action_rows)},
    }
    write_json(directory / "failure_analysis.json", failure_analysis)
    return {
        "model_id": model_id,
        "revision": metadata["revision"],
        "license": metadata["license"],
        "parameter_count": metadata.get("parameter_count"),
        "native_context_length": metadata.get("native_context_length"),
        "config_hash": metadata.get("config_hash"),
        "tokenizer_hash": metadata.get("tokenizer_hash"),
        "chat_template_hash": metadata.get("chat_template_hash"),
        "closed_loop": closed,
        "oracle_evidence": oracle,
        "next_action": action,
        "tokenization": tokenization,
        "hardware": hardware,
        "failure_analysis": failure_analysis,
        "static_evidence_diagnosis": {"status": "not_run", "reason": "03D static manifest is provenance-only; no separate materialized static input was available without creating a new packaging strategy"},
    }


def main() -> int:
    split = read_json(Path("results/incident_telemetry_03d/split_manifest.json"))
    if split.get("fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("frozen split fingerprint mismatch")
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(SOURCE_ROOT)
    scan = scan_corpus(SOURCE_ROOT, source_revision=SOURCE_REVISION, fail_closed=True)
    subsplit = read_json(ROOT / "validation_subsplit.json")
    screen_groups = set(subsplit["screen_case_groups"])
    split_by_group = {item["source_case_group"]: item["split"] for item in split["entries"]}
    cases_by_group = {source_case_group_id(case): case for case in scan.cases}
    screen_cases = [cases_by_group[group] for group in sorted(screen_groups)]
    train_cases = [case for case in scan.cases if split_by_group.get(source_case_group_id(case)) == "TRAIN"]
    if len(screen_cases) != 57 or len(subsplit["selection_case_groups"]) != 66:
        raise ValueError("validation sub-split counts changed")

    metadata = update_model_metadata()
    # All three pinned candidates share the tokenizer files/hash.  Measure once
    # from the cached tokenizer and attach the identical measurements to each
    # candidate rather than loading any model weights.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(metadata["Qwen/Qwen3.5-0.8B"]["snapshot_path"], local_files_only=True, trust_remote_code=False)
    screen_values = collect_context_lengths(tokenizer, screen_cases, skip_golden_fallbacks=False)
    train_values = collect_context_lengths(tokenizer, train_cases, skip_golden_fallbacks=True)
    for model_id, directory in MODELS.items():
        path = ROOT / directory / "tokenization_metrics.json"
        tokenization = read_json(path)
        tokenization["threshold_counts"] = {"screen_final_context_tokens": threshold_counts(screen_values), "train_final_context_tokens": threshold_counts(train_values)}
        tokenization["screen_context_recomputed_count"] = len(screen_values)
        tokenization["train_context_recomputed_count"] = len(train_values)
        tokenization["tokenizer_measurement_shared_across_candidates"] = True
        write_json(path, tokenization)
        metadata[model_id]["tokenization"] = tokenization
        write_json(ROOT / directory / "model_metadata.json", metadata[model_id])

    candidates = {model_id: candidate_summary(model_id, metadata[model_id], ROOT / directory) for model_id, directory in MODELS.items()}
    common = {"experiment": "03E", "implementation_version": "cloud-opsbench-03e-screening-v1", "source_revision": SOURCE_REVISION, "source_root": str(SOURCE_ROOT), "split_manifest_fingerprint": SPLIT_FINGERPRINT, "representation_fingerprint": REPRESENTATION_FINGERPRINT, "model_training": False, "provider_called": False, "test_model_facing": False}
    registry = {**common, "candidates": [{"model_id": model_id, "revision": REVISIONS[model_id], "post_trained_variant": True, "license": metadata[model_id]["license"], "parameter_count": metadata[model_id].get("parameter_count"), "native_context_length": metadata[model_id].get("native_context_length"), "config_hash": metadata[model_id].get("config_hash"), "tokenizer_hash": metadata[model_id].get("tokenizer_hash"), "chat_template_hash": metadata[model_id].get("chat_template_hash"), "primary_quantization": "NF4 4-bit / BF16 / double quantization"} for model_id in MODELS]}
    write_json(ROOT / "candidate_registry.json", registry)
    write_json(ROOT / "candidate_comparison.json", {**common, "candidate_count": len(candidates), "comparison_basis": "same frozen prompt, tool contract, non-thinking mode, deterministic decode, and screen cases", "candidates": candidates})
    write_json(ROOT / "hardware_environment.json", {**common, "gpu": "NVIDIA GeForce RTX 5070 Laptop GPU", "driver": "581.80", "vram_mib": 8151, "compute_capability": "12.0", "software": {"torch": "2.14.0+cu130", "transformers": "5.16.1", "bitsandbytes": "0.50.2"}, "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}, "note": "4B longest context probe completed technically but exceeded the 8 GiB physical allocated-memory envelope; reserved-memory accounting is reported separately in per-candidate artifacts."})
    recommendation = {**common, "recommendation": "NO_CREDIBLE_STUDENT_YET", "selection_rule": "smallest model must show protocol viability, oracle learnability signal, closed-loop headroom, local feasibility, and a credible larger-model gradient", "evidence": {model_id: {"oracle_joint_exact": candidates[model_id]["oracle_evidence"].get("joint_exact"), "closed_loop_joint_exact": candidates[model_id]["closed_loop"].get("joint_exact"), "oracle_schema_valid_rate": candidates[model_id]["oracle_evidence"].get("schema_valid_rate"), "closed_loop_schema_valid_rate": candidates[model_id]["closed_loop"].get("schema_valid_rate"), "tool_name_exact_rate": candidates[model_id]["next_action"].get("tool_name_exact_rate"), "max_probe_peak_allocated_bytes": candidates[model_id]["hardware"].get("peak_allocated_bytes")} for model_id in MODELS}, "rationale": "All three untouched candidates scored 0/57 joint diagnosis under ORACLE_EVIDENCE_DIAGNOSIS and 0/57 under CLOSED_LOOP_AGENT. Qwen3.5-2B has the strongest protocol/schema combination, but no diagnosis learnability signal; Qwen3.5-4B has higher next-action exactness but is not a credible primary student because its oracle diagnosis remains zero and its longest probe exceeds the 8 GiB allocated-memory envelope.", "recommendation_for_03f": "DO_NOT_START_QLORA; review target/prompt/tool compatibility and obtain a credible oracle diagnosis signal before specialization."}
    write_json(ROOT / "selection_recommendation.json", recommendation)
    write_json(ROOT / "screen_integrity.json", {**common, "validation_screen_cases": len(screen_cases), "validation_selection_cases": len(subsplit["selection_case_groups"]), "test_model_predictions": 0, "per_candidate": {model_id: {"closed_loop_records": len((ROOT / directory / "closed_loop_predictions.jsonl").read_text().splitlines()), "oracle_records": len((ROOT / directory / "oracle_evidence_predictions.jsonl").read_text().splitlines()), "teacher_records": len((ROOT / directory / "next_action_predictions.jsonl").read_text().splitlines())} for model_id, directory in MODELS.items()}})

    hashes = {}
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path.name != "artifact_fingerprints.json":
            hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(ROOT / "artifact_fingerprints.json", {**common, "artifacts": hashes})
    print(json.dumps({"status": "pass", "screen": len(screen_cases), "selection": len(subsplit["selection_case_groups"]), "recommendation": "NO_CREDIBLE_STUDENT_YET"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
