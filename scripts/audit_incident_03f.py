#!/usr/bin/env python3
"""Freeze the CPU-only 03F hierarchical RCA decomposition contract."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from causetune.cloudopsbench import (
    build_evidence_packages,
    build_hierarchy_registry,
    context_size_audit,
    evidence_sufficiency_audit,
    future_record_counts,
    label_support,
    scan_corpus,
    task_contracts,
    trivial_baselines,
)
from causetune.cloudopsbench.screening import _fp
from causetune.cloudopsbench.training_contract import source_case_group_id


ROOT = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
SOURCE_REVISION = "03c415e5709297432282fbbfd499f1bca0f8c347"
SPLIT_FINGERPRINT = "0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76"
REPRESENTATION_FINGERPRINT = "f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf"
VERSION = "cloud-opsbench-03f-hierarchical-decomposition-v1"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata() -> dict:
    return {"experiment": "03F", "version": VERSION, "source_revision": SOURCE_REVISION, "source_root": str(ROOT), "split_manifest_fingerprint": SPLIT_FINGERPRINT, "representation_fingerprint": REPRESENTATION_FINGERPRINT, "model_loaded": False, "model_trained": False, "gpu_used": False, "provider_called": False, "test_model_facing": False, "validation_selection_model_facing": False}


def main() -> int:
    out = Path("results/incident_telemetry_03f")
    out.mkdir(parents=True, exist_ok=True)
    split_path = Path("results/incident_telemetry_03d/split_manifest.json")
    split = read_json(split_path)
    if split.get("fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("03D split fingerprint mismatch")
    prior_manifest = read_json(Path("results/incident_telemetry_03e/validation_subsplit.json"))
    scan = scan_corpus(ROOT, source_revision=SOURCE_REVISION, fail_closed=True)
    by_group = {source_case_group_id(case): case for case in scan.cases}
    split_by_group = {item["source_case_group"]: item["split"] for item in split["entries"]}
    screen_groups = sorted(prior_manifest["screen_case_groups"])
    screen_cases = [by_group[group] for group in screen_groups]
    if len(screen_cases) != 57 or any(split_by_group[group] != "VALIDATION" for group in screen_groups):
        raise ValueError("frozen VALIDATION_SCREEN changed")
    non_test_cases = [case for case in scan.cases if split_by_group[source_case_group_id(case)] in {"TRAIN", "VALIDATION"}]
    train_cases = [case for case in non_test_cases if split_by_group[source_case_group_id(case)] == "TRAIN"]

    # This is a source audit, not model-facing use of TEST.  Taxonomy coverage
    # is derived from the complete pinned corpus; evidence packages exclude it.
    registry = build_hierarchy_registry(scan.cases)
    if registry["native_fault_type_count"] != 57 or registry["conflicts"]:
        raise ValueError("native hierarchy is incomplete or conflicting")
    distribution = {**metadata(), "all_corpus_case_count": len(scan.cases), "by_category": dict(sorted(Counter(case.source_fault_category for case in scan.cases).items())), "by_root_cause": dict(sorted(Counter(case.source_fault_type for case in scan.cases).items())), "by_split_category": {split_name: dict(sorted(Counter(case.source_fault_category for case in scan.cases if split_by_group[source_case_group_id(case)] == split_name).items())) for split_name in ("TRAIN", "VALIDATION", "TEST")}, "by_split_root_cause": {split_name: dict(sorted(Counter(case.source_fault_type for case in scan.cases if split_by_group[source_case_group_id(case)] == split_name).items())) for split_name in ("TRAIN", "VALIDATION", "TEST")}, "category_candidate_set_sizes": registry["candidate_set_sizes"]}
    write_json(out / "hierarchy_registry.json", {**metadata(), **registry, "source_authority": "pinned Cloud-OpsBench metadata.json corpus; no manual taxonomy reconstruction"})
    write_json(out / "hierarchy_distribution.json", distribution)

    contracts = task_contracts(registry)
    for name, contract in contracts.items():
        write_json(out / {"TASK_A_CATEGORY": "task_a_contract.json", "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE": "task_b_contract.json", "TASK_C_HIERARCHICAL_SELF_PREDICTED": "task_c_contract.json", "TASK_D_FAULT_OBJECT": "task_d_contract.json"}[name], {**metadata(), **contract})

    packages = build_evidence_packages(ROOT, non_test_cases)
    package_by_group = {}
    sorted_non_test = sorted(non_test_cases, key=source_case_group_id)
    for case, row in zip(sorted_non_test, packages):
        row["audit_split"] = split_by_group[source_case_group_id(case)]
        row["audit_category"] = case.source_fault_category
        package_by_group[source_case_group_id(case)] = row
    # The package file has no target/process/provenance in its model_input;
    # audit columns are retained separately for offline accounting.
    persisted_packages = [{key: value for key, value in row.items() if key not in {"audit_split", "audit_category"}} for row in packages]
    write_jsonl(out / "evidence_packages.jsonl", persisted_packages)
    evidence_contract = {**metadata(), "version": VERSION, "policy": "PATH1_PREFERRED_WITH_PATH2_FALLBACK_ONLY_WHEN_PATH1_HAS_ZERO_EXECUTABLE_STEPS", "selection_is_model_independent": True, "union_policy": "not used; one deterministic preferred view per case", "deduplication": "not applicable to single-path policy", "budget_chars": 30000, "budget_algorithm": "if serialized package exceeds budget, binary-search one equal deterministic per-observation line-package cap; no target/process/model signal", "packager_inputs": ["sanitized incident request", "RESOLVED_FROM_TOOL_CACHE observations from selected golden path"], "packager_forbidden": ["root cause", "fault category", "fault object", "process labels", "golden final answer", "source paths", "provenance metadata"], "package_count": len(packages), "package_counts_by_split": dict(sorted(Counter(row["audit_split"] for row in packages).items())), "selected_path_counts": dict(sorted(Counter(row["selection"]["selected_path"] for row in packages).items())), "fallback_path2_count": sum(row["selection"]["fallback_used"] for row in packages), "persisted_package_excludes_audit_join_metadata": True, "model_facing_record_schema": {"incident_request": "string", "evidence": [{"tool": "string", "observation": "bounded source observation"}]}}
    write_json(out / "evidence_selection_contract.json", evidence_contract)

    sufficiency = evidence_sufficiency_audit(ROOT, non_test_cases, package_by_group)
    sufficiency.update(metadata())
    write_json(out / "evidence_sufficiency_audit.json", sufficiency)

    prior_context = read_json(Path("results/incident_telemetry_03d1/cumulative_context.json"))
    tokenizer_meta = read_json(Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json"))
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_meta["snapshot_path"]), local_files_only=True, trust_remote_code=False)
    context = context_size_audit(packages, tokenizer=tokenizer, prior_full_history=prior_context, registry=registry)
    context["scope"] = "TRAIN and VALIDATION only; TEST excluded from model-facing packages"
    context["task_c_source_context_note"] = "Stage 2 context sizes include the fixed candidate-set registry; no Stage-1 predictions were manufactured."
    context["decomposed_input_is_not_full_history"] = True
    context.update(metadata())
    write_json(out / "context_size_audit.json", context)

    support = label_support(scan.cases, split_by_group, registry)
    support.update(metadata())
    write_json(out / "train_label_support.json", support)
    write_json(out / "future_training_record_counts.json", {**metadata(), **future_record_counts(scan.cases, split_by_group)})

    baseline = trivial_baselines(registry, screen_cases, train_cases)
    baseline.update(metadata())
    write_json(out / "trivial_baselines.json", baseline)
    selection_policy = {**metadata(), "version": VERSION, "future_screen": "03G", "no_model_run_in_03F": True, "candidate_tasks": ["TASK_A_CATEGORY", "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE", "TASK_C_HIERARCHICAL_SELF_PREDICTED"], "task_d_role": "secondary diagnostic head; not primary specialization target", "pre_registered_criteria": ["materially above deterministic trivial baseline", "clearly below saturation", "reliable schema and enum following", "practical context and local hardware feasibility", "enough TRAIN support for every target class", "no environment-contract confound"], "preference_order": ["TASK_C_HIERARCHICAL_SELF_PREDICTED", "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE", "TASK_A_CATEGORY"], "task_c_gate": "both Stage 1 category and Stage 2 within-category discrimination must show meaningful signal", "task_b_boundary": "always report as ORACLE_CATEGORY_ROOT_CAUSE, never end-to-end RCA", "threshold_policy": "qualitative pre-registered pattern; no post-hoc numerical cutoff", "fallback_if_no_candidate": "TASK_REDESIGN_REQUIRED", "qlora_policy": "03H remains blocked until a future decomposed screen supports a credible specialization target"}
    write_json(out / "specialization_selection_policy.json", selection_policy)

    immutable = {"03d_split_manifest": file_hash(split_path), "03e_validation_subsplit": file_hash(Path("results/incident_telemetry_03e/validation_subsplit.json")), "prior_experiment_fingerprints": {name: read_json(Path(path)).get("split_manifest_fingerprint") for name, path in [("03e", "results/incident_telemetry_03e/artifact_fingerprints.json"), ("03e1", "results/incident_telemetry_03e1/artifact_fingerprints.json"), ("03e2", "results/incident_telemetry_03e2/artifact_fingerprints.json")]}}
    freeze_payload = {"version": VERSION, "source_revision": SOURCE_REVISION, "split_manifest_fingerprint": SPLIT_FINGERPRINT, "representation_fingerprint": REPRESENTATION_FINGERPRINT, "hierarchy_registry_fingerprint": registry["hierarchy_fingerprint"], "evidence_selection_contract_fingerprint": _fp(evidence_contract), "task_contract_fingerprints": {name: _fp(value) for name, value in contracts.items()}, "normalization_policy": contracts["TASK_D_FAULT_OBJECT"]["normalization"], "trivial_baseline_fingerprint": _fp(baseline), "specialization_selection_policy_fingerprint": _fp(selection_policy), "immutable_inputs": immutable}
    freeze_payload["decomposition_freeze_fingerprint"] = _fp(freeze_payload)
    write_json(out / "freeze_manifest.json", {**metadata(), **freeze_payload, "model_outputs_present": False, "model_facing_test_records": 0})

    artifacts = {path.name: file_hash(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "artifact_fingerprints.json"}
    prior_hashes = {}
    for name, directory in (("03a_03b_03c_03d", "results/incident_telemetry_03d"), ("03e", "results/incident_telemetry_03e"), ("03e1", "results/incident_telemetry_03e1"), ("03e2", "results/incident_telemetry_03e2")):
        prior_hashes[name] = {str(path): file_hash(path) for path in sorted(Path(directory).rglob("*")) if path.is_file()}
    write_json(out / "artifact_fingerprints.json", {**metadata(), "decomposition_freeze_fingerprint": freeze_payload["decomposition_freeze_fingerprint"], "artifacts": artifacts, "prior_artifact_hashes": prior_hashes})
    print(json.dumps({"hierarchy": {"categories": registry["native_category_count"], "fault_types": registry["native_fault_type_count"], "candidate_sizes": registry["candidate_set_sizes"]}, "packages": len(packages), "sufficiency": {key: sufficiency[key] for key in ("total_process_label_evidence_patterns", "retained_evidence_patterns", "cases_with_full_coverage", "cases_with_partial_coverage", "cases_with_zero_coverage")}, "freeze_fingerprint": freeze_payload["decomposition_freeze_fingerprint"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
