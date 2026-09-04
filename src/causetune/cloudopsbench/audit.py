"""Deterministic audits for a scanned Cloud-OpsBench corpus."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping

from .models import CloudOpsBenchCase, CloudOpsBenchSource
from .scanner import CorpusScan, case_counts
from .taxonomy import build_taxonomy_report, native_taxonomy


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _stats(values: Iterable[int]) -> dict[str, int]:
    values = list(values)
    return {
        "count": len(values),
        "min": min(values) if values else 0,
        "median": round(median(values)) if values else 0,
        "p90": _percentile(values, 0.90),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values) if values else 0,
    }


def build_corpus_census(scan: CorpusScan, source: CloudOpsBenchSource = CloudOpsBenchSource()) -> dict[str, Any]:
    observed = case_counts(scan.cases)
    expected_taxonomy = native_taxonomy()
    unknown_types = sorted(set(observed["by_native_fault_type"]) - set(expected_taxonomy))
    category_mismatches = sorted(
        {f"{case.source_fault_type}:{case.source_fault_category}" for case in scan.cases if expected_taxonomy.get(case.source_fault_type) not in {None, case.source_fault_category}}
    )
    documented = {
        "total_cases": source.documented_case_count,
        "fault_type_count": source.documented_fault_type_count,
        "system_case_counts": dict(source.documented_system_case_counts),
    }
    return {
        "source_revision": scan.source_revision,
        "observed": observed,
        "documented_upstream": documented,
        "documented_count_comparison": {
            "total_cases_match": observed["total_cases"] == documented["total_cases"],
            "system_case_counts_match": observed["by_system"] == documented["system_case_counts"],
            "native_fault_type_count_match": len(observed["by_native_fault_type"]) == documented["fault_type_count"],
        },
        "taxonomy_integrity": {"unknown_observed_fault_types": unknown_types, "category_mismatches": category_mismatches},
        "scanner_issues": [issue.to_dict() for issue in scan.issues],
        "read_only": True,
    }


def build_modality_availability(cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    values = tuple(cases)
    all_modalities = sorted({modality for case in values for modality in case.available_modalities})
    counts = {modality: sum(modality in case.available_modalities for case in values) for modality in all_modalities}
    def grouped(group_name: str) -> dict[str, dict[str, int]]:
        groups = sorted({getattr(case, group_name) for case in values})
        return {group: {modality: sum(modality in case.available_modalities for case in values if getattr(case, group_name) == group) for modality in all_modalities} for group in groups}

    return {
        "case_count": len(values),
        "counts": counts,
        "availability_rate": {key: (value / len(values) if values else 0.0) for key, value in counts.items()},
        "by_system": {
            system: {modality: sum(modality in case.available_modalities for case in values if case.source_system == system) for modality in all_modalities}
            for system in sorted({case.source_system for case in values})
        },
        "by_fault_category": grouped("source_fault_category"),
        "by_fault_type": grouped("source_fault_type"),
        "note": "metrics and code are optional upstream modalities; absence is reported, not treated as a universal fault",
    }


def _load_json(root: Path, relative_path: str | None) -> Any | None:
    if not relative_path:
        return None
    path = root / relative_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _walk_text(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        return [text for item in value.values() for text in _walk_text(item)]
    if isinstance(value, list):
        return [text for item in value for text in _walk_text(item)]
    return []


def _walk_keys(value: Any, names: set[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in names and isinstance(item, (str, int, float)):
                found.add(str(item))
            found.update(_walk_keys(item, names))
    elif isinstance(value, list):
        for item in value:
            found.update(_walk_keys(item, names))
    return found


def _trajectory_stats(root: Path, case: CloudOpsBenchCase, reference_name: str, process_milestones: set[str]) -> dict[str, Any]:
    reference = getattr(case.references, reference_name)
    if reference is None:
        return {"available": False}
    payload = _load_json(root, reference.relative_path)
    text = "\n".join(_walk_text(payload)) if payload is not None else ""
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    tool_names: set[str] = set()
    step_count = 0
    sequence: list[str] = []

    def visit(value: Any) -> None:
        nonlocal step_count
        if isinstance(value, list):
            step_count += len(value)
            for item in value:
                visit(item)
        elif isinstance(value, Mapping):
            lower = {str(key).lower(): item for key, item in value.items()}
            if any(key in lower for key in ("tool", "tool_name", "toolname", "function")):
                for key in ("tool", "tool_name", "toolname", "function"):
                    candidate = lower.get(key)
                    if isinstance(candidate, str):
                        tool_names.add(candidate)
                        if "tool_name" in lower and key == "tool_name":
                            sequence.append(candidate)
                    elif isinstance(candidate, Mapping) and isinstance(candidate.get("name"), str):
                        tool_names.add(candidate["name"])
            for item in value.values():
                visit(item)

    trace = payload.get("diagnostic_trace") if isinstance(payload, Mapping) else None
    if isinstance(trace, list):
        step_count = len(trace)
        for item in trace:
            if isinstance(item, Mapping) and isinstance(item.get("tool_name"), str):
                sequence.append(item["tool_name"])
    visit(payload)
    if sequence and len(sequence) > step_count:
        sequence = sequence[:step_count]
    if step_count == 0 and payload is not None:
        step_count = 1
    native_label_leak = case.source_fault_type.lower() in normalized or case.source_fault_category.lower() in normalized
    matched_milestones = sorted(milestone for milestone in process_milestones if milestone.lower() in normalized)
    final_answer_present = False
    if isinstance(payload, Mapping):
        final_answer_present = any(str(key).lower().replace("-", "_") in {"final_answer", "answer", "diagnosis", "root_cause"} for key in payload)
    return {
        "available": True,
        "byte_size": reference.byte_size,
        "character_count": len(text),
        "approximate_step_count": step_count,
        "tool_names": sorted(tool_names),
        "tool_call_sequence": sequence,
        "native_label_literal_present": native_label_leak,
        "final_answer_present": final_answer_present,
        "process_milestones_matched": matched_milestones,
        "sha256": reference.sha256,
        "normalized_surface_hash": hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
        "final_value_type": type(payload).__name__ if payload is not None else "unreadable",
    }


def build_golden_trajectory_audit(root: str | Path, cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    root = Path(root)
    values = tuple(cases)
    records = []
    for case in values:
        process_payload = _load_json(root, case.references.process_label.relative_path) if case.references.process_label else None
        process_milestones = _walk_keys(process_payload, {"id", "milestone", "milestone_id", "name"})
        path1 = _trajectory_stats(root, case, "golden_path1", process_milestones)
        path2 = _trajectory_stats(root, case, "golden_path2", process_milestones)
        records.append({"source_system": case.source_system, "source_fault_type": case.source_fault_type, "process_milestones": sorted(process_milestones), "path1": path1, "path2": path2})
    all_paths = [path for record in records for path in (record["path1"], record["path2"]) if path.get("available")]
    exact_hashes = Counter(path["sha256"] for path in all_paths)
    normalized_hashes = Counter(path["normalized_surface_hash"] for path in all_paths)
    sequence_patterns = Counter(" -> ".join(path.get("tool_call_sequence", [])) for path in all_paths)
    final_answer_paths = [path for path in all_paths if path.get("final_answer_present", False)]
    return {
        "case_count": len(values),
        "cases_with_both_paths": sum(record["path1"].get("available", False) and record["path2"].get("available", False) for record in records),
        "path_availability_by_system": {
            system: {
                "path1": sum(record["source_system"] == system and record["path1"].get("available", False) for record in records),
                "path2": sum(record["source_system"] == system and record["path2"].get("available", False) for record in records),
            }
            for system in sorted({case.source_system for case in values})
        },
        "path_availability_by_fault_type": {
            fault_type: {
                "path1": sum(record["source_fault_type"] == fault_type and record["path1"].get("available", False) for record in records),
                "path2": sum(record["source_fault_type"] == fault_type and record["path2"].get("available", False) for record in records),
            }
            for fault_type in sorted({case.source_fault_type for case in values})
        },
        "path_count": len(all_paths),
        "step_count": _stats(path["approximate_step_count"] for path in all_paths),
        "byte_size": _stats(path["byte_size"] for path in all_paths),
        "character_count": _stats(path["character_count"] for path in all_paths),
        "tool_name_counts": dict(sorted(Counter(tool for path in all_paths for tool in path["tool_names"]).items())),
        "tool_call_sequence_patterns": [{"sequence": sequence, "count": count} for sequence, count in sequence_patterns.most_common(20)],
        "final_answer_structure": {"paths_with_explicit_final_answer": len(final_answer_paths), "paths_without_explicit_final_answer": len(all_paths) - len(final_answer_paths), "note": "Cloud-OpsBench path files are tool traces; final diagnostic answer is not present in the audited trajectory object"},
        "exact_duplicate_surfaces": sum(count - 1 for count in exact_hashes.values() if count > 1),
        "normalized_duplicate_surfaces": sum(count - 1 for count in normalized_hashes.values() if count > 1),
        "native_label_literal_paths": sum(path["native_label_literal_present"] for path in all_paths),
        "process_label_structural_agreement": {
            "cases_with_process_milestones": sum(bool(record["process_milestones"]) for record in records),
            "path_records_with_any_process_milestone_match": sum(bool(path.get("process_milestones_matched")) for record in records for path in (record["path1"], record["path2"])),
            "semantic_correctness": "not inferred; lexical/structural agreement only",
        },
        "tokenizer_specific_lengths": "deferred; this audit reports deterministic bytes/chars only",
        "per_case": records,
    }


def build_context_size_audit(root: str | Path, cases: Iterable[CloudOpsBenchCase], *, context_budget_tokens: int = 8192) -> dict[str, Any]:
    root = Path(root)
    values = tuple(cases)
    modality_names = ("k8s_states", "logs", "metrics", "alerts", "tool_cache", "code")
    records = []
    for case in values:
        sizes: dict[str, int] = {}
        for name in modality_names:
            reference = getattr(case.references, name)
            sizes[name] = reference.byte_size if reference is not None else 0
        raw_total = sum(sizes.values())
        golden_total = sum(getattr(case.references, name).byte_size for name in ("golden_path1", "golden_path2") if getattr(case.references, name) is not None)
        path1_size = case.references.golden_path1.byte_size if case.references.golden_path1 else 0
        path2_size = case.references.golden_path2.byte_size if case.references.golden_path2 else 0
        records.append({"source_system": case.source_system, "source_case_id": case.source_case_id, **sizes, "combined_raw_snapshot": raw_total, "golden_trajectories": golden_total, "golden_path1": path1_size, "golden_path2": path2_size, "estimated_text_tokens_raw": round(raw_total / 4)})
    return {
        "case_count": len(records),
        "size_unit": "bytes",
        "estimated_text_token_method": "bytes / 4 heuristic; no tokenizer was loaded",
        "context_budget_tokens": context_budget_tokens,
        "distributions": {name: _stats(record[name] for record in records) for name in (*modality_names, "combined_raw_snapshot", "golden_trajectories", "golden_path1", "golden_path2", "estimated_text_tokens_raw")},
        "cases_exceeding_estimated_budget": sum(record["estimated_text_tokens_raw"] > context_budget_tokens for record in records),
        "one_shot_flattening_realistic": not any(record["estimated_text_tokens_raw"] > context_budget_tokens for record in records),
        "lossy_truncation_applied": False,
        "per_case": records,
    }


def build_target_availability(cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    values = tuple(cases)
    fields = ("fault_type", "fault_category", "component", "namespace", "service", "diagnosis", "query", "difficulty")
    return {
        "case_count": len(values),
        "matrix": {field: {"available_cases": sum(field in case.ground_truth_metadata for case in values), "source": "metadata.json or source path", "model_input": "SUPERVISION_ONLY" if field in {"fault_type", "fault_category", "component", "diagnosis", "difficulty"} else "MODEL_VISIBLE_WITH_TRANSFORM"} for field in fields},
        "observed_metadata_fields": dict(sorted(Counter(key for case in values for key in case.ground_truth_metadata).items())),
        "note": "only mechanically present source fields are reported; no labels are manufactured",
    }


def build_leakage_field_policy() -> dict[str, Any]:
    fields = [
        ("benchmark path category/case ID", "PROVENANCE_ONLY", "directory names can encode native labels or identity"),
        ("metadata fault fields and diagnosis", "SUPERVISION_ONLY", "explicit source targets"),
        ("metadata query/question", "MODEL_VISIBLE_WITH_TRANSFORM", "inspect and strip explicit answer/label text before input"),
        ("raw_data alert/logs/k8s/metrics", "MODEL_VISIBLE_SAFE", "observable telemetry, subject to case-identity/path stripping"),
        ("tool_cache", "MODEL_VISIBLE_WITH_TRANSFORM", "remove cached target answers or source metadata fields"),
        ("code/", "MODEL_VISIBLE_SAFE", "only where the future task explicitly allows code; normalize paths"),
        ("process-label/milestone.json", "SUPERVISION_ONLY", "diagnostic process target"),
        ("golden-trajectory/path1.json and path2.json", "SUPERVISION_ONLY", "expert trajectory and final answer"),
        ("source fingerprint/revision", "PROVENANCE_ONLY", "audit identity, never model input"),
        ("golden final answer/query fields", "FORBIDDEN_INPUT", "would trivially leak the target"),
    ]
    return {"classification_values": ["MODEL_VISIBLE_SAFE", "MODEL_VISIBLE_WITH_TRANSFORM", "SUPERVISION_ONLY", "PROVENANCE_ONLY", "FORBIDDEN_INPUT"], "fields": [{"field": name, "classification": classification, "rationale": rationale} for name, classification, rationale in fields], "policy_version": "cloud-opsbench-leakage-policy-v1"}


def build_real_leakage_audit(root: str | Path, cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    """Report concrete field/path examples without copying source contents."""

    root = Path(root)
    values = tuple(cases)
    query_values = Counter(str(case.ground_truth_metadata["query"]) for case in values if "query" in case.ground_truth_metadata)
    tool_keys: Counter[str] = Counter()
    for case in values:
        payload = _load_json(root, case.references.tool_cache.relative_path)
        if isinstance(payload, Mapping):
            tool_keys.update(str(key).split(":", 1)[0] for key in payload)
    return {
        "case_count": len(values),
        "explicit_target_fields": {"metadata.result.root_cause": len(values), "metadata.result.fault_taxonomy": len(values), "metadata.result.fault_object": len(values)},
        "concrete_examples": {
            "target_metadata": "benchmark/boutique/infrastructure/31/metadata.json::result.root_cause",
            "category_path": "benchmark/boutique/infrastructure/31/",
            "case_id_path_component": "31",
            "query_values": dict(sorted(query_values.items())),
            "process_label": "process-label/boutique/infrastructure/31/milestone.json",
            "golden_final_answer": "golden-trajectory/boutique/infrastructure/31/path1.json (audited object contains diagnostic_trace, no explicit final answer field)",
        },
        "tool_cache_key_prefix_counts": dict(sorted(tool_keys.items())),
        "legitimate_observable_surfaces": ["raw_data/alert.json", "raw_data/logs.json", "raw_data/k8s_states.json", "raw_data/metrics.csv when present"],
        "input_boundary": {"metadata.result.*": "SUPERVISION_ONLY", "benchmark/process-label/golden-trajectory paths": "PROVENANCE_ONLY or SUPERVISION_ONLY", "raw_data": "MODEL_VISIBLE_SAFE subject to path/label stripping", "tool_cache": "MODEL_VISIBLE_WITH_TRANSFORM"},
        "note": "This is a field/path leakage audit, not a semantic claim that every observable signal is safe for every future task.",
    }


def _hash_bucket(value: str, buckets: int = 100) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16) % buckets


def _case_holdout(cases: tuple[CloudOpsBenchCase, ...]) -> dict[str, Any]:
    counts = Counter("TEST" if _hash_bucket(case.grouped_case_id) < 20 else "DEV" if _hash_bucket(case.grouped_case_id) < 30 else "TRAIN" for case in cases)
    return {"counts": dict(sorted(counts.items())), "grouping": "source revision + system + source case ID", "label_generalization": "case holdout; native labels may overlap"}


def _train_dev_test_counts(cases: Iterable[CloudOpsBenchCase]) -> dict[str, int]:
    counts = Counter()
    for case in cases:
        bucket = _hash_bucket(case.grouped_case_id)
        counts["TEST" if bucket < 20 else "DEV" if bucket < 30 else "TRAIN"] += 1
    return dict(sorted(counts.items()))


def _train_dev_counts(cases: Iterable[CloudOpsBenchCase]) -> dict[str, int]:
    counts = Counter("DEV" if _hash_bucket(case.grouped_case_id) < 30 else "TRAIN" for case in cases)
    return dict(sorted(counts.items()))


def _label_summary(train: Iterable[CloudOpsBenchCase], validation: Iterable[CloudOpsBenchCase], test: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    train, validation, test = tuple(train), tuple(validation), tuple(test)
    train_labels = {item.source_fault_type for item in train}
    validation_labels = {item.source_fault_type for item in validation}
    test_labels = {item.source_fault_type for item in test}
    return {
        "train_count": len(train), "validation_count": len(validation), "protected_test_count": len(test),
        "train_labels": sorted(train_labels), "validation_labels": sorted(validation_labels), "protected_test_labels": sorted(test_labels),
        "train_test_label_overlap": sorted(train_labels & test_labels), "test_only_labels": sorted(test_labels - train_labels),
        "train_test_system_overlap": sorted({item.source_system for item in train} & {item.source_system for item in test}),
        "train_test_category_overlap": sorted({item.source_fault_category for item in train} & {item.source_fault_category for item in test}),
    }


def build_split_feasibility(cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    values = tuple(cases)
    if not values:
        return {"status": "not_run_dataset_unavailable_or_empty"}
    native_types = sorted({case.source_fault_type for case in values})
    heldout_types = set(native_types[::5])
    type_counts = Counter(case.source_fault_type for case in values)
    category_holdout = sorted({case.source_fault_category for case in values})[-1]
    systems = sorted({case.source_system for case in values})
    train_system = "boutique" if "boutique" in systems else systems[0]
    test_system = "trainticket" if "trainticket" in systems else (systems[-1] if len(systems) > 1 else systems[0])
    code_cases = [case for case in values if case.references.code is not None]
    type_train_dev = [case for case in values if case.source_fault_type not in heldout_types]
    category_train_dev = [case for case in values if case.source_fault_category != category_holdout]
    case_train = [case for case in values if _hash_bucket(case.grouped_case_id) >= 30]
    case_dev = [case for case in values if 20 <= _hash_bucket(case.grouped_case_id) < 30]
    case_test = [case for case in values if _hash_bucket(case.grouped_case_id) < 20]
    cross_train = [case for case in values if case.source_system == train_system and _hash_bucket(case.grouped_case_id) >= 20]
    cross_dev = [case for case in values if case.source_system == train_system and _hash_bucket(case.grouped_case_id) < 20]
    cross_test = [case for case in values if case.source_system == test_system]
    code_test = [case for case in values if case.references.code is not None]
    code_train_dev = [case for case in values if case.references.code is None]
    code_train = [case for case in code_train_dev if _hash_bucket(case.grouped_case_id) >= 20]
    code_dev = [case for case in code_train_dev if _hash_bucket(case.grouped_case_id) < 20]
    return {
        "status": "candidate_strategies_only; no final split frozen",
        "case_holdout": {**_case_holdout(values), "label_summary": _label_summary(case_train, case_dev, case_test)},
        "fault_type_holdout": {"heldout_native_fault_types": sorted(heldout_types), "counts": {**_train_dev_counts(type_train_dev), "TEST": sum(case.source_fault_type in heldout_types for case in values)}, "label_summary": _label_summary([case for case in type_train_dev if _hash_bucket(case.grouped_case_id) >= 30], [case for case in type_train_dev if _hash_bucket(case.grouped_case_id) < 30], [case for case in values if case.source_fault_type in heldout_types]), "label_generalization": "unseen-label classification, not topology OOD", "coverage": {key: type_counts[key] for key in sorted(type_counts)}},
        "category_holdout": {"heldout_category": category_holdout, "counts": {**_train_dev_counts(category_train_dev), "TEST": sum(case.source_fault_category == category_holdout for case in values)}, "label_summary": _label_summary([case for case in category_train_dev if _hash_bucket(case.grouped_case_id) >= 30], [case for case in category_train_dev if _hash_bucket(case.grouped_case_id) < 30], [case for case in values if case.source_fault_category == category_holdout]), "label_generalization": "cross-category generalization; category is confounded with fault labels"},
        "cross_system_holdout": {"train_development_system": train_system, "test_system": test_system, "counts": {system: sum(case.source_system == system for case in values) for system in systems}, "label_summary": _label_summary(cross_train, cross_dev, cross_test), "overlap_fault_types": sorted({case.source_fault_type for case in values if case.source_system == train_system} & {case.source_fault_type for case in values if case.source_system == test_system}), "train_only_fault_types": sorted({case.source_fault_type for case in values if case.source_system == train_system} - {case.source_fault_type for case in values if case.source_system == test_system}), "test_only_fault_types": sorted({case.source_fault_type for case in values if case.source_system == test_system} - {case.source_fault_type for case in values if case.source_system == train_system}), "interpretation": "system holdout is not pure topology OOD when native labels differ"},
        "code_defect_holdout": {"code_present_cases": len(code_cases), "code_absent_cases": len(values) - len(code_cases), "label_summary": _label_summary(code_train, code_dev, code_test), "interpretation": "feasible only for cases with code artifacts; code availability is a confound and needs explicit 03D design"},
        "protected_evidence_policy": {"case_group_is_atomic": True, "derived_variants_inherit_source_case_split": True, "golden_trajectories_stay_with_case": True, "process_labels_stay_with_case": True, "rcaeval_in_training": False, "sealed_holdout_consumed_once": True},
    }


def audit_corpus(scan: CorpusScan, source: CloudOpsBenchSource = CloudOpsBenchSource()) -> dict[str, Any]:
    errors = [issue.to_dict() for issue in scan.issues if issue.severity == "error"]
    census = build_corpus_census(scan, source)
    taxonomy_integrity = census["taxonomy_integrity"]
    if taxonomy_integrity["unknown_observed_fault_types"]:
        errors.append({"path": "native taxonomy", "message": f"unknown observed native fault types: {taxonomy_integrity['unknown_observed_fault_types']}", "severity": "error"})
    if taxonomy_integrity["category_mismatches"]:
        errors.append({"path": "native taxonomy", "message": f"fault category mismatches: {taxonomy_integrity['category_mismatches']}", "severity": "error"})
    fingerprints = [case.source_fingerprint for case in scan.cases]
    duplicates = sorted(fingerprint for fingerprint, count in Counter(fingerprints).items() if count > 1)
    if duplicates:
        errors.append({"path": "case fingerprints", "message": f"duplicate source fingerprints: {duplicates}", "severity": "error"})
    return {"status": "fail" if errors else "pass", "error_count": len(errors), "errors": errors, "case_count": len(scan.cases), "duplicate_source_fingerprints": duplicates, "taxonomy_integrity": taxonomy_integrity, "source_revision": scan.source_revision, "native_labels_preserved": True, "read_only": True}
