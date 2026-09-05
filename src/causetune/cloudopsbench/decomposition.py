"""CPU-only 03F hierarchical RCA decomposition contracts.

This module contains no model or provider code.  It builds machine-readable
diagnostic task contracts from the pinned Cloud-OpsBench corpus and packages
only deterministic source-derived evidence for future experiments.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causetune.incident_telemetry.fingerprints import canonical_json

from .models import CloudOpsBenchCase
from .training_contract import MAX_OBSERVATION_CHARS, _sanitized_query, _uniform_line_package, normalize_trajectory, source_case_group_id


DECOMPOSITION_VERSION = "cloud-opsbench-03f-hierarchical-decomposition-v1"
TASK_NAMES = ("TASK_A_CATEGORY", "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE", "TASK_C_HIERARCHICAL_SELF_PREDICTED", "TASK_D_FAULT_OBJECT")
EXECUTABLE_REPLAY_STATUS = "RESOLVED_FROM_TOOL_CACHE"
PACKAGE_BUDGET_CHARS = 30000


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def build_hierarchy_registry(cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    """Derive the native category -> root-cause hierarchy from case metadata."""

    by_type: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        by_type[str(case.source_fault_type)].add(str(case.source_fault_category))
    conflicts = {fault_type: sorted(categories) for fault_type, categories in by_type.items() if len(categories) != 1}
    if conflicts:
        raise ValueError(f"native fault types have conflicting categories: {conflicts}")
    by_category: dict[str, list[str]] = defaultdict(list)
    for fault_type, categories in by_type.items():
        by_category[next(iter(categories))].append(fault_type)
    categories = sorted(by_category)
    ordered = {category: sorted(by_category[category]) for category in categories}
    return {
        "version": DECOMPOSITION_VERSION,
        "derivation": "sorted unique metadata.result.root_cause grouped by metadata.result.fault_taxonomy from the pinned corpus",
        "ordered_categories": categories,
        "categories": ordered,
        "fault_type_to_category": {fault_type: category for category in categories for fault_type in ordered[category]},
        "native_fault_type_count": len(by_type),
        "native_category_count": len(categories),
        "conflicts": conflicts,
        "candidate_set_sizes": {category: len(ordered[category]) for category in categories},
        "hierarchy_fingerprint": fingerprint({"ordered_categories": categories, "categories": ordered}),
    }


def task_contracts(registry: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    categories = list(registry["ordered_categories"])
    return {
        "TASK_A_CATEGORY": {
            "version": DECOMPOSITION_VERSION,
            "name": "TASK_A_CATEGORY",
            "label": "CATEGORY_CLASSIFICATION",
            "primary": True,
            "input_schema": {"type": "object", "required": ["incident_request", "evidence"], "properties": {"incident_request": {"type": "string"}, "evidence": {"type": "array", "items": {"type": "object", "required": ["tool", "observation"]}}}},
            "output_schema": {"type": "object", "required": ["fault_category"], "additionalProperties": False, "properties": {"fault_category": {"type": "string", "enum_ref": "hierarchy_registry.ordered_categories"}}},
            "allowed_categories": categories,
            "forbidden_outputs": ["root_cause", "fault_object", "confidence", "reasoning", "explanation"],
            "metrics": ["category_exact", "schema_valid", "enum_valid", "confusion_matrix"],
        },
        "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE": {
            "version": DECOMPOSITION_VERSION,
            "name": "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE",
            "label": "ORACLE_CATEGORY_ROOT_CAUSE",
            "primary": True,
            "oracle_context": True,
            "input_schema": {"type": "object", "required": ["incident_request", "evidence", "fault_category_context"], "properties": {"incident_request": {"type": "string"}, "evidence": {"type": "array"}, "fault_category_context": {"type": "string", "enum_ref": "hierarchy_registry.ordered_categories"}}},
            "output_schema": {"type": "object", "required": ["root_cause"], "additionalProperties": False, "properties": {"root_cause": {"type": "string", "enum_ref": "hierarchy_registry.categories[fault_category_context]"}}},
            "candidate_restriction": "only root causes belonging to supplied authoritative category",
            "end_to_end_rca": False,
            "metrics": ["root_cause_exact", "schema_valid", "enum_valid", "per_category_accuracy"],
        },
        "TASK_C_HIERARCHICAL_SELF_PREDICTED": {
            "version": DECOMPOSITION_VERSION,
            "name": "TASK_C_HIERARCHICAL_SELF_PREDICTED",
            "label": "HIERARCHICAL_SELF_PREDICTED_RCA",
            "primary": True,
            "stages": [{"stage": 1, "input": "evidence", "output": "predicted fault_category", "context": "complete native category enum"}, {"stage": 2, "input": "same evidence plus Stage-1 predicted category", "output": "root_cause", "candidate_set": "registry[predicted category]", "authoritative_category_used": False}],
            "output_schema": {"type": "object", "required": ["fault_category", "root_cause"], "additionalProperties": False, "properties": {"fault_category": {"type": "string", "enum_ref": "hierarchy_registry.ordered_categories"}, "root_cause": {"type": "string", "enum_ref": "hierarchy_registry.categories[predicted_fault_category]"}}},
            "metrics": ["category_exact", "root_cause_exact", "hierarchical_joint_exact", "stage2_conditional_accuracy_given_category_correct", "stage2_failure_wrong_stage1_category"],
            "error_decomposition": ["CATEGORY_ERROR", "WITHIN_CATEGORY_DISCRIMINATION_ERROR"],
        },
        "TASK_D_FAULT_OBJECT": {
            "version": DECOMPOSITION_VERSION,
            "name": "TASK_D_FAULT_OBJECT",
            "label": "FAULT_OBJECT_RESOLUTION",
            "primary": False,
            "input_schema": {"type": "object", "required": ["incident_request", "evidence"], "properties": {"incident_request": {"type": "string"}, "evidence": {"type": "array"}}},
            "output_schema": {"type": "object", "required": ["fault_object"], "additionalProperties": False, "properties": {"fault_object": {"type": "string"}}},
            "normalization": {"allowed": True, "policy": "bare application name -> app/name only when a unique app identity is visible in packaged source observations; do not collapse different Kubernetes resources", "uses_target": False, "uses_process_labels": False, "uses_golden_target": False},
            "metrics": ["strict_object_exact", "normalized_object_exact"],
        },
    }


def _executable_steps(root: Path, case: CloudOpsBenchCase, path: str) -> list[dict[str, Any]]:
    normalized = normalize_trajectory(root, case, path)
    return [step for step in normalized["replay_steps"] if step["replay_status"] == EXECUTABLE_REPLAY_STATUS]


def build_evidence_package(root: Path, case: CloudOpsBenchCase) -> dict[str, Any]:
    """Build one target-blind package using a fixed path1-preferred policy."""

    path1 = _executable_steps(root, case, "golden_path1")
    path2 = _executable_steps(root, case, "golden_path2")
    if path1:
        selected_path, steps, fallback_used = "golden_path1", path1, False
    else:
        selected_path, steps, fallback_used = "golden_path2", path2, True
    original_evidence = [{"tool": str(step["tool_id"]), "observation": str(step["observation"]["text"])} for step in steps]
    query = _sanitized_query(case)

    def render(items: list[dict[str, str]]) -> str:
        return json.dumps({"incident_request": query, "evidence": items}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    evidence = original_evidence
    rendered = render(evidence)
    # A fixed source-derived budget is applied before future model use. If
    # needed, every observation receives the same deterministic line-package
    # cap found by binary search; no target/process/model signal affects it.
    if len(rendered) > PACKAGE_BUDGET_CHARS and evidence:
        low, high = 0, max(len(item["observation"]) for item in evidence)
        while low < high:
            candidate = (low + high + 1) // 2
            trial = [{"tool": item["tool"], "observation": _uniform_line_package(item["observation"], candidate)["text"]} for item in original_evidence]
            if len(render(trial)) <= PACKAGE_BUDGET_CHARS:
                low = candidate
            else:
                high = candidate - 1
        evidence = [{"tool": item["tool"], "observation": _uniform_line_package(item["observation"], low)["text"]} for item in original_evidence]
        rendered = render(evidence)
    model_input = {"incident_request": query, "evidence": evidence}
    return {
        "model_input": model_input,
        "selection": {"policy": "PATH1_PREFERRED_WITH_PATH2_FALLBACK_ONLY_WHEN_PATH1_HAS_ZERO_EXECUTABLE_STEPS", "selected_path": selected_path, "fallback_used": fallback_used, "golden_path1_executable_steps": len(path1), "golden_path2_executable_steps": len(path2)},
        "package_chars": len(rendered),
        "package_bytes": len(rendered.encode("utf-8")),
        "package_budget_chars": PACKAGE_BUDGET_CHARS,
        "original_evidence_steps": len(original_evidence),
        "retained_evidence_steps": len(evidence),
        "observations_truncated": any(left["observation"] != right["observation"] for left, right in zip(original_evidence, evidence)),
        "target_blind": True,
        "process_labels_in_model_input": False,
        "golden_answers_in_model_input": False,
        "source_paths_in_model_input": False,
        "provenance_in_model_input": False,
    }


def build_evidence_packages(root: Path, cases: Sequence[CloudOpsBenchCase]) -> list[dict[str, Any]]:
    rows = []
    for index, case in enumerate(sorted(cases, key=source_case_group_id)):
        package = build_evidence_package(root, case)
        rows.append({"package_id": f"case-{index:04d}", **package})
    return rows


def _match_pattern(kind: str, value: Any, text: str) -> bool:
    if not isinstance(value, str):
        return False
    if kind == "regex":
        try:
            return re.search(value, text, re.DOTALL | re.MULTILINE) is not None
        except re.error:
            return False
    if kind == "code_snippet":
        return re.sub(r"\s+", " ", value).strip().casefold() in re.sub(r"\s+", " ", text).strip().casefold()
    return value.casefold() in text.casefold()


def evidence_sufficiency_audit(root: Path, cases: Sequence[CloudOpsBenchCase], packages: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    records = []
    case_counts: Counter[str] = Counter()
    for case in sorted(cases, key=source_case_group_id):
        group = source_case_group_id(case)
        payload = json.loads((root / case.references.process_label.relative_path).read_text(encoding="utf-8")) if case.references.process_label else {"milestones": []}
        observations = packages[group]["model_input"]["evidence"]
        total = retained = 0
        for milestone in payload.get("milestones", []):
            for admissible in milestone.get("admissible_tool_uses", []):
                tool = str(admissible.get("tool_name", ""))
                for pattern in admissible.get("evidence_patterns", []):
                    total += 1
                    hit = any(item["tool"] == tool and _match_pattern(str(pattern.get("kind", "literal")), pattern.get("value"), item["observation"]) for item in observations)
                    retained += hit
                    records.append({"source_case_group": group, "milestone": milestone.get("id"), "tool": tool, "pattern_kind": pattern.get("kind"), "retained": hit})
        case_counts["full" if total and retained == total else "partial" if retained else "zero"] += 1
        records.append({"source_case_group": group, "case_summary": True, "total_patterns": total, "retained_patterns": retained, "missing_patterns": total - retained, "coverage": retained / total if total else None, "coverage_class": "full" if total and retained == total else "partial" if retained else "zero", "target_blind_packager": True})
    pattern_rows = [row for row in records if row.get("case_summary")]
    return {"version": DECOMPOSITION_VERSION, "auditor_only_process_labels": True, "packager_used_process_labels": False, "total_process_label_evidence_patterns": sum(row["total_patterns"] for row in pattern_rows), "retained_evidence_patterns": sum(row["retained_patterns"] for row in pattern_rows), "missing_evidence_patterns": sum(row["missing_patterns"] for row in pattern_rows), "retention_rate": sum(row["retained_patterns"] for row in pattern_rows) / sum(row["total_patterns"] for row in pattern_rows) if sum(row["total_patterns"] for row in pattern_rows) else 1.0, "cases_with_full_coverage": case_counts["full"], "cases_with_partial_coverage": case_counts["partial"], "cases_with_zero_coverage": case_counts["zero"], "records": records}


def _stats(values: Sequence[int]) -> dict[str, Any]:
    ordered = sorted(int(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    def at(percentile: float) -> int:
        return ordered[round((len(ordered) - 1) * percentile)]
    return {"count": len(ordered), "min": ordered[0], "median": at(.5), "p75": at(.75), "p90": at(.9), "p95": at(.95), "p99": at(.99), "max": ordered[-1]}


def context_size_audit(packages: Sequence[Mapping[str, Any]], *, tokenizer: Any = None, prior_full_history: Mapping[str, Any] | None = None, registry: Mapping[str, Any] | None = None) -> dict[str, Any]:
    def text_for(task: str, package: Mapping[str, Any], category: str | None = None) -> str:
        evidence = package["model_input"]
        if task == "TASK_A_CATEGORY":
            contract = {"task": task, "allowed_categories": registry["ordered_categories"]}
        elif task == "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE":
            contract = {"task": task, "fault_category_context": category, "allowed_root_causes": registry["categories"][category]}
        elif task == "TASK_C_HIERARCHICAL_SELF_PREDICTED":
            contract = {"task": task, "stage1_categories": registry["ordered_categories"], "stage2_candidate_sets": registry["categories"]}
        else:
            contract = {"task": task}
        return json.dumps({"contract": contract, "input": evidence}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    result: dict[str, Any] = {"version": DECOMPOSITION_VERSION, "measurement": "serialized deterministic task input; no model inference", "tasks": {}}
    for task in ("TASK_A_CATEGORY", "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE", "TASK_C_HIERARCHICAL_SELF_PREDICTED", "TASK_D_FAULT_OBJECT"):
        texts = []
        for package in packages:
            category = None
            if task == "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE":
                category = str(package.get("audit_category"))
            texts.append(text_for(task, package, category))
        values = {"chars": [len(text) for text in texts], "bytes": [len(text.encode("utf-8")) for text in texts]}
        summary = {unit: _stats(items) for unit, items in values.items()}
        if tokenizer is not None:
            summary["tokens"] = _stats([len(tokenizer(text, add_special_tokens=False)["input_ids"]) for text in texts])
        result["tasks"][task] = summary
    if prior_full_history:
        prior = prior_full_history.get("history_policy_comparison", {}).get("FULL_HISTORY", {}).get("all", {})
        result["prior_full_history"] = {"characters": prior.get("characters"), "bytes": prior.get("bytes")}
        result["reduction_vs_prior_full_history"] = {"median_chars_fraction": result["tasks"]["TASK_A_CATEGORY"]["chars"]["median"] / prior["characters"]["median"] if prior.get("characters", {}).get("median") else None, "max_chars_fraction": result["tasks"]["TASK_A_CATEGORY"]["chars"]["max"] / prior["characters"]["max"] if prior.get("characters", {}).get("max") else None}
    return result


def trivial_baselines(registry: Mapping[str, Any], screen_cases: Sequence[CloudOpsBenchCase], train_cases: Sequence[CloudOpsBenchCase], *, seed: int = 20260303) -> dict[str, Any]:
    rng = random.Random(seed)
    categories = list(registry["ordered_categories"])
    train_category_counts = Counter(case.source_fault_category for case in train_cases)
    train_type_counts = Counter(case.source_fault_type for case in train_cases)
    majority_category = min(categories, key=lambda item: (-train_category_counts[item], item))
    majority_within = {category: min(registry["categories"][category], key=lambda item: (-train_type_counts[item], item)) for category in categories}
    a_random = [rng.choice(categories) for _ in screen_cases]
    b_random = [rng.choice(registry["categories"][case.source_fault_category]) for case in screen_cases]
    c_random_categories = [rng.choice(categories) for _ in screen_cases]
    c_random_types = [rng.choice(registry["categories"][category]) for category in c_random_categories]
    def exact(values: Sequence[str], targets: Sequence[str]) -> int:
        return sum(left == right for left, right in zip(values, targets))
    target_categories = [case.source_fault_category for case in screen_cases]
    target_types = [case.source_fault_type for case in screen_cases]
    return {"version": DECOMPOSITION_VERSION, "seed": seed, "screen_count": len(screen_cases), "method": "fixed-seed deterministic random draws plus TRAIN-derived majority; no model run", "TASK_A": {"uniform_random_expected_accuracy": 1 / len(categories), "uniform_random_screen_exact": exact(a_random, target_categories) / len(screen_cases), "majority_category": majority_category, "majority_screen_exact": sum(majority_category == target for target in target_categories) / len(screen_cases)}, "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE": {"uniform_random_expected_accuracy": sum(1 / len(registry["categories"][case.source_fault_category]) for case in screen_cases) / len(screen_cases), "uniform_random_screen_exact": exact(b_random, target_types) / len(screen_cases), "majority_within_category": majority_within, "majority_screen_exact": sum(majority_within[case.source_fault_category] == case.source_fault_type for case in screen_cases) / len(screen_cases)}, "TASK_C_HIERARCHICAL_SELF_PREDICTED": {"uniform_random_category_expected_accuracy": 1 / len(categories), "uniform_random_hierarchical_joint_expected_accuracy": sum(1 / (len(categories) * len(registry["categories"][case.source_fault_category])) for case in screen_cases) / len(screen_cases), "uniform_random_category_screen_exact": exact(c_random_categories, target_categories) / len(screen_cases), "uniform_random_root_screen_exact": exact(c_random_types, target_types) / len(screen_cases), "uniform_random_joint_screen_exact": sum(a == b and c == d for a, b, c, d in zip(c_random_categories, target_categories, c_random_types, target_types)) / len(screen_cases), "majority_category": majority_category, "majority_root_given_majority_category": majority_within[majority_category]}, "TASK_D_FAULT_OBJECT": {"baseline": "not defined: open-resource identity has no fixed closed vocabulary; use strict and deterministic normalized exact only"}}


def label_support(cases: Sequence[CloudOpsBenchCase], split_by_group: Mapping[str, str], registry: Mapping[str, Any]) -> dict[str, Any]:
    train = [case for case in cases if split_by_group.get(source_case_group_id(case)) == "TRAIN"]
    all_types = registry["fault_type_to_category"]
    type_counts = Counter(case.source_fault_type for case in train)
    category_counts = Counter(case.source_fault_category for case in train)
    return {"version": DECOMPOSITION_VERSION, "train_case_count": len(train), "by_category": {category: category_counts[category] for category in registry["ordered_categories"]}, "by_root_cause": {fault_type: {"category": all_types[fault_type], "count": type_counts[fault_type], "support_class": "zero" if type_counts[fault_type] == 0 else "low" if type_counts[fault_type] < 3 else "supported"} for fault_type in sorted(all_types)}, "by_system": dict(sorted(Counter(case.source_system for case in train).items())), "by_difficulty": dict(sorted(Counter(str(case.upstream_difficulty) for case in train).items())), "minimum_examples_per_root_cause": min(type_counts[fault_type] for fault_type in all_types), "median_examples_per_root_cause": sorted(type_counts[fault_type] for fault_type in all_types)[len(all_types) // 2], "maximum_examples_per_root_cause": max(type_counts[fault_type] for fault_type in all_types), "low_support_threshold": "<3 TRAIN examples; descriptive audit only, no oversampling"}


def future_record_counts(cases: Sequence[CloudOpsBenchCase], split_by_group: Mapping[str, str]) -> dict[str, Any]:
    train = [case for case in cases if split_by_group.get(source_case_group_id(case)) == "TRAIN"]
    valid_object = sum(bool(case.ground_truth_metadata.get("component")) for case in train)
    return {"version": DECOMPOSITION_VERSION, "path_view_policy": "one preferred evidence view per case; alternate path is a future ablation, not automatic duplication", "TASK_A_CATEGORY": {"deterministic_records": len(train)}, "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE": {"deterministic_records": len(train), "label": "ORACLE_CATEGORY_ROOT_CAUSE"}, "TASK_C_HIERARCHICAL_SELF_PREDICTED": {"deterministic_records_now": 0, "future_records_per_train_case": 1, "note": "Stage-1 predictions are not manufactured before a model run"}, "TASK_D_FAULT_OBJECT": {"deterministic_records": valid_object}, "multi_view_automatic_doubling": False}
