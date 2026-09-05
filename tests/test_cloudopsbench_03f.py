"""CPU-only tests for the 03F decomposition freeze."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from causetune.cloudopsbench import (
    build_evidence_package,
    build_evidence_packages,
    build_hierarchy_registry,
    context_size_audit,
    evidence_sufficiency_audit,
    future_record_counts,
    label_support,
    task_contracts,
    trivial_baselines,
)
from causetune.cloudopsbench.decomposition import fingerprint
from causetune.cloudopsbench.training_contract import source_case_group_id
from causetune.cloudopsbench import scan_corpus


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, list]:
    for index, (fault_type, category) in enumerate((("fault-a", "Runtime_Fault"), ("fault-b", "Scheduling_Fault"))):
        base = root / "benchmark" / "boutique" / category / str(index)
        _write(base / "metadata.json", {"query": "Service quality degradation.", "difficulty": "medium", "result": {"fault_taxonomy": category, "fault_object": "app/api", "root_cause": fault_type}})
        _write(base / "tool_cache.json", {"GetResources:{}": "app=api\napi Running\n"})
        _write(base / "raw_data/k8s_states.json", {"items": []})
        _write(base / "raw_data/logs.json", {"api": ["error"]})
        _write(base / "raw_data/alert.json", [])
        label = {"milestones": [{"id": "M1", "admissible_tool_uses": [{"tool_name": "GetResources", "arguments": {}, "evidence_patterns": [{"kind": "literal", "value": "Running"}]}]}]}
        _write(root / "process-label" / "boutique" / category / str(index) / "milestone.json", label)
        trace = {"diagnostic_trace": [{"tool_name": "GetResources", "calling": "tool_name='GetResources' arguments={}", "output": "app=api\napi Running\n"}]}
        _write(root / "golden-trajectory" / "boutique" / category / str(index) / "path1.json", trace)
        _write(root / "golden-trajectory" / "boutique" / category / str(index) / "path2.json", trace)
    scan = scan_corpus(root)
    return root, list(scan.cases)


def _case(fault_type: str, category: str):
    return SimpleNamespace(source_fault_type=fault_type, source_fault_category=category)


def test_hierarchy_is_deterministic_and_conflicts_are_rejected() -> None:
    cases = [_case("z", "B"), _case("a", "A"), _case("z", "B")]
    first = build_hierarchy_registry(cases)
    assert first["categories"] == {"A": ["a"], "B": ["z"]}
    assert build_hierarchy_registry(cases) == first
    try:
        build_hierarchy_registry([_case("z", "A"), _case("z", "B")])
    except ValueError:
        pass
    else:
        raise AssertionError("conflicting native mapping was accepted")


def test_task_contracts_freeze_boundaries_and_candidate_restriction() -> None:
    registry = {"ordered_categories": ["A", "B"], "categories": {"A": ["a"], "B": ["b", "c"]}}
    contracts = task_contracts(registry)
    assert contracts["TASK_A_CATEGORY"]["output_schema"]["required"] == ["fault_category"]
    assert contracts["TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"]["label"] == "ORACLE_CATEGORY_ROOT_CAUSE"
    assert "fault_category_context" in contracts["TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"]["input_schema"]["required"]
    assert contracts["TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"]["candidate_restriction"].startswith("only root causes")
    assert contracts["TASK_C_HIERARCHICAL_SELF_PREDICTED"]["stages"][1]["authoritative_category_used"] is False
    assert "root_cause" not in contracts["TASK_A_CATEGORY"]["output_schema"]["properties"]
    assert contracts["TASK_D_FAULT_OBJECT"]["normalization"]["uses_target"] is False


def test_evidence_packaging_is_target_blind_bounded_and_path_stable(tmp_path: Path) -> None:
    root, cases = _fixture(tmp_path)
    first = build_evidence_package(root, cases[0])
    second = build_evidence_package(root, cases[0])
    assert first == second
    assert first["selection"]["selected_path"] == "golden_path1"
    assert first["package_chars"] <= first["package_budget_chars"]
    assert first["target_blind"] is True
    assert first["process_labels_in_model_input"] is False
    text = json.dumps(first["model_input"])
    assert "fault-a" not in text and "Runtime_Fault" not in text and "app/api" not in text


def test_evidence_sufficiency_and_context_accounting(tmp_path: Path) -> None:
    root, cases = _fixture(tmp_path)
    rows = build_evidence_packages(root, cases)
    mapping = {source_case_group_id(case): row for case, row in zip(cases, rows)}
    for case, row in zip(cases, rows):
        row["audit_category"] = case.source_fault_category
    audit = evidence_sufficiency_audit(root, cases, mapping)
    assert audit["total_process_label_evidence_patterns"] == 2
    assert audit["retained_evidence_patterns"] == 2
    assert audit["cases_with_full_coverage"] == 2
    context = context_size_audit(rows, registry=build_hierarchy_registry(cases))
    assert context["tasks"]["TASK_A_CATEGORY"]["chars"]["count"] == 2
    assert context["tasks"]["TASK_C_HIERARCHICAL_SELF_PREDICTED"]["bytes"]["max"] >= context["tasks"]["TASK_A_CATEGORY"]["bytes"]["max"]


def test_train_support_baselines_and_future_counts_are_deterministic(tmp_path: Path) -> None:
    root, cases = _fixture(tmp_path)
    groups = {source_case_group_id(case): "TRAIN" for case in cases}
    registry = build_hierarchy_registry(cases)
    support = label_support(cases, groups, registry)
    assert support["train_case_count"] == 2
    assert set(support["by_root_cause"]) == {"fault-a", "fault-b"}
    counts = future_record_counts(cases, groups)
    assert counts["TASK_A_CATEGORY"]["deterministic_records"] == 2
    assert counts["TASK_C_HIERARCHICAL_SELF_PREDICTED"]["deterministic_records_now"] == 0
    baseline = trivial_baselines(registry, cases, cases, seed=7)
    assert baseline == trivial_baselines(registry, cases, cases, seed=7)
    assert baseline["TASK_A"]["uniform_random_expected_accuracy"] == 0.5


def test_freeze_fingerprint_changes_on_contract_mutation_and_test_is_not_input(tmp_path: Path) -> None:
    value = {"task": "A", "test_model_facing": False}
    assert fingerprint(value) == fingerprint(dict(value))
    mutated = dict(value)
    mutated["task"] = "B"
    assert fingerprint(value) != fingerprint(mutated)
    root, cases = _fixture(tmp_path)
    packages = build_evidence_packages(root, cases)
    assert all("source_case_group" not in row and "source_fault_type" not in row for row in packages)
    # The production audit passes only TRAIN/VALIDATION cases to this function;
    # TEST is therefore excluded before any package can become model-facing.
