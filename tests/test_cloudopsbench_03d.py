"""Offline tests for the frozen 03D training contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.cloudopsbench import (
    build_duplicate_report,
    build_packaging_audit,
    build_split_manifest,
    build_static_view_manifest,
    build_trajectory_example_manifest,
    freeze_contract,
    model_visible_schema,
    normalize_trajectory,
    scan_corpus,
    source_case_group_id,
    source_case_derivation_graph,
    split_distribution,
    supervision_policy,
    target_contract,
    validate_split_manifest,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _make_case(root: Path, case_id: str, fault_type: str, category: str = "Runtime_Fault", *, large: bool = False) -> None:
    base = root / "benchmark" / "boutique" / category / case_id
    output = ("line\n" * 40000) if large else "pod api Running\n"
    _write(base / "metadata.json", {"namespace": "boutique", "query": "Service quality degradation.", "difficulty": "medium", "result": {"fault_taxonomy": category, "fault_object": "pod/api", "root_cause": fault_type}})
    _write(base / "tool_cache.json", {"GetResources:{}": output})
    _write(base / "raw_data" / "k8s_states.json", {"items": [{"kind": "Pod", "metadata": {"name": "api"}}]})
    _write(base / "raw_data" / "logs.json", [{"message": "request failed"}])
    _write(base / "raw_data" / "alert.json", [{"name": "latency"}])
    _write(root / "process-label" / "boutique" / category / case_id / "milestone.json", {"milestones": [{"id": "M1"}]})
    trace = {"statistics": {"total_trace_steps": 1}, "diagnostic_trace": [{"tool_name": "GetResources", "calling": "tool_name='GetResources' arguments={}", "output": output}]}
    _write(root / "golden-trajectory" / "boutique" / category / case_id / "path1.json", trace)
    _write(root / "golden-trajectory" / "boutique" / category / case_id / "path2.json", trace)


@pytest.fixture()
def training_fixture(tmp_path: Path) -> Path:
    for index in range(9):
        _make_case(tmp_path, str(index), f"fault_{index % 3}", "Runtime_Fault" if index % 2 else "Scheduling_Fault")
    return tmp_path


def test_target_and_model_visible_contract_have_a_hard_authority_boundary() -> None:
    assert target_contract()["native_taxonomy_authoritative"] is True
    schema = model_visible_schema()
    assert schema["initial_context"]["source_labels"] is False
    assert schema["initial_context"]["process_labels"] is False
    assert "metadata.result.*" in schema["forbidden_input_fields"]
    assert supervision_policy()["reasoning_fields"]["disposition"] == "DROP"


def test_source_case_group_contains_full_case_path(training_fixture: Path) -> None:
    case = scan_corpus(training_fixture).cases[0]
    assert source_case_group_id(case).endswith(f":{case.source_fault_category}:{case.source_case_id}")
    assert source_case_group_id(case) != case.grouped_case_id


def test_deterministic_split_and_both_paths_stay_together(training_fixture: Path) -> None:
    scan = scan_corpus(training_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    first = build_split_manifest(scan, duplicate)
    second = build_split_manifest(scan, duplicate)
    assert first == second
    assert validate_split_manifest(first, duplicate, (source_case_group_id(case) for case in scan.cases))["status"] == "pass"
    rows = build_trajectory_example_manifest(scan.root, scan, first)
    by_case = {}
    for row in rows:
        by_case.setdefault(row["source_case_group"], set()).add(row["split"])
    assert all(len(splits) == 1 for splits in by_case.values())
    assert len(rows) == 18


def test_packaging_is_bounded_and_deterministic(training_fixture: Path) -> None:
    _make_case(training_fixture, "large", "fault_large", large=True)
    scan = scan_corpus(training_fixture)
    case = next(item for item in scan.cases if item.source_case_id == "large")
    normalized = normalize_trajectory(scan.root, case, "golden_path1")
    observation = normalized["replay_steps"][0]["observation"]
    assert observation["omitted"] is True
    assert observation["packaged_chars"] <= 32768
    assert normalize_trajectory(scan.root, case, "golden_path1") == normalized


def test_static_manifest_excludes_targets(training_fixture: Path) -> None:
    scan = scan_corpus(training_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    manifest = build_split_manifest(scan, duplicate)
    rows = build_static_view_manifest(scan, manifest)
    assert len(rows) == 9
    assert all(row["target_included"] is False for row in rows)
    assert all("source_fault_type" not in row["initial_context"] for row in rows)


def test_target_and_modality_semantics_are_deterministic(training_fixture: Path) -> None:
    scan = scan_corpus(training_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    manifest = build_split_manifest(scan, duplicate)
    case = scan.cases[0]
    normalized = normalize_trajectory(scan.root, case, "golden_path1")
    assert normalized["target"] == {"native_fault_type": case.source_fault_type, "native_fault_category": case.source_fault_category, "fault_object": "pod/api"}
    static = build_static_view_manifest(scan, manifest)
    assert static[0]["modalities"]["metrics"]["state"] == "MODALITY_NOT_AVAILABLE"
    assert static[0]["modalities"]["code"]["state"] == "MODALITY_NOT_AVAILABLE"
    graph = source_case_derivation_graph(scan, manifest)
    assert len(graph["edges"]) == len(scan.cases)
    assert all(any(item.endswith(":future_renderer_augmentation") for item in edge["descendants"]) for edge in graph["edges"])


def test_unknown_tool_fails_closed(training_fixture: Path) -> None:
    path = training_fixture / "golden-trajectory/boutique/Runtime_Fault/1/path1.json"
    payload = json.loads(path.read_text())
    payload["diagnostic_trace"][0]["tool_name"] = "InventDiagnosis"
    path.write_text(json.dumps(payload), encoding="utf-8")
    scan = scan_corpus(training_fixture)
    case = next(item for item in scan.cases if item.source_case_id == "1")
    with pytest.raises(ValueError, match="unknown golden tool"):
        normalize_trajectory(scan.root, case, "golden_path1")


def test_split_validator_rejects_cross_split_contamination() -> None:
    duplicate = {"contamination_case_groups": [["a", "b"]]}
    manifest = {"entries": [{"source_case_group": "a", "split": "TRAIN"}, {"source_case_group": "b", "split": "TEST"}]}
    result = validate_split_manifest(manifest, duplicate, ("a", "b"))
    assert result["status"] == "fail"


def test_freeze_fingerprint_changes_when_contract_changes(training_fixture: Path) -> None:
    scan = scan_corpus(training_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    manifest = build_split_manifest(scan, duplicate)
    first = freeze_contract(scan, manifest, duplicate)["component_fingerprint"]
    changed = dict(manifest)
    changed["test_protected"] = False
    changed["fingerprint"] = "changed"
    second = freeze_contract(scan, changed, duplicate)["component_fingerprint"]
    assert first != second


def test_packaging_audit_is_manifest_only(training_fixture: Path) -> None:
    scan = scan_corpus(training_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    manifest = build_split_manifest(scan, duplicate)
    rows = build_trajectory_example_manifest(scan.root, scan, manifest)
    audit = build_packaging_audit(rows)
    assert audit["trajectory_count"] == 18
    assert audit["full_raw_snapshot_primary"] is False
