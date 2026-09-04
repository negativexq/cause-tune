"""Offline contract tests for the Cloud-OpsBench adoption audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.cloudopsbench import (
    CLOUD_OPSBENCH_REVISION,
    CloudOpsBenchSource,
    build_context_size_audit,
    build_corpus_census,
    build_golden_trajectory_audit,
    build_leakage_field_policy,
    build_modality_availability,
    build_split_feasibility,
    build_target_availability,
    build_taxonomy_mapping,
    scan_corpus,
)
from causetune.cloudopsbench.audit import audit_corpus
from causetune.cloudopsbench.scanner import CorpusScanError


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _make_case(root: Path, system: str, category: str, case_id: str, fault_type: str, *, metrics: bool, code: bool) -> None:
    base = root / "benchmark" / system / category / case_id
    _write(base / "metadata.json", {"fault_type": fault_type, "fault_category": category, "namespace": "default", "service": "api", "difficulty": "medium", "diagnosis": {"component": "api"}, "query": "diagnose this case"})
    _write(base / "tool_cache.json", {"get_logs": {"result": "observed"}})
    _write(base / "raw_data" / "alert.json", [{"name": "latency"}])
    _write(base / "raw_data" / "k8s_states.json", {"items": [{"kind": "Pod", "metadata": {"name": "api"}}]})
    _write(base / "raw_data" / "logs.json", [{"message": "request failed"}])
    if metrics:
        _write(base / "raw_data" / "metrics.csv", "time,value\n0,1\n")
    if code:
        _write(base / "code" / "main.py", "def handle():\n    return 1\n")
    _write(root / "process-label" / system / category / case_id / "milestone.json", {"milestones": [{"id": "observe"}]})
    _write(root / "golden-trajectory" / system / category / case_id / "path1.json", [{"tool_name": "get_logs"}, {"role": "assistant", "content": "The diagnosis is supported by logs."}])
    _write(root / "golden-trajectory" / system / category / case_id / "path2.json", [{"tool_name": "get_k8s"}, {"role": "assistant", "content": "The final answer is structured."}])


@pytest.fixture()
def cloudops_fixture(tmp_path: Path) -> Path:
    _make_case(tmp_path, "boutique", "runtime", "case_001", "ContainerMemoryLimitTooLow", metrics=True, code=True)
    _make_case(tmp_path, "trainticket", "performance", "case_002", "PodCPUOverload", metrics=False, code=False)
    return tmp_path


def test_source_registry_uses_pinned_upstream_revision() -> None:
    source = CloudOpsBenchSource()
    assert source.upstream_revision == CLOUD_OPSBENCH_REVISION
    assert source.license == "MIT"
    assert source.documented_system_case_counts == {"boutique": 550, "trainticket": 204}


def test_scanner_preserves_native_labels_and_optional_modalities(cloudops_fixture: Path) -> None:
    scan = scan_corpus(cloudops_fixture)
    assert len(scan.cases) == 2
    first, second = scan.cases
    assert first.source_fault_type == "ContainerMemoryLimitTooLow"
    assert first.references.code is not None
    assert second.references.code is None
    assert second.references.metrics is None
    assert first.source_revision == CLOUD_OPSBENCH_REVISION
    assert first.grouped_case_id.startswith(CLOUD_OPSBENCH_REVISION + ":")
    assert "ground_truth_metadata" not in first.model_visible_view()


def test_scanner_fingerprint_is_stable_and_changes_on_source_mutation(cloudops_fixture: Path) -> None:
    first = scan_corpus(cloudops_fixture).cases[0].source_fingerprint
    second = scan_corpus(cloudops_fixture).cases[0].source_fingerprint
    assert first == second
    (cloudops_fixture / "benchmark/boutique/runtime/case_001/raw_data/logs.json").write_text("changed", encoding="utf-8")
    assert scan_corpus(cloudops_fixture).cases[0].source_fingerprint != first


def test_missing_required_metadata_fails_closed(cloudops_fixture: Path) -> None:
    (cloudops_fixture / "benchmark/boutique/runtime/case_001/metadata.json").write_text("{", encoding="utf-8")
    with pytest.raises(CorpusScanError):
        scan_corpus(cloudops_fixture)


def test_audit_views_are_deterministic_and_report_optional_data(cloudops_fixture: Path) -> None:
    scan = scan_corpus(cloudops_fixture)
    census = build_corpus_census(scan)
    assert census["observed"]["total_cases"] == 2
    assert build_modality_availability(scan.cases)["counts"]["metrics"] == 1
    assert build_target_availability(scan.cases)["matrix"]["fault_type"]["available_cases"] == 2
    trajectories = build_golden_trajectory_audit(scan.root, scan.cases)
    assert trajectories["cases_with_both_paths"] == 2
    assert "get_logs" in trajectories["tool_name_counts"]
    context = build_context_size_audit(scan.root, scan.cases)
    assert context["lossy_truncation_applied"] is False
    assert build_split_feasibility(scan.cases)["status"].startswith("candidate")


def test_native_taxonomy_is_not_forced_into_causetune_ontology() -> None:
    mapping = build_taxonomy_mapping()
    assert len(mapping) == 57
    assert any(item.classification == "UNMAPPED" for item in mapping)
    assert any(item.native_fault_type == "db_connection_exhaustion" and item.normalized_causetune_fault_id == "db_connection_pool_exhaustion" for item in mapping)


def test_unknown_native_type_is_reported_fail_closed(cloudops_fixture: Path) -> None:
    metadata = cloudops_fixture / "benchmark/boutique/runtime/case_001/metadata.json"
    value = json.loads(metadata.read_text(encoding="utf-8"))
    value["fault_type"] = "UnannouncedFutureFault"
    metadata.write_text(json.dumps(value), encoding="utf-8")
    scan = scan_corpus(cloudops_fixture)
    report = audit_corpus(scan)
    assert report["status"] == "fail"
    assert "UnannouncedFutureFault" in report["taxonomy_integrity"]["unknown_observed_fault_types"]


def test_leakage_policy_keeps_source_truth_out_of_model_input() -> None:
    policy = build_leakage_field_policy()
    by_name = {item["field"]: item["classification"] for item in policy["fields"]}
    assert by_name["metadata fault fields and diagnosis"] == "SUPERVISION_ONLY"
    assert by_name["golden final answer/query fields"] == "FORBIDDEN_INPUT"


def test_source_contract_requires_real_root() -> None:
    with pytest.raises(ValueError, match="root is required"):
        scan_corpus(None)
