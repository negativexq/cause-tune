"""Offline tests for 03D.1 replay and context integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.cloudopsbench import (
    build_duplicate_report,
    build_split_manifest,
    classify_fallbacks,
    cumulative_context_audit,
    packaging_omission_audit,
    replay_parity,
    representation_manifest,
    scan_corpus,
    source_case_group_id,
    token_budget_projection,
    validate_split_manifest,
)


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value), encoding="utf-8")


@pytest.fixture()
def replay_fixture(tmp_path: Path) -> Path:
    base = tmp_path / "benchmark" / "boutique" / "Runtime_Fault" / "1"
    _write(base / "metadata.json", {"query": "Service quality degradation.", "difficulty": "medium", "result": {"fault_taxonomy": "Runtime_Fault", "fault_object": "pod/api", "root_cause": "pod_cpu_overload"}})
    _write(base / "tool_cache.json", {"GetResources:{}": "pod api Running\n"})
    _write(base / "raw_data" / "k8s_states.json", {"items": []})
    _write(base / "raw_data" / "logs.json", {"api": ["error one\n"]})
    _write(base / "raw_data" / "alert.json", [])
    _write(tmp_path / "process-label/boutique/Runtime_Fault/1/milestone.json", {"milestones": []})
    trace = {"statistics": {}, "diagnostic_trace": [{"tool_name": "GetResources", "calling": "tool_name='GetResources' arguments={}", "output": "pod api Running\n"}]}
    _write(tmp_path / "golden-trajectory/boutique/Runtime_Fault/1/path1.json", trace)
    _write(tmp_path / "golden-trajectory/boutique/Runtime_Fault/1/path2.json", trace)
    return tmp_path


def test_exact_replay_and_representation_are_deterministic(replay_fixture: Path) -> None:
    scan = scan_corpus(replay_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    split = build_split_manifest(scan, duplicate)
    fallback = classify_fallbacks(scan.root, scan, split)
    parity = replay_parity(scan.root, scan, split, fallback)
    assert fallback["original_fallback_count"] == 0
    assert parity["counts"]["EXACT_REPLAY"] == 2
    assert parity["counts"]["UNRESOLVED"] == 0
    representation = representation_manifest(scan, split, fallback, frozen_split_fingerprint=split["fingerprint"])
    assert representation["representation_fingerprint"] == representation_manifest(scan, split, fallback, frozen_split_fingerprint=split["fingerprint"])["representation_fingerprint"]


def test_golden_only_fallback_is_explicit_and_excluded(replay_fixture: Path) -> None:
    path = replay_fixture / "golden-trajectory/boutique/Runtime_Fault/1/path1.json"
    trace = json.loads(path.read_text())
    trace["diagnostic_trace"][0] = {"tool_name": "GetRecentLogs", "calling": "tool_name='GetRecentLogs' arguments={namespace: 'boutique', service_name: 'missing'}", "output": "only in golden"}
    path.write_text(json.dumps(trace), encoding="utf-8")
    scan = scan_corpus(replay_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    split = build_split_manifest(scan, duplicate)
    fallback = classify_fallbacks(scan.root, scan, split)
    assert fallback["counts"]["reason"]["GOLDEN_ONLY_NOT_REPLAYABLE"] == 1
    representation = representation_manifest(scan, split, fallback, frozen_split_fingerprint=split["fingerprint"])
    assert representation["eligible_trajectory_count"] == 1
    assert representation["auxiliary_trajectory_count"] == 1


def test_context_and_budget_projection_are_model_independent(replay_fixture: Path) -> None:
    scan = scan_corpus(replay_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    split = build_split_manifest(scan, duplicate)
    context = cumulative_context_audit(scan.root, scan, split)
    projection = token_budget_projection(context)
    assert context["per_final_diagnosis_action"]["all"]["count"] == 2
    assert "p75" in context["per_complete_trajectory"]["all"]["characters"]
    assert projection["method"].startswith("deterministic")


def test_packaging_audit_is_target_blind_and_retains_small_observation(replay_fixture: Path) -> None:
    scan = scan_corpus(replay_fixture)
    duplicate = build_duplicate_report(scan.root, scan.cases)
    split = build_split_manifest(scan, duplicate)
    audit = packaging_omission_audit(scan.root, scan, split)
    assert audit["omission_count"] == 0
    assert audit["budget_chars"] == 32768


def test_split_identity_includes_category_and_validation_rejects_unknown_case(replay_fixture: Path) -> None:
    scan = scan_corpus(replay_fixture)
    assert source_case_group_id(scan.cases[0]).count(":") == 3
    duplicate = build_duplicate_report(scan.root, scan.cases)
    split = build_split_manifest(scan, duplicate)
    result = validate_split_manifest(split, duplicate, ("not-the-source-case",))
    assert result["status"] == "fail"
