"""CPU-only tests for the 03F.1 evidence-integrity contract."""

from __future__ import annotations

import json
from pathlib import Path

from causetune.cloudopsbench.integrity_gate import (
    _budget_evidence,
    canonicalize_text,
    classify_case_reason,
    deduplicate_observations,
    matcher_result,
    sha256_value,
)


def test_zero_reason_taxonomy_is_deterministic() -> None:
    assert classify_case_reason([], [], path1_rows=[], path2_rows=[], union_rows=[], source_rows=[]) == "PROCESS_LABEL_NO_MATCHABLE_PATTERNS"
    pattern = {"tool": "GetResources", "kind": "literal", "value": "Pending"}
    rows = [{"strict_match": False, "matcher_schema_gap": False}]
    source = [{"strict_match": False, "direct_source_match": False, "trace_match": False, "matcher_schema_gap": False}]
    assert classify_case_reason([pattern], rows, path1_rows=rows, path2_rows=rows, union_rows=rows, source_rows=source) == "PROCESS_PATTERN_UNMATCHABLE_OR_EVIDENCE_SPARSE"


def test_matcher_canonicalization_is_mechanical_only() -> None:
    assert canonicalize_text("Runtime_Fault\n") == "runtime_fault"
    assert matcher_result("literal", "Runtime_Fault", "runtime fault") == {"strict": False, "canonical": False, "matcher_schema_gap": False, "invalid": False}
    assert matcher_result("literal", "runtime fault", "Runtime\nFault")["canonical"] is True
    assert matcher_result("literal", "pod unavailable", "pod ready")["canonical"] is False


def test_impossible_regex_boundary_is_a_schema_gap() -> None:
    pattern = r"(?m)^\S+\s+0/1\s+Pending\s+\S+\s+\S+\s+<none>\s+<none>\b"
    text = "pod 0/1 Pending 0 1s <none> <none> <none>"
    result = matcher_result("regex", pattern, text)
    assert result["strict"] is False
    assert result["matcher_schema_gap"] is True


def test_path_union_is_stable_and_deduplicated() -> None:
    left = [{"tool": "GetResources", "observation": "a"}, {"tool": "GetResources", "observation": "b"}]
    right = [{"tool": "GetResources", "observation": "b"}, {"tool": "GetAlerts", "observation": "c"}]
    assert deduplicate_observations([*left, *right]) == [*left, right[1]]
    assert deduplicate_observations([*left, *right]) == deduplicate_observations([*left, *right])


def test_budget_accounting_is_deterministic() -> None:
    evidence = [{"tool": "GetResources", "observation": "line\n" * 1000}, {"tool": "GetAlerts", "observation": "alert\n" * 1000}]
    bounded, rendered = _budget_evidence("Investigate.", evidence, 500)
    assert len(rendered) <= 500
    assert bounded == _budget_evidence("Investigate.", evidence, 500)[0]


def test_fingerprint_stability_and_mutation() -> None:
    payload = {"candidate": "R2", "budget": 30000, "dedup": "stable"}
    assert sha256_value(payload) == sha256_value({"dedup": "stable", "budget": 30000, "candidate": "R2"})
    assert sha256_value(payload) != sha256_value({**payload, "budget": 40000})


def test_packager_boundary_artifacts_are_target_blind() -> None:
    packages = Path("results/incident_telemetry_03f1/evidence_packages.jsonl")
    if not packages.exists():
        return
    first = json.loads(packages.read_text(encoding="utf-8").splitlines()[0])
    assert first["target_blind"] is True
    assert first["process_labels_in_model_input"] is False
    assert first["golden_answers_in_model_input"] is False
    assert first["source_paths_in_model_input"] is False
    assert first["provenance_in_model_input"] is False
    assert "root_cause" not in first["model_input"]
    assert "fault_category" not in first["model_input"]


def test_validation_screen_and_test_exclusion_are_frozen() -> None:
    path = Path("results/incident_telemetry_03f1/validation_screen_sufficiency.json")
    if not path.exists():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["screen_count"] == 57
    assert value["screen_membership_unchanged"]["set_equal"] is True
    assert value["diagnosis_input_validity"]["clear_package_failures"] == 0
    assert value["test_model_facing"] is False


def test_previous_03f_fingerprint_is_preserved() -> None:
    path = Path("results/incident_telemetry_03f1/evidence_representation_manifest.json")
    if not path.exists():
        return
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["03f_decomposition_fingerprint"] == "249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9"
