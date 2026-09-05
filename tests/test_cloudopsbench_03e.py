"""CPU-only tests for the 03E frozen screening contract."""

from __future__ import annotations

import pytest

from causetune.cloudopsbench.screening import (
    _fp,
    build_validation_subsplit,
    parse_final_diagnosis,
    parse_tool_call,
    validate_validation_subsplit,
)


def _split_manifest() -> dict:
    entries = []
    for index, fault_type in enumerate(("fault_a", "fault_b")):
        entries.extend(
            {
                "source_case_group": f"group-{fault_type}-{split}",
                "split": "VALIDATION",
                "source_fault_type": fault_type,
            }
            for split in ("a", "b")
        )
    entries.append({"source_case_group": "test-only", "split": "TEST", "source_fault_type": "fault_a"})
    return {"fingerprint": "frozen-split", "entries": entries}


def test_qwen_control_token_is_not_semantic_output() -> None:
    prediction, category = parse_final_diagnosis(
        '{"native_fault_type":"fault_a","native_fault_category":"cat","fault_object":"svc"}<|im_end|>'
    )
    assert category == "valid"
    assert prediction == {"native_fault_type": "fault_a", "native_fault_category": "cat", "fault_object": "svc"}


def test_tool_parser_preserves_strict_tool_contract() -> None:
    tool, arguments, category = parse_tool_call(
        '<tool_call><function=GetResources></function></tool_call><|im_end|>'
    )
    assert (tool, arguments, category) == ("GetResources", {}, "xml_tool_call")


def test_validation_screen_is_one_per_label_and_excludes_test() -> None:
    split = _split_manifest()
    manifest = build_validation_subsplit(split)
    assert manifest["screen_count"] == 2
    assert manifest["selection_count"] == 2
    assert set(manifest["screen_case_groups"]) == {"group-fault_a-a", "group-fault_b-a"}
    validate_validation_subsplit(manifest, split)
    assert "test-only" not in manifest["screen_case_groups"]


def test_validation_subsplit_rejects_mutation() -> None:
    split = _split_manifest()
    manifest = build_validation_subsplit(split)
    manifest["screen_case_groups"] = ["test-only", "group-fault_b-a"]
    with pytest.raises(ValueError):
        validate_validation_subsplit(manifest, split)


def test_screen_manifest_fingerprint_is_stable_and_mutation_changes_it() -> None:
    split = _split_manifest()
    first = build_validation_subsplit(split)
    second = build_validation_subsplit(split)
    assert first["fingerprint"] == second["fingerprint"]
    mutated = dict(first)
    mutated["selection_count"] += 1
    assert _fp({key: value for key, value in mutated.items() if key != "fingerprint"}) != first["fingerprint"]
