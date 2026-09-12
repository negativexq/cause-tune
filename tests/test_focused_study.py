from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.focused_study import (
    StudyContractError,
    data_efficiency_study,
    learning_rate_study,
    lora_capacity_study,
    select_data_fraction,
    validate_study_contract,
)
from causetune.e04a_data import materialize_data_efficiency_subsets
from causetune.incident_training import TRAINING_GENERATOR_VERSION
from causetune.incident_taxonomy import FAILURE_FAMILIES, FAILURE_SPECS


def _kwargs(tmp_path: Path) -> dict:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    rows = [
        {"incident_id": f"i-{index}", "metadata": {"failure_mode": family}}
        for index, family in enumerate(("a", "a", "b", "b"))
    ]
    train.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    validation.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
    return {"train_path": train, "validation_path": validation, "benchmark_fingerprint": "blind-fp"}


def test_each_study_predeclares_one_intervention(tmp_path: Path) -> None:
    kwargs = _kwargs(tmp_path)
    for builder, intervention in (
        (data_efficiency_study, "data_fraction"),
        (lora_capacity_study, "lora_rank"),
        (learning_rate_study, "learning_rate"),
    ):
        contract = builder(**kwargs)
        validate_study_contract(contract)
        assert contract["primary_intervention"] == intervention
        assert contract["selection"]["benchmark_used_for_selection"] is False


def test_data_fraction_selection_is_deterministic_and_balanced(tmp_path: Path) -> None:
    kwargs = _kwargs(tmp_path)
    first = select_data_fraction(kwargs["train_path"], 0.5, seed=42)
    second = select_data_fraction(kwargs["train_path"], 0.5, seed=42)
    assert first == second
    assert len(first) == 2
    assert {row["metadata"]["failure_mode"] for row in first} == {"a", "b"}


def test_intervention_cannot_be_hidden_in_controlled_fields(tmp_path: Path) -> None:
    kwargs = _kwargs(tmp_path)
    contract = data_efficiency_study(**kwargs)
    contract["controlled"] = {"data_fraction": 1.0}
    with pytest.raises(StudyContractError):
        validate_study_contract(contract)
    contract = data_efficiency_study(**kwargs)
    contract["selection"]["tuning_on_blind_benchmark"] = True
    with pytest.raises(StudyContractError):
        validate_study_contract(contract)


def _write_training_fixture(path: Path) -> None:
    inputs = []
    truths = []
    for family_index, family in enumerate(FAILURE_FAMILIES):
        for index in range(4):
            incident_id = f"fixture-{family_index:02d}-{index}"
            component = f"svc-{family_index}"
            inputs.append({
                "incident_id": incident_id,
                "split": "train",
                "incident_packet": f"INCIDENT {incident_id}\nTOPOLOGY\n{component}\nRECENT CHANGES\nmarker family{chr(97 + family_index)}sample{chr(97 + index)}\nMETRICS\nM1 signal\nLOGS / EVENTS\nE1 signal\nALERTS\nA1 signal",
                "metadata": {
                    "difficulty": "standard" if index < 2 else "hard",
                    "topology_family": "topology-a" if index % 2 else "topology-b",
                    "red_herring": index >= 2,
                    "present_components": [component],
                    "evidence_ids": ["M1", "E1", "A1"],
                },
            })
            truths.append({
                "incident_id": incident_id,
                "culprit_service": component,
                "failure_mode": family,
                "recommended_action": FAILURE_SPECS[family].action,
                "evidence_ids": ["M1", "E1", "A1"],
                "metadata": {
                    "difficulty": inputs[-1]["metadata"]["difficulty"],
                    "failure_family": family,
                    "topology_family": inputs[-1]["metadata"]["topology_family"],
                    "red_herring": inputs[-1]["metadata"]["red_herring"],
                    "generator_version": TRAINING_GENERATOR_VERSION,
                },
            })
    path.mkdir(parents=True)
    (path / "train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in inputs), encoding="utf-8")
    (path / "ground_truth_train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in truths), encoding="utf-8")
    (path / "validation.jsonl").write_text(json.dumps(inputs[0]) + "\n", encoding="utf-8")
    (path / "ground_truth_validation.jsonl").write_text(json.dumps(truths[0]) + "\n", encoding="utf-8")


def test_e04a_subsets_are_nested_and_path_independent(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_training_fixture(source)
    first = tmp_path / "first"
    second = tmp_path / "second"
    result_a = materialize_data_efficiency_subsets(source, first, seed=42, fractions=(0.25, 0.5, 0.75, 1.0))
    result_b = materialize_data_efficiency_subsets(source, second, seed=42, fractions=(0.25, 0.5, 0.75, 1.0))
    assert [row["subset_hash"] for row in result_a["subsets"]] == [row["subset_hash"] for row in result_b["subsets"]]
    ids = []
    for subset in ("025pct", "050pct", "075pct", "100pct"):
        rows = [json.loads(line) for line in (first / subset / "train.jsonl").read_text().splitlines()]
        ids.append({row["incident_id"] for row in rows})
    assert ids[0] < ids[1] < ids[2] < ids[3]
    assert [len(item) for item in ids] == [12, 24, 36, 48]
