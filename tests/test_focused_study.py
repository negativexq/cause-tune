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
