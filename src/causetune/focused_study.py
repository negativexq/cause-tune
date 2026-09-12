"""Predeclared one-intervention study contracts for Experiment 04."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Mapping

from .evidence import sha256_file


class StudyContractError(ValueError):
    """Raised when a focused optimization study is scientifically ambiguous."""


INTERVENTIONS = {"data_fraction", "lora_rank", "learning_rate"}


def _fingerprint_rows(rows: list[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_study_contract(
    *,
    study_id: str,
    intervention: str,
    levels: tuple[Any, ...],
    train_path: str | Path,
    validation_path: str | Path,
    benchmark_fingerprint: str,
    seed: int = 20260941,
    controlled: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not study_id.strip():
        raise StudyContractError("study_id must not be empty")
    if intervention not in INTERVENTIONS:
        raise StudyContractError("study must predeclare exactly one supported intervention")
    if not levels or len(set(levels)) != len(levels):
        raise StudyContractError("study levels must be non-empty and unique")
    if not benchmark_fingerprint:
        raise StudyContractError("blind benchmark fingerprint is required")
    controlled_values = dict(controlled or {})
    if intervention in controlled_values:
        raise StudyContractError("intervention cannot also appear in controlled variables")
    return {
        "schema_version": 1,
        "study_id": study_id,
        "primary_intervention": intervention,
        "levels": list(levels),
        "controlled": controlled_values,
        "data": {
            "train": str(train_path),
            "validation": str(validation_path),
            "benchmark_fingerprint": benchmark_fingerprint,
        },
        "selection": {
            "checkpoint_source": "validation",
            "benchmark_used_for_selection": False,
            "tuning_on_blind_benchmark": False,
        },
        "seed": seed,
        "raw_predictions_required": True,
        "evidence_verification_required": True,
        "negative_results_retained": True,
    }


def data_efficiency_study(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    benchmark_fingerprint: str,
    seed: int = 20260941,
) -> dict[str, Any]:
    return build_study_contract(
        study_id="e04a-data-efficiency",
        intervention="data_fraction",
        levels=(0.25, 0.5, 0.75, 1.0),
        train_path=train_path,
        validation_path=validation_path,
        benchmark_fingerprint=benchmark_fingerprint,
        seed=seed,
        controlled={"lora_rank": 16, "learning_rate": 2e-4, "max_steps_policy": "same_budget_rule"},
    )


def lora_capacity_study(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    benchmark_fingerprint: str,
    seed: int = 20260941,
) -> dict[str, Any]:
    return build_study_contract(
        study_id="e04b-lora-capacity",
        intervention="lora_rank",
        levels=(8, 16, 32),
        train_path=train_path,
        validation_path=validation_path,
        benchmark_fingerprint=benchmark_fingerprint,
        seed=seed,
        controlled={"data_fraction": 1.0, "learning_rate": 2e-4},
    )


def learning_rate_study(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    benchmark_fingerprint: str,
    seed: int = 20260941,
) -> dict[str, Any]:
    return build_study_contract(
        study_id="e04c-learning-rate",
        intervention="learning_rate",
        levels=(1e-4, 2e-4, 4e-4),
        train_path=train_path,
        validation_path=validation_path,
        benchmark_fingerprint=benchmark_fingerprint,
        seed=seed,
        controlled={"data_fraction": 1.0, "lora_rank": 16},
    )


def select_data_fraction(path: str | Path, fraction: float, *, seed: int) -> list[dict[str, Any]]:
    if not 0 < fraction <= 1:
        raise StudyContractError("data fraction must be in (0, 1]")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        family = str(row.get("metadata", {}).get("failure_mode", row.get("label", "unknown")))
        grouped.setdefault(family, []).append(row)
    selected: list[dict[str, Any]] = []
    for family in sorted(grouped):
        candidates = list(grouped[family])
        random.Random(seed + sum(ord(char) for char in family)).shuffle(candidates)
        count = max(1, round(len(candidates) * fraction))
        selected.extend(candidates[:count])
    return sorted(selected, key=lambda row: str(row.get("incident_id", row.get("example_id", ""))))


def validate_study_contract(contract: Mapping[str, Any]) -> None:
    required = {
        "schema_version", "study_id", "primary_intervention", "levels", "controlled", "data",
        "selection", "seed", "raw_predictions_required", "evidence_verification_required", "negative_results_retained",
    }
    if set(contract) != required:
        raise StudyContractError("study contract has unexpected or missing fields")
    intervention = contract["primary_intervention"]
    if intervention not in INTERVENTIONS:
        raise StudyContractError("unsupported primary intervention")
    if intervention in contract["controlled"]:
        raise StudyContractError("primary intervention is not controlled independently")
    selection = contract["selection"]
    if selection != {
        "checkpoint_source": "validation",
        "benchmark_used_for_selection": False,
        "tuning_on_blind_benchmark": False,
    }:
        raise StudyContractError("study selection boundary is not validation-only")
    if not all(contract[key] is True for key in ("raw_predictions_required", "evidence_verification_required", "negative_results_retained")):
        raise StudyContractError("study must retain raw predictions, verification and negative results")


def write_study_contract(contract: Mapping[str, Any], path: str | Path) -> None:
    validate_study_contract(contract)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(contract, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
