"""CPU-only integrity tests for the completed 03G screening artefacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
OUT = ROOT / "results" / "incident_telemetry_03g"


def _json(name: str):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_screen_and_prior_fingerprints_are_unchanged() -> None:
    contract = _json("experiment_contract.json")
    assert contract["screen_count"] == 57
    assert contract["test_model_facing"] is False
    assert contract["validation_selection_used"] is False
    assert contract["decomposition_fingerprint"] == "249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9"
    assert contract["evidence_representation_fingerprint"] == "b676b10a17d690f3796e237e942c340f2722a1aa5bcd97af1b43de1c0e972215"
    assert _json("artifact_fingerprints.json")["immutable_inputs"]["03f_decomposition_fingerprint"] == contract["decomposition_fingerprint"]
    assert _json("artifact_fingerprints.json")["immutable_inputs"]["03f1_representation_fingerprint"] == contract["evidence_representation_fingerprint"]


def test_each_task_has_exactly_the_frozen_screen_and_raw_outputs() -> None:
    for task in ("task_a", "task_b", "task_c", "task_d"):
        rows = [line for line in (OUT / task / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(rows) == 57
        assert _json(f"{task}/metrics.json")["count"] == 57


def test_task_c_reuses_task_a_and_task_b_preserves_oracle_boundary() -> None:
    task_c = _json("task_c/metrics.json")
    task_b_prompt = _json("task_b/prompt.json")
    assert task_c["stage1_source"] == "task_a/predictions.jsonl; no regeneration"
    assert task_b_prompt["oracle_category"] is True
    assert "ORACLE_CATEGORY_ROOT_CAUSE" in task_b_prompt["system"]


def test_baselines_are_frozen_and_screen_majority_tie_is_explicit() -> None:
    baseline = _json("baseline_verification.json")
    task_a = baseline["TASK_A"]
    assert task_a["majority_categories"] == ["Runtime_Fault", "Scheduling_Fault"]
    assert task_a["majority_tie"] is True
    assert task_a["majority_count"] == 12
    assert task_a["uniform_category_expected_rate"] == 0.125
    assert baseline["TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"]["uniform_within_category_expected_rate"] == 0.14035087719298245
    assert baseline["TASK_C_HIERARCHICAL_SELF_PREDICTED"]["hierarchical_random_expected_joint_rate"] == 0.017543859649122806


def test_no_test_or_provider_and_future_selection_is_not_training() -> None:
    selection = _json("specialization_selection.json")
    hardware = _json("hardware_metrics.json")
    assert selection["decision"] == "SELECT_TASK_B_FOR_QLORA"
    assert selection["qlora_started"] is False
    assert selection["training"] is False
    assert selection["provider_called"] is False
    assert selection["test_model_facing"] is False
    assert hardware["oom_count"] == 0
    assert hardware["context_overflow_count"] == 0


def test_raw_prediction_fingerprint_is_recorded() -> None:
    fingerprints = _json("artifact_fingerprints.json")["artifacts"]
    for task in ("task_a", "task_b", "task_c", "task_d"):
        path = OUT / task / "predictions.jsonl"
        assert fingerprints[f"{task}/predictions.jsonl"] == _sha256(path)
