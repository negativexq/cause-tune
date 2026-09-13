from __future__ import annotations

import json
import math
from pathlib import Path

from causetune.evidence import sha256_path
from causetune.verify import score_incident_predictions


ROOT = Path("results/experiment_05")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_metrics_close(expected, actual) -> None:
    if isinstance(expected, float) and isinstance(actual, float):
        assert math.isclose(expected, actual, rel_tol=0.0, abs_tol=1e-15)
    elif isinstance(expected, dict) and isinstance(actual, dict):
        assert expected.keys() == actual.keys()
        for key in expected:
            _assert_metrics_close(expected[key], actual[key])
    elif isinstance(expected, list) and isinstance(actual, list):
        assert len(expected) == len(actual)
        for left, right in zip(expected, actual):
            _assert_metrics_close(left, right)
    else:
        assert expected == actual


def test_e05_gate_and_frozen_identity() -> None:
    summary = _json(ROOT / "g05b_summary.json")
    assert summary["status"] == "PASS"
    assert summary["systems"] == ["base", "e02", "e04"]
    assert summary["one_shot_per_system"] is True
    assert summary["fresh_reload_per_system"] is True
    assert summary["e03_used"] is False
    assert summary["semantic_generation_rerun_during_finalization"] is False
    assert summary["benchmark_fingerprint"] == _json(ROOT / "protocol.json")["benchmark_fingerprint"]


def test_e05_persisted_predictions_reproduce_and_are_hashed() -> None:
    index = _json(ROOT / "artifact_hashes.json")
    actual = {
        path.relative_to(ROOT).as_posix(): sha256_path(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    assert index["artifacts"] == actual
    for name in ("base", "e02", "e04"):
        prediction_path = ROOT / "evaluations" / name / "predictions.jsonl"
        persisted = _json(ROOT / "evaluations" / name / "evaluation.json")["metrics"]
        assert sum(1 for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()) == 120
        _assert_metrics_close(score_incident_predictions(prediction_path), persisted)


def test_e05_transition_accounting_is_complete() -> None:
    transitions = _json(ROOT / "transition_analysis.json")
    for value in transitions.values():
        assert value["count"] == 120
        assert sum(value["counts"].values()) == 120
        assert len(value["rows"]) == 120
