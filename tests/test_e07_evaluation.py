from __future__ import annotations

import json
from pathlib import Path

from scripts.run_e07_boundary import _score


ROOT = Path(__file__).resolve().parents[1]


def test_e07_evaluation_contract_uses_three_frozen_systems() -> None:
    protocol = json.loads((ROOT / "results/experiment_07/protocol.json").read_text())
    assert protocol["systems"] == ["base", "e02", "e04"]
    assert protocol["selection_exclusion"].startswith("evaluation-only")


def test_e07_invalid_sufficient_output_counts_as_false_abstention() -> None:
    records = [{
        "incident_id": "case-1",
        "metadata": {"boundary_category": "sufficient", "present_components": ["service-a"]},
        "incident_packet": "M1 E1 A1",
    }]
    truths = {"case-1": {
        "expected_status": "diagnose",
        "culprit_service": "service-a",
        "failure_mode": "memory_leak",
        "recommended_action": "restart_or_replace_instance",
    }}
    result = _score(records, truths, {"case-1": "not-json"})
    assert result["false_abstention_count"] == 1
    assert result["false_abstention_rate"] == 1.0
