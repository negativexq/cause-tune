from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "incident_diagnosis_e07_boundary"
PROTOCOL = ROOT / "results" / "experiment_07" / "protocol.json"


def test_e07_boundary_freeze_is_complete() -> None:
    assert PROTOCOL.exists()
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["experiment"] == "E07"
    assert protocol["inputs_frozen"] is True
    assert protocol["truths_frozen"] is True
    assert protocol["case_count"] == 60
    assert protocol["systems"] == ["base", "e02", "e04"]
    assert protocol["contamination_audit"]["status"] == "PASS"
    assert set(protocol["abstention_semantics"]) == {"diagnose", "insufficient_evidence", "ambiguous_evidence", "out_of_taxonomy"}


def test_e07_boundary_has_expected_categories() -> None:
    manifest = json.loads((DATA / "manifest.json").read_text())
    assert manifest["case_count"] == 60
    assert manifest["case_counts"] == {
        "sufficient": 12,
        "insufficient_evidence": 12,
        "contradictory_evidence": 8,
        "multiple_plausible_culprits": 8,
        "missing_topology": 6,
        "missing_metrics": 6,
        "out_of_taxonomy": 8,
    }
