import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_v1_readiness_audit_is_pass_and_complete():
    audit = json.loads((ROOT / "results/release/v1.0-readiness-audit.json").read_text())
    assert audit["status"] == "PASS"
    assert audit["mandatory_gates"] == {
        "E04": "PASS",
        "E05": "PASS",
        "E06": "PASS",
        "E07": "PASS",
        "E08": "PASS",
    }
    assert [row["experiment"] for row in audit["rows"]] == [
        "E04-A",
        "E04-B",
        "E04-C",
        "E05",
        "E06",
        "E07",
        "E08",
    ]
    assert audit["rows"][3]["headline_metrics"]["e04_vs_e02_diagnosis_delta_pp"] < -5
    assert audit["rows"][-1]["headline_metrics"]["e04_quality_preserving"] is False
