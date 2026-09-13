from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_e06_final_benchmark_is_frozen_separate_and_clean() -> None:
    manifest = json.loads((ROOT / "data/incident_diagnosis_e06_final/manifest.json").read_text(encoding="utf-8"))
    protocol = json.loads((ROOT / "results/experiment_06/final_protocol.json").read_text(encoding="utf-8"))
    audit = manifest["contamination_audit"]
    assert manifest["frozen"] is True
    assert manifest["total_cases"] == 60
    assert manifest["fingerprint"] == protocol["benchmark_fingerprint"]
    assert manifest["final_evidence_separate_from_capability_screen"] is True
    assert audit["status"] == "pass"
    assert audit["exact_packet_overlap_count"] == 0
    assert audit["normalized_packet_overlap_count"] == 0
    assert audit["incident_id_overlap_count"] == 0
    assert audit["canonical_structural_overlap_count"] == 60
    assert protocol["one_shot_per_system"] is True
    assert protocol["capability_screen_used_as_final_evidence"] is False
