from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e08_protocol_freezes_only_measured_candidates() -> None:
    protocol = json.loads((ROOT / "results/experiment_08/protocol.json").read_text())
    assert protocol["status"] == "FROZEN"
    assert [candidate["name"] for candidate in protocol["candidates"]] == ["e02_original", "e04_selected"]
    assert protocol["reference_candidate"] == "e02_original"
    assert protocol["automatic_search"] is False
    assert protocol["new_semantic_evaluations"] is False
    assert protocol["quality_tolerance"]["diagnosis_exact_max_loss_pp"] == 1.0
