from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e08_frontier_retains_negative_quality_tradeoff() -> None:
    result = json.loads((ROOT / "results/experiment_08/frontier.json").read_text())
    assert result["status"] == "PASS"
    assert result["semantic_evaluations_launched"] is False
    candidate = result["candidates"]["e04_selected"]
    assert candidate["comparison_to_reference"]["quality_preserving"] is False
    assert candidate["comparison_to_reference"]["frontier_status"] == "negative_quality_tradeoff"
    assert candidate["comparison_to_reference"]["strict_pareto_dominated"] is False
