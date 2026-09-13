from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_e07_evaluation_contract_uses_three_frozen_systems() -> None:
    protocol = json.loads((ROOT / "results/experiment_07/protocol.json").read_text())
    assert protocol["systems"] == ["base", "e02", "e04"]
    assert protocol["selection_exclusion"].startswith("evaluation-only")
