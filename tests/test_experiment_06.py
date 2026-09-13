from __future__ import annotations

import json
from pathlib import Path


def test_e06_capability_screen_failure_is_preserved() -> None:
    record = json.loads(Path("results/experiment_06/capability_gap/technical_failure.json").read_text(encoding="utf-8"))
    assert record["status"] == "TECHNICAL_FAILURE"
    assert record["valid_semantic_run"] is False
    assert record["stage"] == "model_load_or_generation_or_screen_scoring"
    assert "LossKwargs" in record["exception"]
    assert not list(Path("results/experiment_06/capability_gap").glob("*predictions*"))
