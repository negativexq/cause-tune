from __future__ import annotations

import json
from pathlib import Path

from causetune.evidence import sha256_path
from causetune.verify import score_incident_predictions


ROOT = Path("results/experiment_06/capability_gap-retry-01")


def test_e06_screen_retry_is_complete_and_offline_reproducible() -> None:
    provenance = json.loads((ROOT / "retry_provenance.json").read_text(encoding="utf-8"))
    evaluation = json.loads((ROOT / "evaluation.json").read_text(encoding="utf-8"))
    assert provenance["status"] == "PASS"
    assert provenance["attempt"] == 1
    assert provenance["previous_attempt"].endswith("capability_gap/technical_failure.json")
    assert provenance["semantic_generation_count"] == 1
    assert score_incident_predictions(ROOT / "predictions.jsonl") == evaluation["metrics"]
    decision = json.loads((ROOT / "capability_gap_decision.json").read_text(encoding="utf-8"))
    assert decision["status"] == "CAPABILITY_GAP_PRESENT"
    assert decision["training_performed"] is False


def test_e06_screen_retry_hashes_match() -> None:
    index = json.loads((ROOT / "artifact_hashes.json").read_text(encoding="utf-8"))
    actual = {
        path.relative_to(ROOT).as_posix(): sha256_path(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file() and path.name != "artifact_hashes.json"
    }
    assert index["artifacts"] == actual
