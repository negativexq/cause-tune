from __future__ import annotations

import json
from pathlib import Path


def test_e06_native_compatibility_recovery_is_frozen() -> None:
    record = json.loads(Path("results/experiment_06/compatibility_recovery.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["attempt"] == 1
    assert record["attempt_0"]["status"] == "TECHNICAL_FAILURE"
    assert record["model_id"] == "microsoft/Phi-4-mini-instruct"
    assert record["model_revision"] == "cfbefacb99257ffa30c83adab238a50856ac3083"
    assert record["trust_remote_code"] is False
    assert record["benchmark_unchanged"] is True
    assert record["prompt_unchanged"] is True
    assert record["scorer_unchanged"] is True
    assert record["decoding_unchanged"] is True
    assert record["trivial_non_benchmark_generation"]["status"] == "PASS"
    assert record["trivial_non_benchmark_generation"]["scored"] is False
