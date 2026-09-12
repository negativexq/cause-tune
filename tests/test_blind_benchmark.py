from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.benchmark_blind_v2 import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    generate_blind_benchmark,
    scorer_fingerprint,
    write_blind_benchmark,
)


def test_blind_benchmark_is_deterministic_and_independent_namespace() -> None:
    first = generate_blind_benchmark()
    second = generate_blind_benchmark()
    assert first == second
    inputs, truth = first
    assert sum(len(rows) for rows in inputs.values()) == 60
    assert {split: len(rows) for split, rows in inputs.items()} == {"standard": 24, "hard": 24, "transfer": 12}
    assert all(row["metadata"]["generator_version"] == GENERATOR_VERSION for row in truth)


def test_blind_manifest_freezes_scorer_and_contamination_boundary(tmp_path: Path) -> None:
    source = tmp_path / "old.jsonl"
    source.write_text('{"incident_id":"old","incident_packet":"different"}\n', encoding="utf-8")
    manifest = write_blind_benchmark(tmp_path / "blind", contamination_sources=(source,))
    assert manifest["benchmark_version"] == BENCHMARK_VERSION
    assert manifest["scorer_fingerprint"] == scorer_fingerprint()
    assert manifest["frozen"] is True
    assert manifest["contamination_audit"]["status"] == "pass"
    assert json.loads((tmp_path / "blind" / "manifest.json").read_text())["fingerprint"] == manifest["fingerprint"]


def test_exact_overlap_blocks_freeze(tmp_path: Path) -> None:
    inputs, _truth = generate_blind_benchmark()
    packet = inputs["standard"][0]["incident_packet"]
    source = tmp_path / "old.jsonl"
    source.write_text(json.dumps({"incident_id": "old", "incident_packet": packet}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="contamination"):
        write_blind_benchmark(tmp_path / "blind", contamination_sources=(source,))
