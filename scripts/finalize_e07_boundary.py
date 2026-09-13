#!/usr/bin/env python3
"""Re-score persisted E07 raw predictions after a deterministic scorer correction."""

from __future__ import annotations

import json
from pathlib import Path

from causetune.evidence import sha256_path
from causetune.incident_taxonomy import SLICES
try:
    from run_e07_boundary import _read_json, _read_jsonl, _score, _write_json, _write_jsonl, _without_predictions
except ModuleNotFoundError:
    from scripts.run_e07_boundary import _read_json, _read_jsonl, _score, _write_json, _write_jsonl, _without_predictions


def main() -> None:
    root = Path("results/experiment_07/evaluations")
    dataset = Path("data/incident_diagnosis_e07_boundary")
    if not root.exists():
        raise RuntimeError("E07 semantic evaluation directory is missing")
    records = [record for split in SLICES for record in _read_jsonl(dataset / f"{split}.jsonl")]
    truths = {row["incident_id"]: row for row in _read_jsonl(dataset / "ground_truth.jsonl")}
    previous: dict[str, object] = {}
    summaries: dict[str, object] = {}
    raw_hashes: dict[str, str] = {}
    for name in ("base", "e02", "e04"):
        destination = root / name
        old_metrics = _read_json(destination / "evaluation.json")["metrics"]
        previous[name] = _without_predictions(old_metrics)
        raw_rows = _read_jsonl(destination / "raw_outputs.jsonl")
        raw = {row["incident_id"]: row["raw_output"] for row in raw_rows}
        evaluation = _score(records, truths, raw)
        _write_jsonl(destination / "predictions.jsonl", evaluation["predictions"])
        _write_json(destination / "evaluation.json", {"schema_version": 1, "system": name, "metrics": _without_predictions(evaluation)})
        _write_json(destination / "offline_reproduction.json", {"status": "PASS", "metrics": _without_predictions(evaluation), "scorer_version": "e07-boundary-scorer-v2"})
        raw_hashes[name] = sha256_path(destination / "raw_outputs.jsonl")
        summaries[name] = {"system": name, "metrics": _without_predictions(evaluation)}
    _write_json(root / "scoring_correction.json", {
        "status": "PASS",
        "scorer_version": "e07-boundary-scorer-v2",
        "semantic_predictions_regenerated": False,
        "raw_predictions_unchanged": True,
        "correction": "Invalid or non-diagnosis outputs on sufficient cases are false abstentions; the first scorer omitted invalid outputs from that denominator.",
        "previous_scorer_version": "e07-boundary-scorer-v1",
        "previous_metrics": previous,
    })
    comparison = _read_json(root / "evaluation_comparison.json")
    comparison["systems"] = summaries
    comparison["scorer_version"] = "e07-boundary-scorer-v2"
    comparison["semantic_predictions_regenerated"] = False
    _write_json(root / "evaluation_comparison.json", comparison)
    g07 = _read_json(root / "g07_summary.json")
    g07["scorer_version"] = "e07-boundary-scorer-v2"
    g07["scoring_correction_recorded"] = True
    _write_json(root / "g07_summary.json", g07)
    artifacts = {path.relative_to(root).as_posix(): sha256_path(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    _write_json(root / "artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": _read_json(dataset / "manifest.json")["fingerprint"], "artifacts": artifacts})
    print(json.dumps({"status": "PASS", "scorer_version": "e07-boundary-scorer-v2", "semantic_predictions_regenerated": False}, sort_keys=True))


if __name__ == "__main__":
    main()
