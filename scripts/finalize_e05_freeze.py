#!/usr/bin/env python3
"""Persist the frozen E05 protocol, contamination report, and benchmark hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from causetune.benchmark_e05 import scorer_fingerprint


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="data/incident_diagnosis_e05")
    parser.add_argument("--output", default="results/experiment_05")
    args = parser.parse_args()
    benchmark = Path(args.benchmark)
    manifest = json.loads((benchmark / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest["scorer_fingerprint"] != scorer_fingerprint():
        raise ValueError("E05 benchmark manifest is not frozen with the expected scorer")
    files = sorted(path for path in benchmark.iterdir() if path.is_file())
    hashes = {path.name: sha256(path) for path in files}
    output = Path(args.output)
    write_json(output / "benchmark_artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": manifest["fingerprint"], "artifacts": hashes})
    write_json(output / "contamination_report.json", {"schema_version": 1, "benchmark_fingerprint": manifest["fingerprint"], "status": manifest["contamination_audit"]["status"], "sources": manifest["contamination_sources"], "audit": manifest["contamination_audit"]})
    write_json(output / "protocol.json", {
        "schema_version": 1,
        "experiment": "E05",
        "protocol_version": "e05-one-shot-blind-v1",
        "benchmark_version": manifest["benchmark_version"],
        "benchmark_fingerprint": manifest["fingerprint"],
        "benchmark_manifest_sha256": sha256(benchmark / "manifest.json"),
        "scorer_version": manifest["scorer_version"],
        "scorer_fingerprint": manifest["scorer_fingerprint"],
        "prompt_path": "configs/incident_diagnosis_eval.json",
        "prompt_frozen": True,
        "decoding": {"max_new_tokens": 96, "batch_size": 4, "do_sample": False},
        "systems": [
            {"name": "base", "model_id": "Qwen/Qwen3-4B", "adapter": None},
            {"name": "e02", "model_id": "Qwen/Qwen3-4B", "adapter": "outputs/incident_diagnosis_02b2/checkpoint-step-000100"},
            {"name": "e04", "model_id": "Qwen/Qwen3-4B", "adapter": "runs/experiment_04c/lr1e-4/runner/checkpoint-step-000100"},
        ],
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "checkpoint_switching": False,
        "post_result_prompt_change": False,
        "raw_predictions_required": True,
        "offline_reproduction_required": True,
        "e04_selection_after_freeze": True,
    })
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": manifest["fingerprint"], "artifacts_hashed": len(hashes)}, sort_keys=True))


if __name__ == "__main__":
    main()
