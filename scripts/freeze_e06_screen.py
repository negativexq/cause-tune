#!/usr/bin/env python3
"""Freeze the 48-case E06 cross-model capability-gap screen."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from causetune.benchmark_e05 import benchmark_fingerprint, contamination_audit, generate_e05_benchmark


MODEL_ID = "microsoft/Phi-4-mini-instruct"
MODEL_REVISION = "cfbefacb99257ffa30c83adab238a50856ac3083"
SCREEN_SEED = 20260915


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/incident_diagnosis_e06_screen")
    args = parser.parse_args()
    output = Path(args.output_dir)
    inputs, truth, provenance = generate_e05_benchmark(SCREEN_SEED)
    selected = inputs["standard"][:48]
    selected_ids = {row["incident_id"] for row in selected}
    selected_truth = [row for row in truth if row["incident_id"] in selected_ids]
    selected_provenance = [row for row in provenance if row["incident_id"] in selected_ids]
    screen_inputs = {"standard": selected, "hard": [], "transfer": []}
    audit = contamination_audit(
        screen_inputs,
        "data/incident_diagnosis_training",
        "data/incident_diagnosis",
        "data/incident_diagnosis_blind_v2",
    )
    if audit["status"] != "pass":
        raise ValueError(f"E06 capability screen contamination audit failed: {audit}")
    fingerprint = benchmark_fingerprint(screen_inputs, selected_truth)
    _write_jsonl(output / "standard.jsonl", selected)
    _write_jsonl(output / "ground_truth.jsonl", selected_truth)
    _write_json(output / "case_provenance.json", selected_provenance)
    _write_json(output / "manifest.json", {
        "schema_version": 1,
        "experiment": "E06",
        "benchmark_version": "e06-capability-screen-v1",
        "source_generator": "e05-independent-generator-namespace-v1",
        "generation_seed": SCREEN_SEED,
        "fingerprint": fingerprint,
        "counts": {"standard": 48, "hard": 0, "transfer": 0},
        "total_cases": 48,
        "origin_counts": dict(Counter(row["origin"] for row in selected_provenance)),
        "taxonomy_reused": True,
        "screen_is_not_final_evidence": True,
        "contamination_audit": audit,
        "frozen": True,
    })
    _write_json(Path("results/experiment_06/capability_gap_protocol.json"), {
        "schema_version": 1,
        "experiment": "E06",
        "protocol_version": "e06-capability-gap-screen-v1",
        "status": "FROZEN_BEFORE_MODEL_EVALUATION",
        "purpose": "pre-training capability-gap screen for the selected non-Qwen model family",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "license": "MIT",
        "benchmark_dir": str(output),
        "benchmark_fingerprint": fingerprint,
        "case_count": 48,
        "screen_source": "new deterministic seed using the independent E05 generator namespace; not final E06 evidence",
        "prompt_path": "configs/incident_diagnosis_eval.json",
        "prompt_sha256": hashlib.sha256(Path("configs/incident_diagnosis_eval.json").read_bytes()).hexdigest(),
        "prompt_frozen": True,
        "decoding": {"max_new_tokens": 96, "batch_size": 4, "do_sample": False},
        "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
        "trust_remote_code": True,
        "rejection_threshold": {
            "diagnosis_exact_min": 0.90,
            "resolution_exact_min": 0.90,
            "failure_mode_macro_f1_min": 0.90,
            "strict_json_min": 0.90,
            "rule": "reject only when all four primary screen metrics meet or exceed thresholds; otherwise a meaningful capability gap exists",
        },
        "training_before_screen": False,
        "final_evidence_is_separate": True,
    })
    print(json.dumps({"status": "PASS", "fingerprint": fingerprint, "case_count": 48, "model_id": MODEL_ID, "model_revision": MODEL_REVISION}, sort_keys=True))


if __name__ == "__main__":
    main()
