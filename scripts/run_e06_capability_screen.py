#!/usr/bin/env python3
"""Run the frozen E06 capability-gap screen without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch

from causetune.evidence import sha256_path
from causetune.incident_benchmark import packet_evidence_ids
from causetune.incident_evaluation import evaluate_incidents
from causetune.model import load_frozen_quantized_base, load_tokenizer_for_model
from causetune.verify import score_incident_predictions
from run_experiment_05 import _generate, _write_json, _write_jsonl


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _without_predictions(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e06_screen")
    parser.add_argument("--protocol", default="results/experiment_06/capability_gap_protocol.json")
    parser.add_argument("--compatibility-record", default="results/experiment_06/compatibility_recovery.json")
    parser.add_argument("--output-dir", default="results/experiment_06/capability_gap")
    args = parser.parse_args()
    dataset = Path(args.dataset_dir)
    protocol_path = Path(args.protocol)
    compatibility_path = Path(args.compatibility_record)
    output = Path(args.output_dir)
    protocol = _read_json(protocol_path)
    compatibility = _read_json(compatibility_path)
    manifest = _read_json(dataset / "manifest.json")
    prompt_path = Path(protocol["prompt_path"])
    if protocol.get("status") != "FROZEN_BEFORE_MODEL_EVALUATION" or not protocol.get("prompt_frozen"):
        raise ValueError("E06 capability screen protocol is not frozen")
    if protocol.get("benchmark_fingerprint") != manifest.get("fingerprint"):
        raise ValueError("E06 capability screen fingerprint mismatch")
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != protocol.get("prompt_sha256"):
        raise ValueError("E06 capability screen protocol is missing its prompt hash")
    if compatibility.get("status") != "PASS" or compatibility.get("attempt_0", {}).get("status") != "TECHNICAL_FAILURE":
        raise ValueError("E06 compatibility recovery record is invalid")
    for key in ("model_id", "model_revision", "benchmark_fingerprint", "prompt_sha256", "scorer_version", "scorer_fingerprint"):
        if compatibility.get(key) != (protocol.get("model_id") if key == "model_id" else protocol.get(key)):
            raise ValueError(f"E06 compatibility recovery changed frozen field: {key}")
    if compatibility.get("decoding") != protocol.get("decoding") or not compatibility.get("decoding_unchanged"):
        raise ValueError("E06 compatibility recovery changed decoding")
    records = _read_jsonl(dataset / "standard.jsonl")
    truth = _read_jsonl(dataset / "ground_truth.jsonl")
    if len(records) != 48 or len(truth) != 48:
        raise ValueError("E06 capability screen requires exactly 48 cases")
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"E06 capability screen output already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)
    truth_by_id = {row["incident_id"]: row for row in truth}
    config = _read_json(Path("configs/incident_diagnosis_eval.json"))
    try:
        tokenizer = load_tokenizer_for_model(protocol["model_id"], revision=protocol["model_revision"], trust_remote_code=compatibility["trust_remote_code"])
        model = load_frozen_quantized_base(protocol["model_id"], revision=protocol["model_revision"], trust_remote_code=compatibility["trust_remote_code"], **protocol["quantization"])
        raw = _generate(model, tokenizer, records, config["evaluation_contract"]["system_instruction"], max_new_tokens=protocol["decoding"]["max_new_tokens"], batch_size=protocol["decoding"]["batch_size"])
        _write_jsonl(output / "raw_outputs.jsonl", [{"incident_id": row["incident_id"], "raw_output": raw[row["incident_id"]]} for row in records])
        evaluation = evaluate_incidents(records, truth_by_id, raw)
        for incident, row in zip(records, evaluation["predictions"]):
            row["input_metadata"] = {
                "present_components": incident["metadata"]["present_components"],
                "available_evidence_ids": sorted(packet_evidence_ids(incident["incident_packet"])),
            }
        _write_jsonl(output / "predictions.jsonl", evaluation["predictions"])
        _write_json(output / "evaluation.json", {"schema_version": 1, "metrics": evaluation, "compatibility_record": str(compatibility_path)})
        reproduced = score_incident_predictions(output / "predictions.jsonl")
        if reproduced != evaluation:
            raise ValueError("E06 capability screen offline reproduction mismatch")
        _write_json(output / "offline_reproduction.json", {"status": "PASS", "metrics": reproduced})
        metrics = _without_predictions(evaluation)
        threshold = protocol["rejection_threshold"]
        observed = {
            "diagnosis_exact": metrics["diagnosis_exact_match"]["rate"],
            "resolution_exact": metrics["resolution_exact_match"]["rate"],
            "failure_mode_macro_f1": metrics["failure_mode_macro_f1"],
            "strict_json": metrics["json_compliance"]["rate"],
        }
        passed = {
            "diagnosis_exact": observed["diagnosis_exact"] >= threshold["diagnosis_exact_min"],
            "resolution_exact": observed["resolution_exact"] >= threshold["resolution_exact_min"],
            "failure_mode_macro_f1": observed["failure_mode_macro_f1"] >= threshold["failure_mode_macro_f1_min"],
            "strict_json": observed["strict_json"] >= threshold["strict_json_min"],
        }
        status = "REJECTED_NO_CAPABILITY_GAP" if all(passed.values()) else "CAPABILITY_GAP_PRESENT"
        _write_json(output / "capability_gap_decision.json", {
            "schema_version": 1,
            "status": status,
            "training_performed": False,
            "metrics": observed,
            "thresholds": threshold,
            "threshold_pass": passed,
            "final_evidence_not_used": True,
            "benchmark_fingerprint": manifest["fingerprint"],
        })
        artifact_paths = {path.relative_to(output).as_posix(): sha256_path(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
        _write_json(output / "artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": manifest["fingerprint"], "artifacts": artifact_paths})
        print(json.dumps({"status": status, "metrics": observed, "threshold_pass": passed}, sort_keys=True))
    except Exception as exc:
        _write_json(output / "technical_failure.json", {
            "status": "TECHNICAL_FAILURE",
            "valid_semantic_run": False,
            "stage": "model_load_or_generation_or_screen_scoring",
            "exception": f"{type(exc).__name__}: {exc}",
            "git_sha": _git_sha(),
            "benchmark_fingerprint": manifest["fingerprint"],
            "benchmark_hash": sha256_path(dataset),
            "attempt": 0,
        })
        raise
    finally:
        if "model" in locals():
            del model
        if "tokenizer" in locals():
            del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
