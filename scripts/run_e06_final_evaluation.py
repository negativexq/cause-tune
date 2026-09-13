#!/usr/bin/env python3
"""Run the frozen one-shot E06 base/tuned final evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

import torch

from causetune.benchmark_e05 import SCORER_VERSION, scorer_fingerprint
from causetune.evidence import sha256_path
from causetune.incident_benchmark import packet_evidence_ids
from causetune.incident_evaluation import evaluate_incidents
from causetune.incident_taxonomy import SLICES
from causetune.model import load_adapter, load_frozen_quantized_base, load_tokenizer_for_model
from causetune.verify import score_incident_predictions
from run_experiment_05 import _generate, _write_json, _write_jsonl


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _check_protocol(protocol: Mapping[str, Any], benchmark: Mapping[str, Any]) -> None:
    if not protocol.get("inputs_frozen") or not protocol.get("truths_frozen"):
        raise ValueError("E06 final benchmark is not frozen")
    if protocol.get("benchmark_fingerprint") != benchmark.get("fingerprint"):
        raise ValueError("E06 final benchmark fingerprint mismatch")
    if protocol.get("scorer_version") != SCORER_VERSION or protocol.get("scorer_fingerprint") != scorer_fingerprint():
        raise ValueError("E06 final scorer is not canonical")
    prompt = Path(protocol["prompt_path"])
    if hashlib.sha256(prompt.read_bytes()).hexdigest() != protocol["prompt_sha256"]:
        raise ValueError("E06 final prompt hash mismatch")
    if protocol.get("decoding") != {"max_new_tokens": 96, "batch_size": 4, "do_sample": False}:
        raise ValueError("E06 final decoding is not frozen")
    systems = protocol.get("systems")
    if [system.get("name") for system in systems] != ["base", "tuned"]:
        raise ValueError("E06 final systems are not base/tuned")
    for system in systems:
        if system.get("model_id") != "microsoft/Phi-4-mini-instruct" or system.get("revision") != "cfbefacb99257ffa30c83adab238a50856ac3083":
            raise ValueError("E06 final model identity changed")
        if system.get("trust_remote_code") is not False:
            raise ValueError("E06 final native-loading recovery is not frozen")


def _enrich(records: list[dict[str, Any]], evaluation: dict[str, Any]) -> dict[str, Any]:
    by_id = {record["incident_id"]: record for record in records}
    for row in evaluation["predictions"]:
        source = by_id[row["incident_id"]]
        row["input_metadata"] = {
            "present_components": source["metadata"]["present_components"],
            "available_evidence_ids": sorted(packet_evidence_ids(source["incident_packet"])),
        }
    return evaluation


def _transition(source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    source_rows = {row["incident_id"]: row for row in source["predictions"]}
    target_rows = {row["incident_id"]: row for row in target["predictions"]}
    categories = ("source_wrong_target_correct", "source_correct_target_wrong", "persistent_correct", "persistent_wrong")
    rows = []
    for incident_id in source_rows:
        source_correct = bool(source_rows[incident_id]["diagnosis_exact"])
        target_correct = bool(target_rows[incident_id]["diagnosis_exact"])
        category = (
            categories[0] if not source_correct and target_correct else
            categories[1] if source_correct and not target_correct else
            categories[2] if source_correct and target_correct else categories[3]
        )
        rows.append({
            "incident_id": incident_id,
            "category": category,
            "source_diagnosis_exact": source_correct,
            "target_diagnosis_exact": target_correct,
            "source_predicted": source_rows[incident_id]["predicted"],
            "target_predicted": target_rows[incident_id]["predicted"],
            "expected": target_rows[incident_id]["expected"],
        })
    return {"source": "base", "target": "tuned", "count": len(rows), "counts": {category: sum(row["category"] == category for row in rows) for category in categories}, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e06_final")
    parser.add_argument("--protocol", default="results/experiment_06/final_protocol.json")
    parser.add_argument("--output-dir", default="results/experiment_06/final_evaluation")
    args = parser.parse_args()
    root = Path(args.output_dir)
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing to reuse immutable E06 final evaluation directory: {root}")
    dataset = Path(args.dataset_dir)
    protocol = _read_json(Path(args.protocol))
    benchmark = _read_json(dataset / "manifest.json")
    _check_protocol(protocol, benchmark)
    records = [record for split in SLICES for record in _read_jsonl(dataset / f"{split}.jsonl")]
    truth = _read_jsonl(dataset / "ground_truth.jsonl")
    if len(records) != 60 or len(truth) != 60:
        raise ValueError("E06 final benchmark requires exactly 60 cases")
    truth_by_id = {row["incident_id"]: row for row in truth}
    eval_config = _read_json(Path("configs/incident_diagnosis_eval.json"))
    system_instruction = eval_config["evaluation_contract"]["system_instruction"]
    root.mkdir(parents=True, exist_ok=False)
    summaries: dict[str, dict[str, Any]] = {}
    evaluations: dict[str, dict[str, Any]] = {}
    raw_hashes: dict[str, str] = {}
    try:
        for system in protocol["systems"]:
            name = system["name"]
            destination = root / name
            print(f"loading={name}", flush=True)
            tokenizer = load_tokenizer_for_model(system["model_id"], revision=system["revision"], trust_remote_code=False)
            model = load_frozen_quantized_base(system["model_id"], revision=system["revision"], trust_remote_code=False, **eval_config["quantization"])
            if system.get("adapter"):
                model = load_adapter(model, system["adapter"])
            raw = _generate(
                model,
                tokenizer,
                records,
                system_instruction,
                max_new_tokens=protocol["decoding"]["max_new_tokens"],
                batch_size=protocol["decoding"]["batch_size"],
            )
            _write_jsonl(destination / "raw_outputs.jsonl", [{"incident_id": record["incident_id"], "raw_output": raw[record["incident_id"]]} for record in records])
            raw_hashes[name] = sha256_path(destination / "raw_outputs.jsonl")
            evaluation = _enrich(records, evaluate_incidents(records, truth_by_id, raw))
            _write_jsonl(destination / "predictions.jsonl", evaluation["predictions"])
            _write_json(destination / "evaluation.json", {"schema_version": 1, "system": name, "metrics": evaluation})
            by_slice = {}
            for split in SLICES:
                split_records = [record for record in records if record["slice"] == split]
                split_truth = {record["incident_id"]: truth_by_id[record["incident_id"]] for record in split_records}
                split_raw = {record["incident_id"]: raw[record["incident_id"]] for record in split_records}
                by_slice[split] = _without_predictions(_enrich(split_records, evaluate_incidents(split_records, split_truth, split_raw)))
            _write_json(destination / "slice_metrics.json", by_slice)
            reproduced = score_incident_predictions(destination / "predictions.jsonl")
            _write_json(destination / "offline_reproduction.json", {"status": "PASS", "metrics": reproduced})
            if reproduced != evaluation:
                raise ValueError(f"offline reproduction mismatch for {name}")
            evaluations[name] = evaluation
            summaries[name] = {"system": name, "metrics": _without_predictions(evaluation), "slice_metrics": by_slice}
            del model, tokenizer
            gc.collect(); torch.cuda.empty_cache()
    except Exception as exc:
        _write_json(root / "technical_failure.json", {
            "status": "TECHNICAL_FAILURE",
            "valid_semantic_run": False,
            "stage": "model_load_or_generation_or_persistence",
            "exception": f"{type(exc).__name__}: {exc}",
            "git_sha": _git_sha(),
            "benchmark_fingerprint": benchmark["fingerprint"],
            "benchmark_hash": sha256_path(dataset),
            "protocol_path": args.protocol,
        })
        raise
    transition = _transition(evaluations["base"], evaluations["tuned"])
    _write_json(root / "transition_analysis.json", transition)
    _write_json(root / "evaluation_comparison.json", {
        "schema_version": 1,
        "experiment": "E06",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "systems": summaries,
        "transitions": {key: value for key, value in transition.items() if key != "rows"},
        "raw_output_sha256": raw_hashes,
        "capability_screen_used_as_final_evidence": False,
    })
    _write_json(root / "g06_summary.json", {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_fingerprint": benchmark["fingerprint"],
        "systems": ["base", "tuned"],
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "raw_predictions": True,
        "offline_reproduction": True,
        "architecture_differences_documented": True,
        "capability_screen_used_as_final_evidence": False,
    })
    artifacts = {path.relative_to(root).as_posix(): sha256_path(path) for path in sorted(root.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"}
    _write_json(root / "artifact_hashes.json", {"schema_version": 1, "benchmark_fingerprint": benchmark["fingerprint"], "artifacts": artifacts})
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": benchmark["fingerprint"], "systems": list(summaries), "raw_output_sha256": raw_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
