#!/usr/bin/env python3
"""Run the frozen Experiment 05 one-shot blind evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from causetune.benchmark_e05 import SCORER_VERSION, scorer_fingerprint
from causetune.evidence import sha256_path
from causetune.incident_benchmark import packet_evidence_ids
from causetune.incident_evaluation import evaluate_incidents
from causetune.model import load_adapter, load_frozen_quantized_base, load_tokenizer_for_model
from causetune.verify import score_incident_predictions


SLICES = ("standard", "hard", "transfer")
EXPECTED_BENCHMARK = "b3daed4f49b123b0270baebf49e7609c06f26e5bb10fd125185d6e0864644eaf"
EXPECTED_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [_read_json_line(line, path, number) for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()]


def _read_json_line(line: str, path: Path, number: int) -> dict[str, Any]:
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError(f"{path}:{number} is not an object")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prompt_ids(tokenizer: Any, system_instruction: str, packet: str) -> list[int]:
    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": packet}]
    try:
        values = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        values = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _generate(model: Any, tokenizer: Any, records: list[dict[str, Any]], system_instruction: str, *, max_new_tokens: int, batch_size: int) -> dict[str, str]:
    model.eval()
    device = next(model.parameters()).device
    outputs: dict[str, str] = {}
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            prompts = [_prompt_ids(tokenizer, system_instruction, item["incident_packet"]) for item in batch]
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
                attention[row, width - len(prompt) :] = 1
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for item, sequence in zip(batch, generated):
                outputs[item["incident_id"]] = tokenizer.decode(sequence[width:], skip_special_tokens=True)
            print(f"generated={min(start + len(batch), len(records))}/{len(records)}", flush=True)
    if len(outputs) != len(records):
        raise ValueError("generation did not produce exactly one output per case")
    return outputs


def _without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def _enrich_predictions(records: list[dict[str, Any]], truth_by_id: Mapping[str, Mapping[str, Any]], raw: Mapping[str, str]) -> dict[str, Any]:
    evaluation = evaluate_incidents(records, truth_by_id, raw)
    for incident, row in zip(records, evaluation["predictions"]):
        row["input_metadata"] = {
            "present_components": incident["metadata"]["present_components"],
            "available_evidence_ids": sorted(packet_evidence_ids(incident["incident_packet"])),
        }
    return evaluation


def _transition_rows(source: Mapping[str, Any], target: Mapping[str, Any]) -> list[dict[str, Any]]:
    source_by_id = {row["incident_id"]: row for row in source["predictions"]}
    target_by_id = {row["incident_id"]: row for row in target["predictions"]}
    rows: list[dict[str, Any]] = []
    for incident_id in source_by_id:
        source_row = source_by_id[incident_id]
        target_row = target_by_id[incident_id]
        source_correct = bool(source_row["diagnosis_exact"])
        target_correct = bool(target_row["diagnosis_exact"])
        category = (
            "source_wrong_target_correct" if not source_correct and target_correct else
            "source_correct_target_wrong" if source_correct and not target_correct else
            "persistent_correct" if source_correct and target_correct else
            "persistent_wrong"
        )
        rows.append({
            "incident_id": incident_id,
            "slice": target_row["slice"],
            "category": category,
            "source_diagnosis_exact": source_correct,
            "target_diagnosis_exact": target_correct,
            "expected": target_row["expected"],
            "source_predicted": source_row["predicted"],
            "target_predicted": target_row["predicted"],
        })
    return rows


def _transition_summary(source_name: str, target_name: str, source: Mapping[str, Any], target: Mapping[str, Any]) -> dict[str, Any]:
    rows = _transition_rows(source, target)
    counts = {category: sum(row["category"] == category for row in rows) for category in (
        "source_wrong_target_correct", "source_correct_target_wrong", "persistent_correct", "persistent_wrong"
    )}
    return {"source": source_name, "target": target_name, "count": len(rows), "counts": counts, "rows": rows}


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()


def _check_frozen_contract(protocol: Mapping[str, Any], manifest: Mapping[str, Any], prompt_path: Path) -> None:
    if not protocol.get("prompt_frozen") or not protocol.get("one_shot_per_system") or not protocol.get("fresh_reload_per_system"):
        raise ValueError("E05 frozen one-shot contract is incomplete")
    if protocol.get("benchmark_fingerprint") != EXPECTED_BENCHMARK or manifest.get("fingerprint") != EXPECTED_BENCHMARK:
        raise ValueError("E05 benchmark fingerprint does not match the frozen contract")
    if protocol.get("scorer_version") != SCORER_VERSION or protocol.get("scorer_fingerprint") != scorer_fingerprint():
        raise ValueError("E05 scorer is not the frozen scorer")
    if hashlib.sha256(prompt_path.read_bytes()).hexdigest() != protocol.get("prompt_sha256"):
        raise ValueError("E05 prompt hash does not match the frozen protocol")
    systems = protocol.get("systems")
    if not isinstance(systems, list) or [system.get("name") for system in systems] != ["base", "e02", "e04"]:
        raise ValueError("E05 systems are not the frozen base/e02/e04 set")
    for system in systems:
        if system.get("revision") != EXPECTED_REVISION:
            raise ValueError(f"{system.get('name')} model revision is not pinned")


def _system_model(system: Mapping[str, Any], quantization: Mapping[str, Any]) -> Any:
    base = load_frozen_quantized_base(system["model_id"], revision=system["revision"], **quantization)
    if system["adapter"]:
        return load_adapter(base, system["adapter"])
    return base


def _persist_system_evaluation(
    root: Path,
    name: str,
    records: list[dict[str, Any]],
    truth_by_id: Mapping[str, Mapping[str, Any]],
    raw: Mapping[str, str],
) -> dict[str, Any]:
    destination = root / "evaluations" / name
    raw_rows = [{"incident_id": incident["incident_id"], "raw_output": raw[incident["incident_id"]]} for incident in records]
    _write_jsonl(destination / "raw_outputs.jsonl", raw_rows)
    evaluation = _enrich_predictions(records, truth_by_id, raw)
    prediction_rows = evaluation["predictions"]
    _write_jsonl(destination / "predictions.jsonl", prediction_rows)
    _write_json(destination / "evaluation.json", {"schema_version": 1, "system": name, "metrics": evaluation})
    by_slice = {}
    for split in SLICES:
        split_records = [record for record in records if record["slice"] == split]
        split_truth = {record["incident_id"]: truth_by_id[record["incident_id"]] for record in split_records}
        split_raw = {record["incident_id"]: raw[record["incident_id"]] for record in split_records}
        by_slice[split] = _without_predictions(_enrich_predictions(split_records, split_truth, split_raw))
    _write_json(destination / "slice_metrics.json", by_slice)
    reproduced = score_incident_predictions(destination / "predictions.jsonl")
    _write_json(destination / "offline_reproduction.json", {"status": "PASS", "metrics": reproduced})
    return {"system": name, "metrics": _without_predictions(evaluation), "slice_metrics": by_slice}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_e05")
    parser.add_argument("--protocol", default="results/experiment_05/protocol.json")
    parser.add_argument("--output-dir", default="results/experiment_05")
    parser.add_argument("--seed", type=int, default=20260913)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    root = Path(args.output_dir)
    protocol = _read_json(Path(args.protocol))
    manifest = _read_json(dataset_dir / "manifest.json")
    prompt_path = Path(protocol["prompt_path"])
    _check_frozen_contract(protocol, manifest, prompt_path)
    if not torch.cuda.is_available():
        raise RuntimeError("E05 requires CUDA; no scientific CPU fallback is allowed")
    evaluation_root = root / "evaluations"
    if evaluation_root.exists() and any(evaluation_root.rglob("*")):
        raise RuntimeError(f"E05 evaluation output already exists; one-shot run is immutable: {evaluation_root}")
    records = [record for split in SLICES for record in _read_jsonl(dataset_dir / f"{split}.jsonl")]
    truth = _read_jsonl(dataset_dir / "ground_truth.jsonl")
    truth_by_id = {row["incident_id"]: row for row in truth}
    config = _read_json(Path("configs/incident_diagnosis_eval.json"))
    system_instruction = config["evaluation_contract"]["system_instruction"]
    systems = {system["name"]: system for system in protocol["systems"]}
    summaries: dict[str, dict[str, Any]] = {}
    raw_hashes: dict[str, str] = {}
    for name in ("base", "e02", "e04"):
        destination = evaluation_root / name
        try:
            print(f"loading={name}", flush=True)
            tokenizer = load_tokenizer_for_model(systems[name]["model_id"], revision=systems[name]["revision"])
            model = _system_model(systems[name], config["quantization"])
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
            summaries[name] = _persist_system_evaluation(root, name, records, truth_by_id, raw)
            del model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            _write_json(destination / "technical_failure.json", {
                "status": "TECHNICAL_FAILURE",
                "valid_semantic_run": False,
                "stage": "model_load_or_generation_or_persistence",
                "exception": f"{type(exc).__name__}: {exc}",
                "git_sha": _git_sha(),
                "protocol_path": str(args.protocol),
                "benchmark_fingerprint": manifest["fingerprint"],
                "benchmark_hash": sha256_path(dataset_dir),
                "attempt": 0,
            })
            raise
    transitions = {
        "base_to_e04": _transition_summary("base", "e04", summaries["base"]["metrics"], summaries["e04"]["metrics"]),
        "e02_to_e04": _transition_summary("e02", "e04", summaries["e02"]["metrics"], summaries["e04"]["metrics"]),
    }
    _write_json(root / "transition_analysis.json", transitions)
    _write_json(root / "evaluation_comparison.json", {
        "schema_version": 1,
        "experiment": "E05",
        "benchmark_fingerprint": manifest["fingerprint"],
        "systems": summaries,
        "transitions": {key: {k: v for k, v in value.items() if k != "rows"} for key, value in transitions.items()},
        "raw_output_sha256": raw_hashes,
        "e03_used": False,
    })
    _write_json(root / "g05b_summary.json", {
        "schema_version": 1,
        "status": "PASS",
        "benchmark_fingerprint": manifest["fingerprint"],
        "systems": ["base", "e02", "e04"],
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "same_prompt": True,
        "same_scorer": True,
        "same_decoding": True,
        "raw_predictions": True,
        "offline_reproduction": True,
        "checkpoint_switching": False,
        "post_result_prompt_change": False,
        "e03_used": False,
    })
    artifact_paths = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "artifact_hashes.json":
            artifact_paths[path.relative_to(root).as_posix()] = sha256_path(path)
    _write_json(root / "artifact_hashes.json", {
        "schema_version": 1,
        "benchmark_fingerprint": manifest["fingerprint"],
        "artifacts": artifact_paths,
    })
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": manifest["fingerprint"], "systems": list(summaries), "raw_outputs": raw_hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
