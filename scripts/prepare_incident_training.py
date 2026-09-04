"""Generate and validate the independent 02B.1 incident train/validation data."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

from causetune.incident_contract import contract_fingerprint
from causetune.incident_training import (
    TRAINING_GENERATOR_VERSION,
    TRAINING_VERSION,
    TRAIN_SEED,
    VALIDATION_SEED,
    build_diagnosis_chat_record,
    contamination_report,
    decoded_target_report,
    earliest_within_tolerance,
    generate_training_split,
    preprocess_incident_records,
    training_fingerprint,
    validate_training_manifest,
    validate_training_split,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n", encoding="utf-8")


def _percentile(values: list[int], percentile: float) -> int:
    values = sorted(values)
    if not values:
        return 0
    index = min(len(values) - 1, round((len(values) - 1) * percentile))
    return values[index]


def _token_stats(inputs, truths, tokenizer, system_instruction: str, max_sequence_length: int | None = None):
    # A deliberately generous temporary limit measures the actual distribution
    # before selecting the final safe power-of-two context.
    unbounded = preprocess_incident_records(inputs, truths, tokenizer, 8192, system_instruction)
    input_lengths = [len(item.input_ids) for item in unbounded]
    supervised_lengths = [item.trainable_assistant_token_count for item in unbounded]
    candidates = (512, 768, 1024, 1536, 2048, 4096, 8192)
    selected = max_sequence_length or next((candidate for candidate in candidates if max(input_lengths) <= candidate), None)
    if selected is None:
        raise ValueError(f"generated packet exceeds available safe context: {max(input_lengths)} tokens")
    processed = preprocess_incident_records(inputs, truths, tokenizer, selected, system_instruction)
    decoded = decoded_target_report(processed, truths, tokenizer)
    stats = {
        "examples": len(processed),
        "input_tokens_total": sum(input_lengths),
        "supervised_tokens_total": sum(supervised_lengths),
        "input_length": {"mean": statistics.mean(input_lengths), "p50": _percentile(input_lengths, .50), "p95": _percentile(input_lengths, .95), "max": max(input_lengths)},
        "supervised_length": {"mean": statistics.mean(supervised_lengths), "p50": _percentile(supervised_lengths, .50), "p95": _percentile(supervised_lengths, .95), "max": max(supervised_lengths)},
        "zero_supervised_examples": decoded["zero_supervised"],
        "truncated_examples": 0,
        "exceeding_max_sequence_length": sum(length > selected for length in input_lengths),
        "max_sequence_length": selected,
        "selection_rationale": "smallest configured power-of-two/common context covering every generated conversation without semantic truncation",
        "decoded_target_mismatches": decoded["decoded_target_mismatches"],
    }
    if stats["exceeding_max_sequence_length"] or stats["decoded_target_mismatches"] or stats["zero_supervised_examples"]:
        raise ValueError(f"unsafe preprocessing statistics: {stats}")
    return stats, processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/incident_diagnosis_training")
    parser.add_argument("--benchmark-dir", default="data/incident_diagnosis")
    parser.add_argument("--config", default="configs/incident_diagnosis_training.json")
    parser.add_argument("--tokenizer-model", default="Qwen/Qwen3-4B")
    args = parser.parse_args()
    out = Path(args.output_dir)
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    train_inputs, train_truth = generate_training_split("train", 200, TRAIN_SEED, "train")
    val_inputs, val_truth = generate_training_split("validation", 24, VALIDATION_SEED, "validation")
    train_report = validate_training_split(train_inputs, train_truth, expected_count=2400, expected_per_family=200)
    val_report = validate_training_split(val_inputs, val_truth, expected_count=288, expected_per_family=24)
    contamination = contamination_report(train_inputs, val_inputs, args.benchmark_dir)
    if contamination["status"] != "pass":
        raise ValueError(f"contamination checks failed: {contamination}")
    eval_config_path = Path(args.config).with_name("incident_diagnosis_eval.json")
    eval_config = json.loads(eval_config_path.read_text(encoding="utf-8"))
    system_instruction = eval_config["evaluation_contract"]["system_instruction"]
    evaluation_fingerprint = contract_fingerprint(eval_config)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model, use_fast=True)
    train_stats, train_processed = _token_stats(train_inputs, train_truth, tokenizer, system_instruction)
    val_stats, val_processed = _token_stats(val_inputs, val_truth, tokenizer, system_instruction, train_stats["max_sequence_length"])
    train_fp = training_fingerprint(train_inputs, train_truth)
    val_fp = training_fingerprint(val_inputs, val_truth)
    manifest = {
        "version": TRAINING_VERSION, "generator_version": TRAINING_GENERATOR_VERSION,
        "seeds": {"train": TRAIN_SEED, "validation": VALIDATION_SEED},
        "train_fingerprint": train_fp, "validation_fingerprint": val_fp,
        "counts": {"train": 2400, "validation": 288},
        "family_counts": {"train": train_report["family_counts"], "validation": val_report["family_counts"]},
        "difficulty_counts": {"train": train_report["difficulty_counts"], "validation": val_report["difficulty_counts"]},
        "token_stats": {"train": train_stats, "validation": val_stats},
        "contamination": contamination,
        "benchmark_fingerprint": contamination["benchmark_fingerprint"],
        "evaluation_fingerprint": evaluation_fingerprint,
    }
    resolved = {
        "experiment": config["experiment"], "model_id": config["model_id"], "quantization": config["quantization"], "lora": config["lora"],
        "training": config["training"] | {"max_sequence_length": train_stats["max_sequence_length"]},
        "train_fingerprint": train_fp, "validation_fingerprint": val_fp,
        "frozen_benchmark_fingerprint": contamination["benchmark_fingerprint"], "frozen_evaluation_fingerprint": evaluation_fingerprint,
        "seeds": config["seeds"], "checkpoint_policy": config["checkpoint_policy"], "early_stopping": config["early_stopping"], "max_epochs": config["training"]["max_epochs"],
        "training_sources": {"train": "data/incident_diagnosis_training/train.jsonl", "validation": "data/incident_diagnosis_training/validation.jsonl"},
    }
    validate_training_manifest(resolved, expected_benchmark_fingerprint=contamination["benchmark_fingerprint"], expected_evaluation_fingerprint=evaluation_fingerprint)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "train.jsonl", train_inputs)
    _write_jsonl(out / "validation.jsonl", val_inputs)
    _write_jsonl(out / "ground_truth_train.jsonl", train_truth)
    _write_jsonl(out / "ground_truth_validation.jsonl", val_truth)
    inspection = []
    for rows, truths in ((train_inputs, train_truth), (val_inputs, val_truth)):
        for family in ("db_connection_pool_exhaustion", "cache_stampede", "configuration_regression"):
            for difficulty in ("standard", "hard", "transfer"):
                match = next((truth for truth in truths if truth["failure_mode"] == family and truth["metadata"]["difficulty"] == difficulty), None)
                if match:
                    inspection.append({"split": rows[0]["split"], "incident_id": match["incident_id"], "packet": next(item["incident_packet"] for item in rows if item["incident_id"] == match["incident_id"]), "truth": match})
    _write_jsonl(out / "inspection_sample.jsonl", inspection)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "resolved_training_manifest.json").write_text(json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out / "validation_report.json").write_text(json.dumps({"status": "pass", "train": train_report, "validation": val_report, "contamination": contamination}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "pass", "manifest": manifest, "resolved_training_manifest": resolved}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
