#!/usr/bin/env python3
"""Evaluate frozen Qwen3-4B on the M4 dataset without adapters or optimizer steps."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable

import torch

from causetune.data.preprocess import preprocess_records
from causetune.data.validation import read_realistic_splits
from causetune.evaluation_contract import contract_fingerprint
from causetune.evaluation import aggregate_failures, classification_metrics, evaluate_split
from causetune.model import load_frozen_quantized_base, load_tokenizer_for_model
from causetune.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda


def _subset_metrics(
    records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    predicate: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    selected = [(record, prediction) for record, prediction in zip(records, predictions) if predicate(record)]
    if not selected:
        return {"count": 0}
    return classification_metrics(
        [record["expected_response"]["intent"] for record, _ in selected],
        [prediction["parsed_intent"] for _, prediction in selected],
    )


def _slice_metrics(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "difficulty": {
            difficulty: _subset_metrics(records, predictions, lambda row, value=difficulty: row["difficulty"] == value)
            for difficulty in ("easy", "medium", "hard")
        },
        "phenomena": {},
        "confusable_pairs": {},
    }
    phenomena = sorted({value for record in records for value in record["phenomena"]})
    for phenomenon in phenomena:
        result["phenomena"][phenomenon] = _subset_metrics(
            records, predictions, lambda row, value=phenomenon: value in row["phenomena"]
        )
    pairs = sorted({
        f"{record['expected_response']['intent']} vs {other}"
        for record in records for other in record["confusable_with"]
    })
    for pair in pairs:
        target, other = pair.split(" vs ", 1)
        result["confusable_pairs"][pair] = _subset_metrics(
            records,
            predictions,
            lambda row, target=target, other=other: (
                row["expected_response"]["intent"] == target and other in row["confusable_with"]
            ),
        )
    return result


def _write_failures(path: Path, failures: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/realistic")
    parser.add_argument("--config", default="configs/baseline_eval.json")
    parser.add_argument("--output-dir", default="outputs/baseline_v2")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    actual_fingerprint = contract_fingerprint(config)
    if config.get("contract_fingerprint") != actual_fingerprint:
        raise ValueError("baseline config contract_fingerprint does not match its contract")
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = read_realistic_splits(dataset_dir)
    config["dataset_dir"] = str(dataset_dir)
    config["dataset_version"] = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))["dataset_version"]
    config["evaluation_scope"] = ["validation", "id_test", "hard_test", "ood_test"]
    config["training_performed"] = False
    config["optimizer_steps"] = 0
    config["contract_fingerprint"] = actual_fingerprint
    config["supersedes"] = "outputs/baseline_original"
    config["supersession_reason"] = "Original M4 evaluation omitted an explicit task/output protocol; v2 freezes the corrected contract once."
    (output_dir / "evaluation_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen Qwen3-4B baseline; no CPU fallback is used")
    random.seed(20260904)
    torch.manual_seed(20260904)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tokenizer = load_tokenizer_for_model(config["model_id"])
    preprocessed = {
        split: preprocess_records(
            records,
            tokenizer,
            max_sequence_length=config["max_sequence_length"],
            system_message=config["evaluation_contract"]["system_instruction"],
        )
        for split, records in splits.items()
        if split in config["evaluation_scope"]
    }
    model = load_frozen_quantized_base(config["model_id"], **config["quantization"])
    synchronize_cuda()
    model_load_seconds = time.perf_counter() - start
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("baseline model contains trainable parameters")

    split_metrics: dict[str, Any] = {}
    all_failures: list[dict[str, Any]] = []
    category_counts: defaultdict[str, int] = defaultdict(int)
    diagnostic_counts: defaultdict[str, int] = defaultdict(int)
    for split in config["evaluation_scope"]:
        print(f"Evaluating frozen base on {split} ({len(splits[split])} examples)...", flush=True)
        split_start = time.perf_counter()
        metrics = evaluate_split(
            model,
            tokenizer,
            splits[split],
            preprocessed[split],
            system_message=config["evaluation_contract"]["system_instruction"],
            max_new_tokens=config["evaluation_contract"]["generation"]["max_new_tokens"],
            batch_size=config["evaluation_contract"]["generation"]["batch_size"],
        )
        failures, categories = aggregate_failures(split, splits[split], metrics["predictions"])
        all_failures.extend(failures)
        for category, count in categories.items():
            category_counts[category] += count
        for category, count in metrics["generation_diagnostic_counts"].items():
            diagnostic_counts[category] += count
        metrics["runtime_seconds"] = time.perf_counter() - split_start
        metrics["slices"] = _slice_metrics(splits[split], metrics["predictions"])
        split_metrics[split] = metrics
        print(json.dumps({key: metrics[key] for key in ("intent_accuracy", "macro_f1", "valid_json_rate", "runtime_seconds")}, sort_keys=True), flush=True)

    synchronize_cuda()
    snapshot = json_safe_memory(cuda_memory_snapshot())
    metrics = {
        "status": "success",
        "model": {
            "model_id": config["model_id"],
            "frozen": True,
            "adapter": None,
            "quantization": config["quantization"],
            "device": torch.cuda.get_device_name(),
        },
        "runtime_seconds": time.perf_counter() - start,
        "model_load_seconds": model_load_seconds,
        "peak_cuda_memory": snapshot,
        "peak_vram_gib": snapshot["peak_allocated_gib"],
        "training_performed": False,
        "optimizer_steps": 0,
        "splits": split_metrics,
        "failure_category_counts": dict(category_counts),
        "generation_diagnostic_counts": dict(diagnostic_counts),
        "failure_count": len(all_failures),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "confusion_matrix.json").write_text(json.dumps({split: value["confusion_matrix"] for split, value in split_metrics.items()}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_failures(output_dir / "failures.jsonl", all_failures)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({
        "status": "success",
        "output_dir": str(output_dir),
        "runtime_seconds": metrics["runtime_seconds"],
        "peak_vram_gib": metrics["peak_vram_gib"],
        "splits": {
            split: {key: value[key] for key in ("intent_accuracy", "macro_f1", "valid_json_rate")}
            for split, value in split_metrics.items()
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
