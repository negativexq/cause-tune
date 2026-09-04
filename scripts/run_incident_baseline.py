#!/usr/bin/env python3
"""Run the untouched Qwen3-4B baseline for Experiment 02A exactly once."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any, Mapping

import torch

from causetune.incident_benchmark import SLICES, benchmark_fingerprint, read_benchmark
from causetune.incident_evaluation import evaluate_incidents, failure_patterns
from causetune.incident_contract import contract_fingerprint
from causetune.model import load_frozen_quantized_base, load_tokenizer_for_model
from causetune.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda


def _prompt_ids(tokenizer: Any, system_instruction: str, packet: str) -> list[int]:
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": packet},
    ]
    try:
        tokenized = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokenized = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(value) for value in tokenized]


def generate_raw_outputs(model: Any, tokenizer: Any, inputs: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, str]:
    if not inputs:
        raise ValueError("cannot generate an empty incident slice")
    device = next(model.parameters()).device
    system_instruction = config["evaluation_contract"]["system_instruction"]
    generation = config["evaluation_contract"]["generation"]
    outputs: dict[str, str] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(inputs), generation["batch_size"]):
            batch = inputs[start : start + generation["batch_size"]]
            prompts = [_prompt_ids(tokenizer, system_instruction, item["incident_packet"]) for item in batch]
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.full(
                (len(prompts), width), tokenizer.pad_token_id, dtype=torch.long, device=device
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
                attention_mask[row, width - len(prompt) :] = 1
            synchronize_cuda()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=generation["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            synchronize_cuda()
            for item, sequence in zip(batch, generated):
                outputs[item["incident_id"]] = tokenizer.decode(sequence[width:], skip_special_tokens=True)
            print(f"generated {min(start + len(batch), len(inputs))}/{len(inputs)}", flush=True)
    return outputs


def _without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def _top_confusions(matrix: Mapping[str, Mapping[str, int]], limit: int = 10) -> list[dict[str, Any]]:
    pairs = [
        {"expected": actual, "predicted": predicted, "count": count}
        for actual, row in matrix.items()
        for predicted, count in row.items()
        if actual != predicted and count
    ]
    return sorted(pairs, key=lambda item: (-item["count"], item["expected"], item["predicted"]))[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/incident_diagnosis_eval.json")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    actual_contract = contract_fingerprint(config)
    if config.get("contract_fingerprint") != actual_contract:
        raise ValueError("incident evaluation contract fingerprint is missing or incorrect")
    dataset_dir = args.dataset_dir or config["benchmark_dir"]
    output_dir = Path(args.output_dir or config["output_dir"])
    inputs_by_slice, ground_truth_rows, manifest = read_benchmark(dataset_dir)
    if manifest["benchmark_fingerprint"] != benchmark_fingerprint(inputs_by_slice, ground_truth_rows):
        raise ValueError("benchmark fingerprint changed after freeze")
    truth_by_id = {row["incident_id"]: row for row in ground_truth_rows}
    all_inputs = [item for split in SLICES for item in inputs_by_slice[split]]
    if len(all_inputs) != 144:
        raise ValueError("Experiment 02A requires exactly 144 incidents")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the untouched Qwen3-4B baseline; no CPU fallback is used")
    random.seed(20260904)
    torch.manual_seed(20260904)
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    tokenizer = load_tokenizer_for_model(config["model_id"])
    model = load_frozen_quantized_base(config["model_id"], **config["quantization"])
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("baseline model unexpectedly has trainable parameters")
    if hasattr(model, "peft_config"):
        raise AssertionError("baseline must not load a PEFT adapter")
    model_load_seconds = time.perf_counter() - start_time
    raw_outputs = generate_raw_outputs(model, tokenizer, all_inputs, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_config.json").write_text(json.dumps({**config, "contract_fingerprint": actual_contract}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    split_metrics: dict[str, Any] = {}
    analyses: dict[str, Any] = {}
    all_predictions: list[dict[str, Any]] = []
    for split in SLICES:
        split_outputs = {item["incident_id"]: raw_outputs[item["incident_id"]] for item in inputs_by_slice[split]}
        metrics = evaluate_incidents(inputs_by_slice[split], truth_by_id, split_outputs)
        split_metrics[split] = _without_predictions(metrics)
        analyses[split] = {
            "failure_patterns": failure_patterns(inputs_by_slice[split], metrics),
            "top_failure_mode_confusions": _top_confusions(metrics["failure_mode_confusion"]),
            "per_failure_family": metrics["per_failure_family"],
        }
        all_predictions.extend(metrics["predictions"])
    all_metrics = evaluate_incidents(all_inputs, truth_by_id, raw_outputs)
    split_metrics["all"] = _without_predictions(all_metrics)
    analyses["all"] = {
        "failure_patterns": failure_patterns(all_inputs, all_metrics),
        "top_failure_mode_confusions": _top_confusions(all_metrics["failure_mode_confusion"]),
        "per_failure_family": all_metrics["per_failure_family"],
    }
    synchronize_cuda()
    memory = json_safe_memory(cuda_memory_snapshot())
    runtime_seconds = time.perf_counter() - start_time
    result = {
        "status": "success",
        "experiment": config["experiment"],
        "topic": config["topic"],
        "model": {
            "model_id": config["model_id"],
            "frozen": True,
            "adapter": None,
            "quantization": config["quantization"],
            "device": torch.cuda.get_device_name(),
        },
        "benchmark": {
            "directory": dataset_dir,
            "version": manifest["benchmark_version"],
            "fingerprint": manifest["benchmark_fingerprint"],
            "incident_count": 144,
            "slice_counts": manifest["slice_counts"],
            "failure_family_counts": manifest["failure_family_counts"],
        },
        "evaluation_contract_fingerprint": actual_contract,
        "generation": config["evaluation_contract"]["generation"],
        "runtime_seconds": runtime_seconds,
        "model_load_seconds": model_load_seconds,
        "peak_cuda_memory": memory,
        "training_performed": False,
        "optimizer_steps": 0,
        "splits": split_metrics,
        "analysis": analyses,
        "prediction_count": len(all_predictions),
    }
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in all_predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "failure_analysis.json").write_text(json.dumps(analyses, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    public = {
        "experiment": config["experiment"],
        "topic": config["topic"],
        "model": config["model_id"],
        "primary_metric": "diagnosis_exact_match",
        "benchmark_fingerprint": manifest["benchmark_fingerprint"],
        "evaluation_contract_fingerprint": actual_contract,
        "runtime_seconds": runtime_seconds,
        "peak_vram_gib": memory["peak_allocated_gib"],
        "splits": split_metrics,
        "analysis": analyses,
        "prediction_count": len(all_predictions),
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/incident_diagnosis_base.json").write_text(json.dumps(public, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(json.dumps({"status": "success", "output_dir": str(output_dir), "runtime_seconds": runtime_seconds, "splits": {split: split_metrics[split]["diagnosis_exact_match"] for split in SLICES}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
