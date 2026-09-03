#!/usr/bin/env python3
"""Run the single configured 200-example QLoRA SFT smoke milestone."""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from pathlib import Path
from typing import Any

import torch

from fineforge.config import load_config
from fineforge.data.preprocess import preprocess_records
from fineforge.data.schema import read_jsonl
from fineforge.data.split import load_split_manifest
from fineforge.evaluation import evaluate_split
from fineforge.model import (
    adapter_parameter_count,
    attach_lora,
    load_adapter,
    load_quantized_base,
    load_tokenizer,
)
from fineforge.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda
from fineforge.training import QLoRATrainingOOMError, train_qlora


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def comparison_view(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "teacher_forced_loss",
            "intent_accuracy",
            "valid_json_rate",
        )
    }


def metric_delta(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, float]:
    keys = ("teacher_forced_loss", "intent_accuracy", "valid_json_rate")
    return {key: float(tuned[key]) - float(base[key]) for key in keys}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_smoke.json")
    parser.add_argument(
        "--split-manifest",
        help="override the manifest next to config.dataset_path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "resolved_training_config.json", config.to_dict())

    records = read_jsonl(config.dataset_path)
    if len(records) != config.max_examples:
        raise ValueError(
            f"dataset contains {len(records)} records, expected {config.max_examples}"
        )
    manifest_path = Path(args.split_manifest or Path(config.dataset_path).with_suffix(".splits.json"))
    splits = load_split_manifest(records, manifest_path)
    expected_counts = {"train": 160, "validation": 20, "test": 20}
    actual_counts = {name: len(items) for name, items in splits.items()}
    if actual_counts != expected_counts:
        raise ValueError(f"expected split counts {expected_counts}, got {actual_counts}")

    print("Resolved training configuration:")
    print(config.human_readable())
    print("Using existing split manifest:", manifest_path)
    print("Split counts:", actual_counts)
    print("Loading tokenizer:", config.model_id)
    tokenizer = load_tokenizer(config)

    # Pre-tokenization happens before any model or optimizer is created. The
    # labels are produced by the foundation's assistant-only masking rule and
    # are passed unchanged to training and teacher-forced evaluation.
    preprocessed = {
        name: preprocess_records(items, tokenizer, config.max_sequence_length)
        for name, items in splits.items()
    }
    if any(
        example.trainable_assistant_token_count < 1
        for example in preprocessed["train"]
    ):
        raise AssertionError("a training example has no non-ignored assistant labels")

    print("Loading fresh quantized base model for baseline evaluation...")
    base_model = load_quantized_base(config)
    print(
        "Base model parameters:",
        sum(parameter.numel() for parameter in base_model.parameters()),
    )

    # This is the untouched base model baseline. The test split is not used
    # for optimization and is evaluated here only because the milestone
    # explicitly requires a pre-training held-out baseline.
    base_metrics: dict[str, Any] = {}
    for name in ("validation", "test"):
        print(f"Evaluating untouched base model on {name}...")
        base_metrics[name] = evaluate_split(
            base_model,
            tokenizer,
            splits[name],
            preprocessed[name],
        )

    print("Attaching configured LoRA adapter...")
    model = attach_lora(base_model, config)
    model.print_trainable_parameters()
    trainable_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    logical_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Trainable parameters: {trainable_count:,}")
    print(f"Logical parameters: {logical_count:,}")
    print("Starting the single configured training run...")
    training_metrics, loss_history = train_qlora(
        model,
        tokenizer,
        preprocessed["train"],
        config,
    )
    training_metrics["effective_batch_size"] = (
        config.micro_batch_size * config.gradient_accumulation_steps
    )
    training_metrics["model_id"] = config.model_id
    training_metrics["seed"] = config.seed
    write_json(output_dir / "training_metrics.json", training_metrics)
    write_json(output_dir / "loss_history.json", loss_history)

    print("Saving adapter and tokenizer:", output_dir)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # The original trained model is intentionally discarded before reload.
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        synchronize_cuda()

    print("Loading fresh quantized base model for adapter reload...")
    reload_base = load_quantized_base(config)
    reloaded_model = load_adapter(reload_base, str(output_dir))
    reloaded_adapter_parameters = adapter_parameter_count(reloaded_model)
    if reloaded_adapter_parameters <= 0:
        raise AssertionError("reloaded model has no materialized LoRA parameters")
    reload_verification = {
        "adapter_parameters_present": True,
        "adapter_parameter_count": reloaded_adapter_parameters,
        "adapter_dir": str(output_dir),
    }
    write_json(output_dir / "reload_verification.json", reload_verification)

    tuned_metrics: dict[str, Any] = {}
    for name in ("validation", "test"):
        print(f"Evaluating reloaded tuned model on {name}...")
        tuned_metrics[name] = evaluate_split(
            reloaded_model,
            tokenizer,
            splits[name],
            preprocessed[name],
        )

    comparison = {
        "base": {
            name: comparison_view(metrics)
            for name, metrics in base_metrics.items()
        },
        "tuned": {
            name: comparison_view(metrics)
            for name, metrics in tuned_metrics.items()
        },
        "delta": {
            name: metric_delta(base_metrics[name], tuned_metrics[name])
            for name in ("validation", "test")
        },
    }
    write_json(output_dir / "base_vs_tuned.json", comparison)
    write_json(
        output_dir / "metrics.json",
        {
            "status": "success",
            "model": {
                "model_id": config.model_id,
                "quantization": config.quantization.__dict__,
                "lora": config.lora.to_dict()
                if hasattr(config.lora, "to_dict")
                else {
                    "rank": config.lora.rank,
                    "alpha": config.lora.alpha,
                    "dropout": config.lora.dropout,
                    "target_modules": list(config.lora.target_modules),
                },
            },
            "split_counts": actual_counts,
            "training": training_metrics,
            "reload": reload_verification,
            "base": {
                name: comparison_view(metrics)
                for name, metrics in base_metrics.items()
            },
            "tuned": {
                name: comparison_view(metrics)
                for name, metrics in tuned_metrics.items()
            },
            "delta": comparison["delta"],
        },
    )

    print("\n--- TRAINING METRICS ---")
    for key, value in training_metrics.items():
        print(f"{key}: {value}")
    print("\n--- BASE VS TUNED METRICS ---")
    print(json.dumps(comparison, indent=2, sort_keys=True))
    print("\nSMOKE RUN: PASS")


if __name__ == "__main__":
    try:
        main()
    except QLoRATrainingOOMError as exc:
        print("\nQLORA SMOKE: CUDA OOM")
        print("Stage:", exc.stage)
        print("Peak memory:", json.dumps(exc.snapshot, sort_keys=True))
        raise

