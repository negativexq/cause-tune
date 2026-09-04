#!/usr/bin/env python3
"""Run exactly one M5 realistic-data QLoRA train and frozen-contract evaluation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import random
import subprocess
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from fineforge.config import LoraConfig, QuantizationConfig, SFTConfig
from fineforge.data.preprocess import preprocess_records
from fineforge.data.validation import read_realistic_splits
from fineforge.evaluation import aggregate_failures, evaluate_split, teacher_forced_metrics
from fineforge.evaluation_contract import contract_fingerprint
from fineforge.m5 import (
    EXPECTED_COUNTS,
    EXPECTED_EVALUATION_FINGERPRINT,
    TOP_HARD_PAIRS,
    adapter_artifact_metadata,
    confusion_delta,
    dataset_fingerprint,
    git_state,
    metric_delta,
    prediction_transitions,
    parameter_structure,
    sequence_summary,
    validate_m5_config,
)
from fineforge.model import (
    adapter_parameter_count,
    attach_lora,
    load_adapter,
    load_quantized_base,
    load_tokenizer,
)
from fineforge.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda
from fineforge.training import QLoRATrainingOOMError, train_qlora


EXPECTED_TRAINABLE_PARAMETERS = 33_030_144
EXPECTED_LOGICAL_PARAMETERS = 4_055_498_240


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _sft_config(config: dict[str, Any]) -> SFTConfig:
    training = config["training"]
    return SFTConfig(
        model_id=config["model_id"],
        seed=config["seed"],
        dataset_path=str(Path(config["dataset_dir"]) / "train.jsonl"),
        max_examples=EXPECTED_COUNTS["train"],
        train_ratio=1.0,
        validation_ratio=0.0,
        test_ratio=0.0,
        max_sequence_length=training["max_sequence_length"],
        micro_batch_size=training["micro_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        num_epochs=training["num_epochs"],
        quantization=QuantizationConfig(**config["quantization"]),
        lora=LoraConfig(
            rank=config["lora"]["rank"],
            alpha=config["lora"]["alpha"],
            dropout=config["lora"]["dropout"],
            target_modules=tuple(config["lora"]["target_modules"]),
        ),
        gradient_checkpointing=training["gradient_checkpointing"],
        gradient_checkpointing_use_reentrant=training["gradient_checkpointing_use_reentrant"],
        output_dir=config["output_dir"],
    )


def _cuda_only(model: Any) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for M5 training")
    devices = {str(parameter.device) for parameter in model.parameters()}
    if any(not device.startswith("cuda") for device in devices):
        raise RuntimeError(f"CPU/offloaded model parameters detected: {sorted(devices)}")
    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        bad = {name: value for name, value in device_map.items() if str(value).lower() in {"cpu", "disk"}}
        if bad:
            raise RuntimeError(f"CPU/disk model placement detected: {bad}")
    return {"parameter_devices": sorted(devices), "hf_device_map": device_map}


def _environment() -> dict[str, Any]:
    versions = {}
    for package in ("transformers", "bitsandbytes", "peft", "trl"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "torch": torch.__version__,
        "packages": versions,
        "cuda_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "gpu_capacity_gib": (
            round(torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3, 6)
            if torch.cuda.is_available() else None
        ),
    }


def _failure_file(path: Path, failures: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for failure in failures:
            handle.write(json.dumps(failure, ensure_ascii=False, sort_keys=True) + "\n")


def _preflight(config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(f"M5 output directory already exists; refusing a second run: {output_dir}")
    contract_config = json.loads(Path(config["evaluation_config"]).read_text(encoding="utf-8"))
    actual_contract = contract_fingerprint(contract_config)
    if actual_contract != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("evaluation contract fingerprint differs from the frozen M4 fingerprint")
    if config["evaluation_contract_fingerprint"] != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("M5 config evaluation contract fingerprint is incorrect")
    splits = read_realistic_splits(config["dataset_dir"])
    actual_counts = {split: len(records) for split, records in splits.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected M4 split counts: {actual_counts}")
    dataset_hashes = dataset_fingerprint(config["dataset_dir"])
    m4_commit = config.get("m4_baseline_commit", "f4ab955")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", m4_commit, "--", config["dataset_dir"]],
        check=False,
    ).returncode == 0
    if not unchanged:
        raise RuntimeError("M4 dataset files differ from the committed M4 baseline")
    preflight = {
        "status": "pass",
        "git_state_before_run": git_state(),
        "m4_baseline_commit": m4_commit,
        "dataset_counts": actual_counts,
        "dataset_fingerprint": dataset_hashes,
        "dataset_unchanged_since_m4": unchanged,
        "evaluation_contract_fingerprint": actual_contract,
        "evaluation_config": config["evaluation_config"],
        "model_weights_loaded": False,
        "training_started": False,
        "test_metrics_read": False,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "preflight.json", preflight)
    return preflight


def _comparison(
    splits: dict[str, list[dict[str, Any]]],
    base_metrics: dict[str, Any],
    tuned_metrics: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "base_artifacts": "outputs/baseline_v2",
        "tuned_artifacts": "outputs/realistic_qlora_v1",
        "splits": {},
    }
    transitions: dict[str, Any] = {}
    all_failures: list[dict[str, Any]] = []
    for split in ("validation", "id_test", "hard_test", "ood_test"):
        base = base_metrics["splits"][split]
        tuned = tuned_metrics[split]
        result["splits"][split] = {
            "BASE": {key: base[key] for key in ("intent_accuracy", "macro_f1", "valid_json_rate", "exact_schema_compliance_rate", "teacher_forced_loss")},
            "TUNED": {key: tuned[key] for key in ("intent_accuracy", "macro_f1", "valid_json_rate", "exact_schema_compliance_rate", "teacher_forced_loss")},
            "DELTA": metric_delta(base, tuned),
            "confusion_deltas": confusion_delta(base["confusion_matrix"], tuned["confusion_matrix"], TOP_HARD_PAIRS),
        }
        base_predictions = base["predictions"]
        tuned_predictions = tuned["predictions"]
        transitions[split] = prediction_transitions(splits[split], base_predictions, tuned_predictions)
        failures, _ = aggregate_failures(split, splits[split], tuned_predictions)
        all_failures.extend(failures)
    result["failure_analysis"] = {
        "by_split": transitions,
        "totals": {
            key: sum(item["counts"].get(key, 0) for item in transitions.values())
            for key in ("base_wrong_tuned_correct", "base_correct_tuned_wrong", "persistent_errors")
        },
    }
    result["new_regressions"] = result["failure_analysis"]["totals"]["base_correct_tuned_wrong"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realistic_qlora_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_m5_config(config)
    output_dir = Path(config["output_dir"])
    preflight = _preflight(config, output_dir)
    print(json.dumps({
        "m5_preflight": "PASS",
        "git_state_before_run": preflight["git_state_before_run"],
        "dataset_fingerprint": preflight["dataset_fingerprint"]["aggregate_sha256"],
        "split_counts": preflight["dataset_counts"],
        "evaluation_contract_fingerprint": preflight["evaluation_contract_fingerprint"],
        "model_weights_loaded": False,
        "training_started": False,
    }, indent=2, sort_keys=True), flush=True)
    eval_config = json.loads(Path(config["evaluation_config"]).read_text(encoding="utf-8"))
    system_instruction = eval_config["evaluation_contract"]["system_instruction"]
    sft_config = _sft_config(config)
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    splits = read_realistic_splits(config["dataset_dir"])
    tokenizer = load_tokenizer(sft_config)
    train_examples = preprocess_records(
        splits["train"], tokenizer, sft_config.max_sequence_length, system_message=system_instruction
    )
    validation_examples = preprocess_records(
        splits["validation"], tokenizer, sft_config.max_sequence_length, system_message=system_instruction
    )
    train_sequence_summary = sequence_summary(train_examples)
    validation_sequence_summary = sequence_summary(validation_examples)
    if train_sequence_summary["zero_supervised_examples"] or validation_sequence_summary["zero_supervised_examples"]:
        raise AssertionError("zero supervised-token example detected")
    model = None
    try:
        print("Loading quantized base for validation and QLoRA training...", flush=True)
        model = load_quantized_base(sft_config)
        base_placement = _cuda_only(model)
        validation_before = teacher_forced_metrics(model, tokenizer, validation_examples)
        model = attach_lora(model, sft_config)
        structure = parameter_structure(model, config["lora"]["target_modules"])
        trainable = structure["trainable_parameter_count"]
        logical = structure["logical_parameter_count"]
        missing_targets = [
            target for target, present in structure["lora_target_modules_present"].items()
            if not present
        ]
        if (
            structure["counting_method"] != "peft.get_nb_trainable_parameters"
            or trainable != EXPECTED_TRAINABLE_PARAMETERS
            or logical != EXPECTED_LOGICAL_PARAMETERS
            or missing_targets
            or not structure["base_parameters_frozen"]
            or not structure["lora_parameters_trainable"]
        ):
            raise RuntimeError(
                "unexpected parameter structure: "
                f"trainable={trainable}, logical={logical}, "
                f"packed_parameter_numel={structure['packed_parameter_numel']}, "
                f"trainable_percentage={structure['trainable_percentage']:.6f}, "
                f"missing_targets={missing_targets}, "
                f"base_parameters_frozen={structure['base_parameters_frozen']}, "
                f"lora_parameters_trainable={structure['lora_parameters_trainable']}; "
                f"expected trainable/logical={EXPECTED_TRAINABLE_PARAMETERS}/"
                f"{EXPECTED_LOGICAL_PARAMETERS}"
            )
        adapter_placement = _cuda_only(model)
        _write_json(output_dir / "resolved_training_config.json", {
            "experiment": config["experiment"],
            "model_id": config["model_id"],
            "seed": seed,
            "dataset_fingerprint": preflight["dataset_fingerprint"],
            "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
            "evaluation_config": eval_config,
            "quantization": config["quantization"],
            "lora": config["lora"],
            "training": config["training"],
            "train_split_only": True,
            "validation_used_for_pretraining_loss_only": True,
            "test_splits_used_during_training": False,
            "base_placement": base_placement,
            "adapter_placement": adapter_placement,
            "trainable_parameter_count": trainable,
            "logical_parameter_count": logical,
            "packed_parameter_numel": structure["packed_parameter_numel"],
            "parameter_counting_method": structure["counting_method"],
            "trainable_percentage": structure["trainable_percentage"],
            "lora_target_modules_present": structure["lora_target_modules_present"],
            "base_parameters_frozen": structure["base_parameters_frozen"],
            "lora_parameters_trainable": structure["lora_parameters_trainable"],
            "effective_batch_size": sft_config.micro_batch_size * sft_config.gradient_accumulation_steps,
        })
        synchronize_cuda()
        training_start = time.perf_counter()
        try:
            training_metrics, loss_history = train_qlora(model, tokenizer, train_examples, sft_config)
        except QLoRATrainingOOMError as exc:
            _write_json(output_dir / "oom_report.json", {
                "status": "cuda_oom",
                "stage": exc.stage,
                "error": str(exc.cause),
                "memory": exc.snapshot,
                "sequence_length": sft_config.max_sequence_length,
                "micro_batch_size": sft_config.micro_batch_size,
                "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
                "model_id": config["model_id"],
                "quantization": config["quantization"],
                "lora": config["lora"],
                "automatic_retry": False,
            })
            raise
        training_duration = time.perf_counter() - training_start
        training_metrics.update({
            "mean_loss": mean(float(item["loss"]) for item in loss_history),
            "validation_teacher_forced_loss_before": validation_before["teacher_forced_loss"],
            "train_sequence_summary": train_sequence_summary,
            "validation_sequence_summary": validation_sequence_summary,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capacity_gib": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3,
            "training_wall_clock_seconds_measured": training_duration,
            "trainable_percentage": trainable / logical * 100,
        })
        _write_json(output_dir / "training_metrics.json", training_metrics)
        _write_json(output_dir / "loss_history.json", loss_history)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            synchronize_cuda()

    print("Loading fresh quantized base and reloading adapter...", flush=True)
    reload_base = load_quantized_base(sft_config)
    reload_placement = _cuda_only(reload_base)
    reloaded_model = load_adapter(reload_base, str(output_dir))
    reloaded_model.config.use_cache = True
    reloaded_adapter_count = adapter_parameter_count(reloaded_model)
    if reloaded_adapter_count != EXPECTED_TRAINABLE_PARAMETERS:
        raise RuntimeError(f"reloaded adapter parameter count {reloaded_adapter_count} differs from expected")
    reload_verification = {
        "fresh_base_loaded": True,
        "adapter_loaded": True,
        "adapter_parameters_present": True,
        "adapter_parameter_count": reloaded_adapter_count,
        "adapter_target_modules": config["lora"]["target_modules"],
        "merged_into_base": False,
        "training_model_discarded_before_reload": True,
        "reload_base_placement": reload_placement,
    }
    _write_json(output_dir / "reload_verification.json", reload_verification)
    tokenizer = load_tokenizer(sft_config)
    post_metrics: dict[str, Any] = {}
    tuned_failures: list[dict[str, Any]] = []
    for split in ("validation", "id_test", "hard_test", "ood_test"):
        print(f"Evaluating reloaded adapter on {split}...", flush=True)
        examples = preprocess_records(
            splits[split], tokenizer, sft_config.max_sequence_length, system_message=system_instruction
        )
        metrics = evaluate_split(
            reloaded_model,
            tokenizer,
            splits[split],
            examples,
            system_message=system_instruction,
            max_new_tokens=eval_config["evaluation_contract"]["generation"]["max_new_tokens"],
            batch_size=eval_config["evaluation_contract"]["generation"]["batch_size"],
        )
        post_metrics[split] = metrics
        failures, _ = aggregate_failures(split, splits[split], metrics["predictions"])
        tuned_failures.extend(failures)
    del reloaded_model
    del reload_base
    gc.collect()
    torch.cuda.empty_cache()
    _write_json(output_dir / "post_training_metrics.json", {
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "metrics": post_metrics,
        "test_metrics_started_after_training": True,
    })
    _failure_file(output_dir / "failures.jsonl", tuned_failures)

    baseline_dir = Path("outputs/baseline_v2")
    baseline_config = json.loads((baseline_dir / "evaluation_config.json").read_text(encoding="utf-8"))
    if baseline_config.get("contract_fingerprint") != EXPECTED_EVALUATION_FINGERPRINT:
        raise RuntimeError("baseline_v2 contract fingerprint does not match frozen contract")
    base_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))
    comparison = _comparison(splits, base_metrics, post_metrics)
    _write_json(output_dir / "base_vs_tuned.json", comparison)
    _write_json(output_dir / "environment_metadata.json", {
        **_environment(),
        "git_state_before_run": preflight["git_state_before_run"],
        "dataset_fingerprint": preflight["dataset_fingerprint"],
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "no_test_metrics_during_training": True,
    })
    _write_json(output_dir / "artifact_metadata.json", adapter_artifact_metadata(output_dir))
    _write_json(output_dir / "m5_summary.json", {
        "status": "success",
        "training": json.loads((output_dir / "training_metrics.json").read_text(encoding="utf-8")),
        "post_training_metrics": {
            split: {key: post_metrics[split][key] for key in ("intent_accuracy", "macro_f1", "valid_json_rate", "exact_schema_compliance_rate", "teacher_forced_loss")}
            for split in post_metrics
        },
        "comparison": comparison,
        "reload_verification": reload_verification,
        "artifact_metadata": json.loads((output_dir / "artifact_metadata.json").read_text(encoding="utf-8")),
    })
    print(json.dumps({
        "status": "success",
        "output_dir": str(output_dir),
        "optimizer_steps": training_metrics["optimizer_steps"],
        "training_seconds": training_metrics["wall_clock_training_seconds"],
        "post_training": {
            split: {
                "accuracy": post_metrics[split]["intent_accuracy"],
                "macro_f1": post_metrics[split]["macro_f1"],
            }
            for split in post_metrics
        },
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
