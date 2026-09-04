#!/usr/bin/env python3
"""Run exactly one M6 deterministic-shuffle QLoRA experiment."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from causetune.config import LoraConfig, QuantizationConfig, SFTConfig
from causetune.data.preprocess import preprocess_records
from causetune.data.validation import read_realistic_splits
from causetune.evaluation import aggregate_failures, evaluate_split
from causetune.evaluation_contract import contract_fingerprint
from causetune.m5 import (
    EXPECTED_COUNTS,
    EXPECTED_EVALUATION_FINGERPRINT,
    dataset_fingerprint,
    git_state,
    metric_delta,
    parameter_structure,
    prediction_transitions,
    sequence_summary,
    validate_m5_config,
)
from causetune.m5_audit import audit_training_order, deterministic_shuffled_indices
from causetune.m6 import EXPECTED_M4_DATASET_FINGERPRINT, m6_artifact_metadata, validate_m6_config
from causetune.model import adapter_parameter_count, attach_lora, load_adapter, load_quantized_base, load_tokenizer
from causetune.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda
from causetune.training import QLoRATrainingOOMError, train_qlora


EXPECTED_TRAINABLE_PARAMETERS = 33_030_144
EXPECTED_LOGICAL_PARAMETERS = 4_055_498_240
VALIDATION_CHECKPOINTS = (0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250)
WRONG_ITEM_PAIRS = (
    ("duplicate_charge", "wrong_item"),
    ("fraud_suspected", "wrong_item"),
    ("order_missing", "wrong_item"),
    ("refund", "wrong_item"),
    ("cancel_order", "wrong_item"),
)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
    handle.flush()


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
        raise RuntimeError("CUDA is required for M6 training")
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


def _selected_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metrics[key]
        for key in (
            "intent_accuracy",
            "macro_f1",
            "valid_json_rate",
            "exact_schema_compliance_rate",
            "teacher_forced_loss",
        )
    }


def _confusion_counts(matrix: dict[str, dict[str, int]], pairs: tuple[tuple[str, str], ...]) -> dict[str, int]:
    return {
        f"{actual} -> {predicted}": int(matrix.get(actual, {}).get(predicted, 0))
        for actual, predicted in pairs
    }


def _preflight(
    config: dict[str, Any],
    m5_config: dict[str, Any],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], list[int], list[dict[str, Any]]]:
    if output_dir.exists():
        raise FileExistsError(f"M6 output directory already exists; refusing a second run: {output_dir}")
    validate_m5_config(m5_config)
    config_diff = validate_m6_config(m5_config, config)
    eval_config = json.loads(Path(config["evaluation_config"]).read_text(encoding="utf-8"))
    actual_contract = contract_fingerprint(eval_config)
    if actual_contract != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("M6 evaluation contract fingerprint differs from frozen M4 fingerprint")
    if config["evaluation_contract_fingerprint"] != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("M6 evaluation contract fingerprint is incorrect")
    splits = read_realistic_splits(config["dataset_dir"])
    actual_counts = {split: len(records) for split, records in splits.items()}
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"unexpected M4 split counts: {actual_counts}")
    dataset_hashes = dataset_fingerprint(config["dataset_dir"])
    if dataset_hashes["aggregate_sha256"] != EXPECTED_M4_DATASET_FINGERPRINT:
        raise ValueError("M4 dataset fingerprint differs from the frozen dataset")
    m4_commit = m5_config.get("m4_baseline_commit", "f4ab955")
    unchanged = subprocess.run(
        ["git", "diff", "--quiet", m4_commit, "--", config["dataset_dir"]],
        check=False,
    ).returncode == 0
    if not unchanged:
        raise RuntimeError("M4 dataset files differ from the committed M4 baseline")

    original_train = splits["train"]
    seed = int(config["training_order"]["seed"])
    order = deterministic_shuffled_indices(len(original_train), seed)
    if order != deterministic_shuffled_indices(len(original_train), seed):
        raise AssertionError("same seed did not reproduce the same M6 order")
    if order == deterministic_shuffled_indices(len(original_train), seed + 1):
        raise AssertionError("different seed did not change the M6 order")
    ordered_train = [original_train[index] for index in order]
    order_audit = audit_training_order(
        ordered_train,
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        block_size=100,
    )
    if order_audit["single_class_window_count"] != 0:
        raise RuntimeError("deterministic shuffle produced a single-class accumulation window")
    if len(set(order_audit["last_50_labels"])) < 2 or len({
        record["expected_response"]["intent"] for record in ordered_train[-100:]
    }) < 2:
        raise RuntimeError("deterministic shuffle did not produce a mixed terminal order")

    preflight = {
        "status": "pass",
        "git_state_before_run": git_state(),
        "dataset_fingerprint": dataset_hashes,
        "dataset_counts": actual_counts,
        "evaluation_contract_fingerprint": actual_contract,
        "m5_config": "configs/realistic_qlora_v1.json",
        "m6_config": "configs/realistic_qlora_m6.json",
        "model_weights_loaded": False,
        "training_started": False,
        "test_metrics_read": False,
        "fixed_m5_fields_unchanged": config_diff["fixed_fields_unchanged"],
        "training_order": config["training_order"],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(output_dir / "experiment_diff.json", config_diff)
    _write_json(output_dir / "shuffled_order.json", {
        "algorithm": config["training_order"]["sampler"],
        "seed": seed,
        "indices": order,
        "example_ids": [record["example_id"] for record in ordered_train],
        "label_counts": dict(Counter(record["expected_response"]["intent"] for record in ordered_train)),
    })
    _write_json(output_dir / "order_audit.json", order_audit)
    return preflight, eval_config, order, ordered_train


def _validation_snapshot(
    step: int,
    model: Any,
    tokenizer: Any,
    records: list[dict[str, Any]],
    examples: list[Any],
    system_instruction: str,
    eval_config: dict[str, Any],
) -> dict[str, Any]:
    metrics = evaluate_split(
        model,
        tokenizer,
        records,
        examples,
        system_message=system_instruction,
        max_new_tokens=eval_config["evaluation_contract"]["generation"]["max_new_tokens"],
        batch_size=eval_config["evaluation_contract"]["generation"]["batch_size"],
    )
    return {
        "step": step,
        "train_loss": None if step == 0 else None,
        "validation_teacher_forced_loss": metrics["teacher_forced_loss"],
        "validation_intent_accuracy": metrics["intent_accuracy"],
        "validation_macro_f1": metrics["macro_f1"],
        "validation_valid_json_rate": metrics["valid_json_rate"],
        "validation_exact_schema_compliance_rate": metrics["exact_schema_compliance_rate"],
    }


def _build_comparison(
    base: dict[str, Any],
    m5: dict[str, Any],
    m6: dict[str, Any],
    records_by_split: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "splits": {},
        "confusion_analysis": {},
        "failure_transitions": {},
    }
    for split in ("validation", "id_test", "hard_test", "ood_test"):
        base_selected = _selected_metrics(base[split])
        m5_selected = _selected_metrics(m5[split])
        m6_selected = _selected_metrics(m6[split])
        result["splits"][split] = {
            "BASE": base_selected,
            "M5_UNSHUFFLED": m5_selected,
            "M6_SHUFFLED": m6_selected,
            "BASE_TO_M6_DELTA": metric_delta(base_selected, m6_selected),
            "M5_TO_M6_DELTA": metric_delta(m5_selected, m6_selected),
        }
        result["confusion_analysis"][split] = {
            "m5_vs_m6_wrong_item_pairs": {
                pair: {
                    "m5": m5[split]["confusion_matrix"].get(pair.split(" -> ")[0], {}).get(pair.split(" -> ")[1], 0),
                    "m6": m6[split]["confusion_matrix"].get(pair.split(" -> ")[0], {}).get(pair.split(" -> ")[1], 0),
                }
                for pair in [f"{actual} -> {predicted}" for actual, predicted in WRONG_ITEM_PAIRS]
            },
            "m5_predicted_label_distribution": dict(Counter(
                prediction.get("parsed_intent") or "<parse_failure>"
                for prediction in m5[split]["predictions"]
            )),
            "m6_predicted_label_distribution": dict(Counter(
                prediction.get("parsed_intent") or "<parse_failure>"
                for prediction in m6[split]["predictions"]
            )),
            "m6_class_recall": {
                label: values["recall"] for label, values in m6[split]["per_class"].items()
            },
        }
        result["failure_transitions"][split] = {
            "BASE_vs_M6": prediction_transitions(
                records_by_split[split], base[split]["predictions"], m6[split]["predictions"]
            ),
            "M5_vs_M6": prediction_transitions(
                records_by_split[split], m5[split]["predictions"], m6[split]["predictions"]
            ),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/realistic_qlora_m6.json")
    parser.add_argument("--m5-config", default="configs/realistic_qlora_v1.json")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    m5_config = json.loads(Path(args.m5_config).read_text(encoding="utf-8"))
    output_dir = Path(config["output_dir"])
    preflight, eval_config, order, ordered_train_records = _preflight(config, m5_config, output_dir)
    print(json.dumps({
        "m6_preflight": "PASS",
        "dataset_fingerprint": preflight["dataset_fingerprint"]["aggregate_sha256"],
        "evaluation_contract_fingerprint": preflight["evaluation_contract_fingerprint"],
        "split_counts": preflight["dataset_counts"],
        "order_length": len(order),
        "training_started": False,
    }, indent=2, sort_keys=True), flush=True)

    splits = read_realistic_splits(config["dataset_dir"])
    system_instruction = eval_config["evaluation_contract"]["system_instruction"]
    sft_config = _sft_config(config)
    seed = config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    tokenizer = load_tokenizer(sft_config)
    train_examples = preprocess_records(
        ordered_train_records, tokenizer, sft_config.max_sequence_length, system_message=system_instruction
    )
    validation_examples = preprocess_records(
        splits["validation"], tokenizer, sft_config.max_sequence_length, system_message=system_instruction
    )
    train_summary = sequence_summary(train_examples)
    validation_summary = sequence_summary(validation_examples)
    if train_summary["zero_supervised_examples"] or validation_summary["zero_supervised_examples"]:
        raise AssertionError("zero supervised-token example detected")

    validation_progression: list[dict[str, Any]] = []
    model = None
    optimizer_steps_path = output_dir / "optimizer_steps.jsonl"
    progression_path = output_dir / "validation_progression.jsonl"
    try:
        print("Loading quantized base for M6 validation and training...", flush=True)
        model = load_quantized_base(sft_config)
        base_placement = _cuda_only(model)
        model = attach_lora(model, sft_config)
        structure = parameter_structure(model, config["lora"]["target_modules"])
        if (
            structure["counting_method"] != "peft.get_nb_trainable_parameters"
            or structure["trainable_parameter_count"] != EXPECTED_TRAINABLE_PARAMETERS
            or structure["logical_parameter_count"] != EXPECTED_LOGICAL_PARAMETERS
            or not all(structure["lora_target_modules_present"].values())
            or not structure["base_parameters_frozen"]
            or not structure["lora_parameters_trainable"]
        ):
            raise RuntimeError(f"unexpected M6 parameter structure: {structure}")
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
            "training_order": config["training_order"],
            "train_split_only": True,
            "validation_used_for_progression_only": True,
            "test_splits_used_during_training": False,
            "base_placement": base_placement,
            "adapter_placement": adapter_placement,
            **structure,
            "effective_batch_size": sft_config.micro_batch_size * sft_config.gradient_accumulation_steps,
        })

        with optimizer_steps_path.open("w", encoding="utf-8", newline="\n") as step_handle, progression_path.open(
            "w", encoding="utf-8", newline="\n"
        ) as progression_handle:
            step_zero = _validation_snapshot(
                0, model, tokenizer, splits["validation"], validation_examples, system_instruction, eval_config
            )
            validation_progression.append(step_zero)
            _write_jsonl(progression_handle, step_zero)
            print(json.dumps(step_zero, sort_keys=True), flush=True)

            def on_optimizer_step(step_record: dict[str, Any], callback_model: Any) -> None:
                _write_jsonl(step_handle, step_record)
                step = int(step_record["optimizer_step"])
                if step in VALIDATION_CHECKPOINTS:
                    snapshot = _validation_snapshot(
                        step,
                        callback_model,
                        tokenizer,
                        splits["validation"],
                        validation_examples,
                        system_instruction,
                        eval_config,
                    )
                    snapshot["train_loss"] = step_record["mean_accumulated_loss"]
                    validation_progression.append(snapshot)
                    _write_jsonl(progression_handle, snapshot)
                if step % 10 == 0:
                    print(json.dumps({
                        "step": f"{step}/250",
                        "loss": step_record["mean_accumulated_loss"],
                        "labels": step_record["label_counts"],
                        "lr": step_record["learning_rate"],
                        "grad_norm": step_record["gradient_norm"],
                        "vram": step_record["vram"],
                    }, sort_keys=True), flush=True)

            synchronize_cuda()
            training_start = time.perf_counter()
            try:
                training_metrics, loss_history = train_qlora(
                    model,
                    tokenizer,
                    train_examples,
                    sft_config,
                    example_order=None,
                    optimizer_step_callback=on_optimizer_step,
                )
            except QLoRATrainingOOMError as exc:
                _write_json(output_dir / "oom_report.json", {
                    "status": "cuda_oom",
                    "stage": exc.stage,
                    "error": str(exc.cause),
                    "memory": exc.snapshot,
                    "sequence_length": sft_config.max_sequence_length,
                    "micro_batch_size": sft_config.micro_batch_size,
                    "gradient_accumulation_steps": sft_config.gradient_accumulation_steps,
                    "automatic_retry": False,
                })
                raise
            training_duration = time.perf_counter() - training_start

        training_metrics.update({
            "training_order": config["training_order"],
            "train_sequence_summary": train_summary,
            "validation_sequence_summary": validation_summary,
            "validation_teacher_forced_loss_before": validation_progression[0]["validation_teacher_forced_loss"],
            "validation_teacher_forced_loss_after": validation_progression[-1]["validation_teacher_forced_loss"],
            "validation_progression_checkpoints": VALIDATION_CHECKPOINTS,
            "gpu_name": torch.cuda.get_device_name(),
            "gpu_capacity_gib": torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1024**3,
            "training_wall_clock_seconds_measured": training_duration,
            "mean_loss": sum(float(item["loss"]) for item in loss_history) / len(loss_history),
            "trainable_percentage": (
                training_metrics["trainable_parameter_count"]
                / training_metrics["logical_parameter_count"]
                * 100
            ),
        })
        _write_json(output_dir / "training_metrics.json", training_metrics)
        _write_json(output_dir / "loss_history.json", loss_history)
        _write_json(output_dir / "validation_progression.json", validation_progression)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    finally:
        if model is not None:
            del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            synchronize_cuda()

    print("Loading fresh quantized base and reloading M6 adapter...", flush=True)
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
        "adapter_config_path": str(output_dir / "adapter_config.json"),
        "base_model_name_or_path": config["model_id"],
        "merged_into_base": False,
        "training_model_discarded_before_reload": True,
        "reload_base_placement": reload_placement,
    }
    _write_json(output_dir / "reload_verification.json", reload_verification)
    tokenizer = load_tokenizer(sft_config)
    post_metrics: dict[str, Any] = {}
    m6_failures: list[dict[str, Any]] = []
    for split in ("validation", "id_test", "hard_test", "ood_test"):
        print(f"Evaluating reloaded M6 adapter on {split}...", flush=True)
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
        m6_failures.extend(failures)
    del reloaded_model
    del reload_base
    gc.collect()
    torch.cuda.empty_cache()

    _write_json(output_dir / "post_training_metrics.json", {
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "metrics": post_metrics,
        "test_metrics_started_after_training": True,
    })
    _failure_file(output_dir / "failures.jsonl", m6_failures)
    baseline_dir = Path("outputs/baseline_v2")
    base_metrics = json.loads((baseline_dir / "metrics.json").read_text(encoding="utf-8"))["splits"]
    m5_metrics = json.loads((Path("outputs/realistic_qlora_v1") / "post_training_metrics.json").read_text(encoding="utf-8"))["metrics"]
    comparison = _build_comparison(base_metrics, m5_metrics, post_metrics, splits)
    _write_json(output_dir / "base_vs_m5_vs_m6.json", comparison)
    _write_json(output_dir / "environment_metadata.json", {
        **_environment(),
        "git_state_before_run": preflight["git_state_before_run"],
        "dataset_fingerprint": preflight["dataset_fingerprint"],
        "evaluation_contract_fingerprint": EXPECTED_EVALUATION_FINGERPRINT,
        "m5_test_metrics_not_read_during_training": True,
    })
    _write_json(output_dir / "artifact_metadata.json", {
        **m6_artifact_metadata(output_dir),
        "m6_specific_files": [
            "experiment_diff.json",
            "shuffled_order.json",
            "order_audit.json",
            "optimizer_steps.jsonl",
            "validation_progression.json",
            "base_vs_m5_vs_m6.json",
        ],
    })
    _write_json(output_dir / "m6_summary.json", {
        "status": "success",
        "order_audit": json.loads((output_dir / "order_audit.json").read_text()),
        "training": training_metrics,
        "validation_progression": validation_progression,
        "post_training_metrics": {
            split: _selected_metrics(post_metrics[split]) for split in post_metrics
        },
        "base_vs_m5_vs_m6": comparison,
        "reload_verification": reload_verification,
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
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
