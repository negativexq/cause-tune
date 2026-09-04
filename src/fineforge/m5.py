"""Pure helpers and integrity checks for the single M5 QLoRA run."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Mapping, Sequence

from .data.preprocess import PreprocessedExample
from .evaluation_contract import contract_fingerprint


EXPECTED_EVALUATION_FINGERPRINT = "1b00a333c26c4cbd03b3e04d990fad3b4adf0d9a03443c9fd5f183ee7e8ef94d"
EXPECTED_COUNTS = {
    "train": 2000,
    "validation": 250,
    "id_test": 250,
    "hard_test": 250,
    "ood_test": 250,
}
TOP_HARD_PAIRS = (
    ("subscription_cancel", "cancel_order"),
    ("order_missing", "refund"),
    ("human_escalation", "order_missing"),
    ("payment_failed", "duplicate_charge"),
    ("fraud_suspected", "duplicate_charge"),
)


def parameter_structure(model: Any, target_modules: Sequence[str]) -> dict[str, Any]:
    """Inspect a PEFT model without confusing packed 4-bit storage with logic.

    ``Parameter.numel()`` reports the physical storage shape of bitsandbytes
    quantized parameters.  PEFT exposes the logical model total through
    ``get_nb_trainable_parameters``; that is the count used for M5's structure
    assertion whenever it is available.
    """

    get_counts = getattr(model, "get_nb_trainable_parameters", None)
    if callable(get_counts):
        trainable, logical_total = get_counts()
        counting_method = "peft.get_nb_trainable_parameters"
    else:
        trainable = sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        logical_total = sum(parameter.numel() for parameter in model.parameters())
        counting_method = "naive_parameter_numel_fallback"

    named_parameters = list(model.named_parameters())
    lora_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if "lora_" in name
    ]
    base_parameters = [
        (name, parameter)
        for name, parameter in named_parameters
        if "lora_" not in name
    ]

    target_presence = {}
    for target in target_modules:
        target_presence[target] = any(
            (name == target or name.endswith(f".{target}"))
            and hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            for name, module in model.named_modules()
        )

    packed_parameter_numel = sum(parameter.numel() for parameter in model.parameters())
    return {
        "counting_method": counting_method,
        "trainable_parameter_count": int(trainable),
        "logical_parameter_count": int(logical_total),
        "packed_parameter_numel": int(packed_parameter_numel),
        "trainable_percentage": (
            float(trainable) / float(logical_total) * 100
            if logical_total
            else 0.0
        ),
        "lora_target_modules_present": target_presence,
        "base_parameters_frozen": bool(base_parameters)
        and all(not parameter.requires_grad for _, parameter in base_parameters),
        "lora_parameters_trainable": bool(lora_parameters)
        and all(parameter.requires_grad for _, parameter in lora_parameters),
        "lora_parameter_count": len(lora_parameters),
        "base_parameter_count": len(base_parameters),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset_fingerprint(dataset_dir: str | Path) -> dict[str, Any]:
    directory = Path(dataset_dir)
    files = {
        name: sha256_file(directory / f"{name}.jsonl")
        for name in EXPECTED_COUNTS
    }
    files["manifest.json"] = sha256_file(directory / "manifest.json")
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {
        "manifest_sha256": files["manifest.json"],
        "files": files,
        "aggregate_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def git_state() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        check=True,
        capture_output=True,
        text=True,
    )
    return {"status": result.stdout.strip()}


def validate_m5_config(config: Mapping[str, Any]) -> str:
    """Validate the one-run M5 configuration and return its contract fingerprint."""

    if config.get("evaluation_contract_fingerprint") != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("M5 evaluation contract fingerprint does not match the frozen M4 fingerprint")
    evaluation_config_path = Path(config["evaluation_config"])
    evaluation_config = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
    actual = contract_fingerprint(evaluation_config)
    if actual != EXPECTED_EVALUATION_FINGERPRINT:
        raise ValueError("referenced baseline evaluation config has a changed fingerprint")
    training = config["training"]
    if training["num_epochs"] != 1 or training["max_sequence_length"] != 1024:
        raise ValueError("M5 requires exactly one epoch and max_sequence_length=1024")
    if training["micro_batch_size"] != 1 or training["gradient_accumulation_steps"] != 8:
        raise ValueError("M5 requires micro_batch_size=1 and gradient_accumulation_steps=8")
    if training["learning_rate"] != 2e-4:
        raise ValueError("M5 requires learning_rate=2e-4")
    if not training["gradient_checkpointing"] or training["gradient_checkpointing_use_reentrant"]:
        raise ValueError("M5 requires non-reentrant gradient checkpointing")
    quant = config["quantization"]
    if quant != {
        "load_in_4bit": True,
        "quant_type": "nf4",
        "compute_dtype": "bfloat16",
        "double_quant": True,
    }:
        raise ValueError("M5 quantization configuration must remain NF4/BF16/double-quantized")
    lora = config["lora"]
    if lora != {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    }:
        raise ValueError("M5 LoRA configuration differs from the verified V1 design")
    return actual


def sequence_summary(examples: Sequence[PreprocessedExample]) -> dict[str, Any]:
    if not examples:
        raise ValueError("cannot summarize an empty example sequence")
    lengths = [len(example.input_ids) for example in examples]
    supervised = [
        sum(label != -100 for label in example.labels[1:])
        for example in examples
    ]
    return {
        "examples": len(examples),
        "model_input_tokens": sum(lengths),
        "supervised_assistant_tokens": sum(supervised),
        "sequence_length": {
            "min": min(lengths),
            "mean": mean(lengths),
            "max": max(lengths),
        },
        "supervised_tokens_per_example": {
            "min": min(supervised),
            "mean": mean(supervised),
            "max": max(supervised),
        },
        "zero_supervised_examples": sum(value == 0 for value in supervised),
    }


def metric_delta(base: Mapping[str, Any], tuned: Mapping[str, Any]) -> dict[str, float]:
    return {
        "accuracy_delta_pp": (float(tuned["intent_accuracy"]) - float(base["intent_accuracy"])) * 100,
        "macro_f1_delta_pp": (float(tuned["macro_f1"]) - float(base["macro_f1"])) * 100,
        "valid_json_delta_pp": (float(tuned["valid_json_rate"]) - float(base["valid_json_rate"])) * 100,
        "schema_compliance_delta_pp": (
            float(tuned["exact_schema_compliance_rate"])
            - float(base["exact_schema_compliance_rate"])
        ) * 100,
        "teacher_loss_delta": float(tuned["teacher_forced_loss"]) - float(base["teacher_forced_loss"]),
    }


def confusion_delta(
    base_matrix: Mapping[str, Mapping[str, int]],
    tuned_matrix: Mapping[str, Mapping[str, int]],
    pairs: Sequence[tuple[str, str]] = TOP_HARD_PAIRS,
) -> dict[str, dict[str, int]]:
    return {
        f"{actual} -> {predicted}": {
            "baseline": int(base_matrix.get(actual, {}).get(predicted, 0)),
            "tuned": int(tuned_matrix.get(actual, {}).get(predicted, 0)),
            "delta": int(tuned_matrix.get(actual, {}).get(predicted, 0))
            - int(base_matrix.get(actual, {}).get(predicted, 0)),
        }
        for actual, predicted in pairs
    }


def adapter_artifact_metadata(output_dir: str | Path) -> dict[str, Any]:
    directory = Path(output_dir)
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "resolved_training_config.json",
        "training_metrics.json",
        "loss_history.json",
        "reload_verification.json",
        "post_training_metrics.json",
        "base_vs_tuned.json",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise ValueError(f"missing M5 artifacts: {missing}")
    return {
        "required_files": list(required),
        "missing_files": missing,
        "adapter_model_sha256": sha256_file(directory / "adapter_model.safetensors"),
    }


def prediction_transitions(
    records: Sequence[Mapping[str, Any]],
    base_predictions: Sequence[Mapping[str, Any]],
    tuned_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not (len(records) == len(base_predictions) == len(tuned_predictions)):
        raise ValueError("prediction transition inputs have different lengths")
    categories: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = {
        "base_wrong_tuned_correct": [],
        "base_correct_tuned_wrong": [],
        "persistent_errors": [],
    }
    for record, base, tuned in zip(records, base_predictions, tuned_predictions):
        base_correct = bool(base["correct"])
        tuned_correct = bool(tuned["correct"])
        if base_correct and tuned_correct:
            continue
        if not base_correct and tuned_correct:
            category = "base_wrong_tuned_correct"
        elif base_correct and not tuned_correct:
            category = "base_correct_tuned_wrong"
        else:
            category = "persistent_errors"
        categories[category] += 1
        examples[category].append({
            "example_id": record["example_id"],
            "expected_intent": record["expected_response"]["intent"],
            "base_prediction": base.get("parsed_intent"),
            "tuned_prediction": tuned.get("parsed_intent"),
            "base_raw_output": base.get("raw_text", ""),
            "tuned_raw_output": tuned.get("raw_text", ""),
            "difficulty": record["difficulty"],
            "phenomena": record["phenomena"],
            "confusable_with": record["confusable_with"],
            "scenario_family": record["scenario_family"],
        })
    return {"counts": dict(categories), "examples": examples}
