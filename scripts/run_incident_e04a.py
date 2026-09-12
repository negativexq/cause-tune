#!/usr/bin/env python3
"""Execute one frozen Experiment 04A variant through the canonical incident loop."""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
from pathlib import Path
from typing import Any

import torch

from causetune.evidence import initialize_evidence, record_checkpoint_selection, record_training_result, finalize_evidence
from causetune.experiment_contract import load_experiment_contract
from causetune.incident_benchmark import packet_evidence_ids
from causetune.incident_evaluation import evaluate_incidents
from causetune.incident_training import preprocess_incident_records
from causetune.model import adapter_parameter_count, attach_lora, load_adapter, load_quantized_base, load_tokenizer_for_model
from causetune.verify import score_incident_predictions
from run_incident_qlora import _generate, _resolved_sft_config, _train


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _training_manifest(contract: Any, train_dir: Path, examples: int) -> dict[str, Any]:
    policy = contract.training
    optimizer = policy.optimizer
    return {
        "model_id": contract.model.model_id,
        "seeds": {"train": policy.seed},
        "max_epochs": optimizer.max_epochs,
        "training_sources": {"train": str(train_dir), "validation": str(contract.data["validation"].path)},
        "training": {
            "max_sequence_length": optimizer.max_sequence_length,
            "micro_batch_size": optimizer.micro_batch_size,
            "gradient_accumulation_steps": optimizer.gradient_accumulation_steps,
            "learning_rate": optimizer.learning_rate,
            "gradient_checkpointing": optimizer.gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": optimizer.gradient_checkpointing_use_reentrant,
            "assistant_only_supervision": True,
            "effective_batch_size": optimizer.micro_batch_size * optimizer.gradient_accumulation_steps,
        },
        "quantization": contract.training.quantization.to_dict(),
        "lora": contract.training.lora.to_dict(),
        "subset_examples": examples,
    }


def _enrich_predictions(metrics: dict[str, Any], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata = {row["incident_id"]: row["metadata"] for row in records}
    packets = {row["incident_id"]: row["incident_packet"] for row in records}
    enriched = []
    for row in metrics["predictions"]:
        item = dict(row)
        item["input_metadata"] = {
            "present_components": metadata[row["incident_id"]]["present_components"],
            "available_evidence_ids": sorted(packet_evidence_ids(packets[row["incident_id"]])),
        }
        enriched.append(item)
    return enriched


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir")
    args = parser.parse_args()
    contract = load_experiment_contract(args.config)
    run_dir = Path(args.run_dir or contract.output.output_dir)
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"refusing to reuse run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    initialize_evidence(contract, run_dir, hardware_snapshot=True)
    train_dir = Path(contract.data["train"].path)
    validation_dir = Path(contract.data["validation"].path)
    train = _read_jsonl(train_dir / "train.jsonl")
    train_truth = _read_jsonl(train_dir / "ground_truth_train.jsonl")
    validation = _read_jsonl(validation_dir / "validation.jsonl")
    validation_truth = _read_jsonl(validation_dir / "ground_truth_validation.jsonl")
    eval_config = json.loads(Path("configs/incident_diagnosis_eval.json").read_text(encoding="utf-8"))
    system = eval_config["evaluation_contract"]["system_instruction"]
    manifest = _training_manifest(contract, train_dir, len(train))
    runner_dir = run_dir / "runner"
    runner_dir.mkdir()
    _write_json(runner_dir / "resolved_manifest.json", manifest)
    random.seed(contract.training.seed)
    torch.manual_seed(contract.training.seed)
    tokenizer = load_tokenizer_for_model(contract.model.model_id, revision=contract.model.revision)
    max_length = contract.training.optimizer.max_sequence_length
    if max_length is None:
        raise ValueError("E04 incident contract requires a finite sequence length")
    train_examples = preprocess_incident_records(train, train_truth, tokenizer, max_length, system)
    val_examples = preprocess_incident_records(validation, validation_truth, tokenizer, max_length, system)
    config = _resolved_sft_config(manifest, runner_dir)
    model = attach_lora(load_quantized_base(config, revision=contract.model.revision), config)
    trainable, logical = model.get_nb_trainable_parameters()
    _write_json(run_dir / "parameter_provenance.json", {"trainable_parameters": trainable, "logical_parameters": logical, "adapter_parameters": adapter_parameter_count(model), "model_revision": contract.model.revision})
    _train(model, tokenizer, train_examples, train, manifest, validation, validation_truth, val_examples, system, eval_config, runner_dir)
    summary = json.loads((runner_dir / "training_summary.json").read_text(encoding="utf-8"))
    selected_step = int(summary["best_checkpoint_step"])
    selected_path = runner_dir / f"checkpoint-step-{selected_step:06d}"
    record_training_result(run_dir, actual_steps=int(summary["optimizer_steps_executed"]), stop_reason=str(summary["stop_reason"]))
    record_checkpoint_selection(run_dir, checkpoint=selected_step, source="validation")
    shutil.copyfile(runner_dir / "optimizer_steps.jsonl", run_dir / "training_metrics.jsonl")
    shutil.copyfile(runner_dir / "checkpoint_selection.json", run_dir / "runner_checkpoint_selection.json")
    del model, train_examples, val_examples
    gc.collect()
    torch.cuda.empty_cache()
    fresh_base = load_quantized_base(config, revision=contract.model.revision)
    reloaded = load_adapter(fresh_base, str(selected_path))
    reloaded.eval()
    _write_json(run_dir / "reload_verification.json", {"status": "PASS" if adapter_parameter_count(reloaded) > 0 else "FAIL", "adapter_parameters": adapter_parameter_count(reloaded), "checkpoint": selected_step, "model_revision": contract.model.revision})
    raw = _generate(reloaded, tokenizer, validation, system, eval_config["evaluation_contract"]["generation"]["max_new_tokens"], eval_config["evaluation_contract"]["generation"]["batch_size"], progress_label="FINAL_VALIDATION_PROGRESS")
    validation_eval_records = [
        {**row, "slice": row.get("slice", row["metadata"]["difficulty"])} for row in validation
    ]
    metrics = evaluate_incidents(validation_eval_records, {row["incident_id"]: row for row in validation_truth}, raw)
    predictions = _enrich_predictions(metrics, validation_eval_records)
    _write_jsonl(run_dir / "predictions.jsonl", predictions)
    _write_json(run_dir / "evaluation.json", {"schema_version": 1, "scope": "validation", "scorer_version": "incident-scorer-v1", "metrics": {**metrics, "predictions": predictions}})
    _write_json(run_dir / "training_summary.json", summary)
    verification_metrics = score_incident_predictions(run_dir / "predictions.jsonl")
    if verification_metrics != {**metrics, "predictions": predictions}:
        raise RuntimeError("incident scorer did not reproduce persisted validation metrics before finalization")
    finalize_evidence(run_dir)
    print(json.dumps({"status": "PASS", "run_dir": str(run_dir), "checkpoint": selected_step, "validation": {key: value for key, value in metrics.items() if key != "predictions"}}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
