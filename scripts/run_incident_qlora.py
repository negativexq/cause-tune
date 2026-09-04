#!/usr/bin/env python3
"""Run the single controlled Experiment 02B.2 QLoRA specialization."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import random
import time
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from causetune.config import LoraConfig, QuantizationConfig, SFTConfig
from causetune.data.preprocess import PreprocessedExample
from causetune.evaluation import collate_preprocessed
from causetune.incident_benchmark import read_benchmark
from causetune.incident_contract import contract_fingerprint
from causetune.incident_evaluation import evaluate_incidents, failure_patterns, failure_patterns_for_splits
from causetune.incident_training import (
    CheckpointSelectionPolicy,
    checkpoint_path,
    checkpoint_metadata,
    earliest_within_tolerance,
    preprocess_incident_records,
    training_fingerprint,
    validate_training_manifest,
    validate_training_split,
    validation_schedule,
)
from causetune.model import adapter_parameter_count, attach_lora, load_adapter, load_quantized_base, load_tokenizer_for_model
from causetune.telemetry import cuda_memory_snapshot, json_safe_memory, synchronize_cuda


EXPECTED_BENCHMARK = "5e9a6d74ba881af7aa1146a23f4b837e139a4e8e5a77dba7fe40c3bb2c9c0a94"
EXPECTED_EVALUATION = "d2dea87fd26a3a8c78a2a6a1b4b6fc583b27070ed2de7e691b53c17e59231cf8"
EXPECTED_TRAIN = "8e660f9380d482b9043a360e4bcbdfe45559f63279882460774c6a6a68756cc0"
EXPECTED_VALIDATION = "1459fdbe699aa98acbe2c79a9efc3d9b22afb3d77dc53781861eb853aed0cb52"
EXPECTED_LOGICAL = 4_055_498_240
EXPECTED_TRAINABLE = 33_030_144
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def _resolved_sft_config(manifest: dict[str, Any], output_dir: Path) -> SFTConfig:
    train = manifest["training"]
    return SFTConfig(
        model_id=manifest["model_id"], seed=manifest["seeds"]["train"],
        dataset_path=manifest["training_sources"]["train"], max_examples=2400,
        train_ratio=1.0, validation_ratio=0.0, test_ratio=0.0,
        max_sequence_length=train["max_sequence_length"], micro_batch_size=train["micro_batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation_steps"], learning_rate=train["learning_rate"],
        num_epochs=manifest["max_epochs"], quantization=QuantizationConfig(**manifest["quantization"]),
        lora=LoraConfig(rank=manifest["lora"]["rank"], alpha=manifest["lora"]["alpha"], dropout=manifest["lora"]["dropout"], target_modules=tuple(manifest["lora"]["target_modules"])),
        gradient_checkpointing=train["gradient_checkpointing"], gradient_checkpointing_use_reentrant=train["gradient_checkpointing_use_reentrant"], output_dir=str(output_dir),
    )


def _preflight(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    if output_dir.exists():
        raise FileExistsError(f"refusing a second 02B.2 run; output exists: {output_dir}")
    dataset_dir = Path("data/incident_diagnosis_training")
    manifest = json.loads((dataset_dir / "resolved_training_manifest.json").read_text(encoding="utf-8"))
    preparation_manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    validate_training_manifest(manifest, expected_benchmark_fingerprint=EXPECTED_BENCHMARK, expected_evaluation_fingerprint=EXPECTED_EVALUATION)
    if manifest["train_fingerprint"] != EXPECTED_TRAIN or manifest["validation_fingerprint"] != EXPECTED_VALIDATION:
        raise ValueError("02B.1 dataset fingerprint mismatch")
    if manifest["training"]["max_sequence_length"] != 768 or manifest["training"]["micro_batch_size"] != 1 or manifest["training"]["gradient_accumulation_steps"] != 8 or manifest["training"]["learning_rate"] != 0.0002 or manifest["max_epochs"] != 2:
        raise ValueError("resolved 02B.1 training configuration mismatch")
    eval_config = json.loads(Path("configs/incident_diagnosis_eval.json").read_text(encoding="utf-8"))
    if contract_fingerprint(eval_config) != EXPECTED_EVALUATION or manifest["frozen_evaluation_fingerprint"] != EXPECTED_EVALUATION:
        raise ValueError("frozen evaluation fingerprint mismatch")
    train = _read_jsonl(dataset_dir / "train.jsonl")
    validation = _read_jsonl(dataset_dir / "validation.jsonl")
    train_truth = _read_jsonl(dataset_dir / "ground_truth_train.jsonl")
    validation_truth = _read_jsonl(dataset_dir / "ground_truth_validation.jsonl")
    validate_training_split(train, train_truth, expected_count=2400, expected_per_family=200)
    validate_training_split(validation, validation_truth, expected_count=288, expected_per_family=24)
    if training_fingerprint(train, train_truth) != EXPECTED_TRAIN or training_fingerprint(validation, validation_truth) != EXPECTED_VALIDATION:
        raise ValueError("training/validation files differ from frozen 02B.1 fingerprints")
    if preparation_manifest["token_stats"]["train"]["exceeding_max_sequence_length"] != 0 or preparation_manifest["token_stats"]["train"]["zero_supervised_examples"] != 0 or preparation_manifest["token_stats"]["train"]["truncated_examples"] != 0:
        raise ValueError("unsafe frozen preprocessing statistics")
    if preparation_manifest["token_stats"]["validation"]["exceeding_max_sequence_length"] != 0 or preparation_manifest["token_stats"]["validation"]["zero_supervised_examples"] != 0 or preparation_manifest["token_stats"]["validation"]["truncated_examples"] != 0:
        raise ValueError("unsafe frozen validation preprocessing statistics")
    order = list(range(len(train)))
    random.Random(manifest["seeds"]["train"]).shuffle(order)
    windows = [order[index:index + 8] for index in range(0, len(order), 8)]
    window_labels = [[train[index]["metadata"]["difficulty"] + ":" + next(item["failure_mode"] for item in train_truth if item["incident_id"] == train[index]["incident_id"]) for index in window] for window in windows]
    family_counts = Counter(label.split(":", 1)[1] for labels in window_labels for label in labels)
    single = sum(len(set(labels)) == 1 for labels in window_labels)
    if single != 0:
        raise RuntimeError("deterministic training shuffle yielded a single-family optimizer window")
    output_dir.mkdir(parents=True, exist_ok=False)
    preflight = {
        "status": "pass", "training_started": False, "optimizer_steps": 0, "model_weights_loaded": False,
        "dataset_fingerprints": {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION, "benchmark": EXPECTED_BENCHMARK, "evaluation": EXPECTED_EVALUATION},
        "counts": {"train": len(train), "validation": len(validation)}, "sequence_length": 768,
        "training_order": {"algorithm": "random.Random(seed).shuffle", "seed": manifest["seeds"]["train"], "optimizer_windows": len(windows), "single_family_windows": single, "mixed_family_windows": len(windows) - single, "family_counts": dict(sorted(family_counts.items())), "example_ids": [train[index]["incident_id"] for index in order]},
        "final_25_window_family_counts": [dict(Counter(label.split(":", 1)[1] for label in labels)) for labels in window_labels[-25:]],
    }
    _write_json(output_dir / "preflight.json", preflight)
    _write_json(output_dir / "resolved_manifest.json", manifest)
    return preflight, eval_config, train, validation, train_truth, validation_truth, order


def _prompt_ids(tokenizer: Any, packet: str, system_instruction: str) -> list[int]:
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


def _generate(model: Any, tokenizer: Any, records: list[dict[str, Any]], system_instruction: str, max_new_tokens: int, batch_size: int, progress_label: str | None = None) -> dict[str, str]:
    was_training = bool(model.training)
    old_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    # Gradient checkpointing requires use_cache=False for training, but
    # generation with that setting disables the KV cache and can make a full
    # validation pass appear hung.  Temporarily enable the inference cache and
    # restore the exact training configuration in finally.
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = True
    device = next(model.parameters()).device
    outputs: dict[str, str] = {}
    try:
        with torch.inference_mode():
            for start in range(0, len(records), batch_size):
                chunk = records[start:start + batch_size]
                prompts = [_prompt_ids(tokenizer, item["incident_packet"], system_instruction) for item in chunk]
                width = max(len(prompt) for prompt in prompts)
                input_ids = torch.full((len(prompts), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
                attention = torch.zeros_like(input_ids)
                for row, prompt in enumerate(prompts):
                    input_ids[row, width - len(prompt):] = torch.tensor(prompt, dtype=torch.long, device=device)
                    attention[row, width - len(prompt):] = 1
                synchronize_cuda()
                generated = model.generate(input_ids=input_ids, attention_mask=attention, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                synchronize_cuda()
                for item, sequence in zip(chunk, generated):
                    outputs[item["incident_id"]] = tokenizer.decode(sequence[width:], skip_special_tokens=True)
                processed = min(start + len(chunk), len(records))
                if progress_label and (processed == len(records) or processed % 48 == 0):
                    print(f"{progress_label} examples={processed}/{len(records)}", flush=True)
                del generated, input_ids, attention, prompts
        return outputs
    finally:
        if old_use_cache is not None:
            model.config.use_cache = old_use_cache
        model.train(was_training)


def _eval_records(model: Any, tokenizer: Any, records: list[dict[str, Any]], truths: list[dict[str, Any]], examples: list[PreprocessedExample], system_instruction: str, eval_config: dict[str, Any], progress_label: str | None = None) -> tuple[dict[str, Any], dict[str, str]]:
    raw = _generate(model, tokenizer, records, system_instruction, eval_config["evaluation_contract"]["generation"]["max_new_tokens"], eval_config["evaluation_contract"]["generation"]["batch_size"], progress_label=progress_label)
    eval_records = [{"incident_id": item["incident_id"], "slice": item.get("slice", item["metadata"]["difficulty"]), "incident_packet": item["incident_packet"], "metadata": item["metadata"]} for item in records]
    metrics = evaluate_incidents(eval_records, {truth["incident_id"]: truth for truth in truths}, raw)
    return metrics, raw


def _validation(model: Any, tokenizer: Any, records: list[dict[str, Any]], truths: list[dict[str, Any]], examples: list[PreprocessedExample], system_instruction: str, eval_config: dict[str, Any], step: int) -> dict[str, Any]:
    was_training = bool(model.training)
    started = time.perf_counter()
    print(f"VALIDATION_START step={step}", flush=True)
    try:
        metrics, _ = _eval_records(model, tokenizer, records, truths, examples, system_instruction, eval_config, f"VALIDATION_PROGRESS step={step}")
        result = {"step": step, "validation_duration_seconds": time.perf_counter() - started, "teacher_forced_loss": float(_teacher_loss(model, tokenizer, examples)["loss"]), "diagnosis_exact_match": metrics["diagnosis_exact_match"]["rate"], "diagnosis_exact_count": metrics["diagnosis_exact_match"]["count"], "resolution_exact_match": metrics["resolution_exact_match"]["rate"], "culprit_accuracy": metrics["culprit_accuracy"]["rate"], "failure_mode_accuracy": metrics["failure_mode_accuracy"]["rate"], "failure_mode_macro_f1": metrics["failure_mode_macro_f1"], "action_accuracy": metrics["recommended_action_accuracy"]["rate"], "evidence_f1": metrics["evidence"]["f1"], "strict_json_compliance": metrics["json_compliance"]["rate"], "valid_json": metrics["json_valid_rate"]}
        print(f"VALIDATION_END step={step} duration={result['validation_duration_seconds']:.2f}", flush=True)
        return result
    finally:
        # Generation and teacher-forced loss both enter eval mode. Restore the
        # caller's state before its next gradient-bearing microbatch.
        model.train(was_training)


def _teacher_loss(model: Any, tokenizer: Any, examples: list[PreprocessedExample]) -> dict[str, float | int]:
    from torch.nn import CrossEntropyLoss
    model.eval(); total = 0.0; count = 0; device = next(model.parameters()).device
    loader = DataLoader(examples, batch_size=1, shuffle=False, collate_fn=lambda batch: collate_preprocessed(batch, tokenizer.pad_token_id))
    criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]).logits[..., :-1, :].contiguous()
            labels = batch["labels"][..., 1:].contiguous()
            total += float(criterion(logits.float().view(-1, logits.shape[-1]), labels.view(-1)))
            count += int((labels != -100).sum())
    return {"loss": total / count, "tokens": count}


def _train(model: Any, tokenizer: Any, train_examples: list[PreprocessedExample], train_records: list[dict[str, Any]], manifest: dict[str, Any], val_records: list[dict[str, Any]], val_truth: list[dict[str, Any]], val_examples: list[PreprocessedExample], system_instruction: str, eval_config: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    config = manifest["training"]
    order = list(range(len(train_examples))); random.Random(manifest["seeds"]["train"]).shuffle(order)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=config["learning_rate"])
    optimizer.zero_grad(set_to_none=True)
    policy = CheckpointSelectionPolicy(min_delta=0.005, patience=3, warmup_floor_steps=50)
    validation_history: list[dict[str, Any]] = []
    loss_history: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    print("TRAINING_START", flush=True)
    step_zero = _validation(model, tokenizer, val_records, val_truth, val_examples, system_instruction, eval_config, 0)
    validation_history.append(step_zero); policy.observe(0, step_zero)
    _write_jsonl(output_dir / "validation_progression.jsonl", validation_history)
    # Validation leaves the model in eval mode.  Restore the training state
    # explicitly before the first gradient-bearing microbatch.
    model.train()
    device = next(model.parameters()).device
    start_time = time.perf_counter(); synchronize_cuda();
    processed_examples = processed_input = processed_supervised = optimizer_steps = 0
    peak = {"allocated_gib": 0.0, "reserved_gib": 0.0}
    stop_reason = "max_optimizer_steps"
    max_steps = (len(train_examples) // config["gradient_accumulation_steps"]) * manifest["max_epochs"]
    for epoch in range(manifest["max_epochs"]):
        ordered = [train_examples[index] for index in order]
        micro_losses: list[float] = []; window_ids: list[str] = []; window_labels: list[str] = []; window_input = window_supervised = 0; window_start = 1
        loader = DataLoader(ordered, batch_size=config["micro_batch_size"], shuffle=False, collate_fn=lambda batch: collate_preprocessed(batch, tokenizer.pad_token_id))
        for micro_step, batch in enumerate(loader, 1):
            batch_start = micro_step - 1
            current = ordered[batch_start]
            batch = {key: value.to(device) for key, value in batch.items()}
            synchronize_cuda(); step_start = time.perf_counter()
            result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = result.loss; loss_value = float(loss.detach())
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at micro-step {micro_step}")
            micro_losses.append(loss_value); window_ids.append(current.example_id); window_labels.append(json.loads(current.messages[-1]["content"])["failure_mode"]); input_tokens = int(batch["attention_mask"].sum()); supervised = int((batch["labels"][..., 1:] != -100).sum()); processed_examples += len(batch["attention_mask"]); processed_input += input_tokens; processed_supervised += supervised; window_input += input_tokens; window_supervised += supervised
            (loss / config["gradient_accumulation_steps"]).backward()
            if micro_step % config["gradient_accumulation_steps"] == 0:
                squared = None
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        value = (parameter.grad.detach().float() ** 2).sum(); squared = value if squared is None else squared + value
                grad_norm = float(torch.sqrt(squared).item()) if squared is not None else None
                optimizer.step(); optimizer.zero_grad(set_to_none=True); optimizer_steps += 1; synchronize_cuda(); print(f"OPTIMIZER_STEP {optimizer_steps}", flush=True)
                duration = time.perf_counter() - step_start; step_loss = sum(micro_losses) / len(micro_losses); mem = json_safe_memory(cuda_memory_snapshot()); peak["allocated_gib"] = max(peak["allocated_gib"], mem["peak_allocated_gib"]); peak["reserved_gib"] = max(peak["reserved_gib"], mem["peak_reserved_gib"])
                row = {"optimizer_step": optimizer_steps, "epoch": epoch + 1, "epoch_progress": micro_step / len(loader), "loss": step_loss, "learning_rate": float(optimizer.param_groups[0]["lr"]), "example_ids": window_ids, "labels": window_labels, "label_counts": dict(Counter(window_labels)), "microbatch_range": {"start": window_start, "end": micro_step}, "input_tokens": window_input, "supervised_tokens": window_supervised, "step_duration_seconds": duration, "cumulative_duration_seconds": time.perf_counter() - start_time, "gradient_norm": grad_norm, "allocated_vram": mem["allocated_gib"], "reserved_vram": mem["reserved_gib"]}
                step_rows.append(row); loss_history.append({"optimizer_step": optimizer_steps, "loss": step_loss})
                if optimizer_steps % 10 == 0: print(f"step {optimizer_steps}/{max_steps} loss={step_loss:.6f} labels={dict(Counter(window_labels))} lr={row['learning_rate']:.2e} grad_norm={grad_norm} vram={mem['allocated_gib']:.3f}/{mem['reserved_gib']:.3f}GiB", flush=True)
                if optimizer_steps % 25 == 0:
                    val = _validation(model, tokenizer, val_records, val_truth, val_examples, system_instruction, eval_config, optimizer_steps); validation_history.append(val); _write_jsonl(output_dir / "validation_progression.jsonl", validation_history)
                    model.train()
                    checkpoint = checkpoint_path(output_dir, optimizer_steps); checkpoint.mkdir(parents=True, exist_ok=False); model.save_pretrained(checkpoint, safe_serialization=True); _write_json(checkpoint / "checkpoint_metadata.json", checkpoint_metadata(step=optimizer_steps, validation_metrics=val, resolved_manifest=manifest) | {"adapter_only": True})
                    if policy.observe(optimizer_steps, val): stop_reason = policy.stop_reason or "validation_no_improvement"; break
                micro_losses.clear(); window_ids.clear(); window_labels.clear(); window_input = window_supervised = 0; window_start = micro_step + 1
        if stop_reason != "max_optimizer_steps": break
    synchronize_cuda(); duration = time.perf_counter() - start_time
    optimizer.zero_grad(set_to_none=True); del optimizer
    _write_jsonl(output_dir / "optimizer_steps.jsonl", step_rows); _write_json(output_dir / "loss_history.json", loss_history)
    best = policy.best_step
    if best is None or best == 0:
        raise RuntimeError("no post-training checkpoint was selected by validation")
    near = earliest_within_tolerance(validation_history)
    validation_duration = sum(float(row.get("validation_duration_seconds", 0.0)) for row in validation_history)
    summary = {"optimizer_steps_executed": optimizer_steps, "max_optimizer_steps": max_steps, "examples_processed": processed_examples, "input_tokens_processed": processed_input, "supervised_tokens_processed": processed_supervised, "wall_clock_training_seconds": duration, "validation_wall_clock_seconds": validation_duration, "input_tokens_per_second": processed_input / duration, "supervised_tokens_per_second": processed_supervised / duration, "peak_vram": peak, "initial_loss": loss_history[0]["loss"], "final_loss": loss_history[-1]["loss"], "min_loss": min(row["loss"] for row in loss_history), "max_loss": max(row["loss"] for row in loss_history), "mean_loss": sum(row["loss"] for row in loss_history) / len(loss_history), "all_losses_finite": all(torch.isfinite(torch.tensor(row["loss"])) for row in loss_history), "actual_stop_step": optimizer_steps, "stop_reason": stop_reason, "best_checkpoint_step": best, "best_validation_metric": policy.best_metrics["diagnosis_exact_match"], "early_stopping_state": policy.state(), "earliest_near_best": near, "optimizer": "AdamW", "scheduler": "none", "weight_decay": 0.0, "gradient_clipping": None, "warmup": 0}
    _write_json(output_dir / "training_summary.json", summary); _write_json(output_dir / "checkpoint_selection.json", {"policy": policy.state(), "validation_history": validation_history, "earliest_within_tolerance": near})
    return summary


def main() -> None:
    experiment_started = time.perf_counter()
    parser = argparse.ArgumentParser(); parser.add_argument("--output-dir", default="outputs/incident_diagnosis_02b2"); args = parser.parse_args(); output_dir = Path(args.output_dir)
    preflight, eval_config, train, validation, train_truth, val_truth, order = _preflight(output_dir)
    print(json.dumps({"preflight": preflight, "status": "passed; loading QLoRA model"}, indent=2, sort_keys=True), flush=True)
    random.seed(20260941); torch.manual_seed(20260941)
    tokenizer = load_tokenizer_for_model("Qwen/Qwen3-4B")
    system = eval_config["evaluation_contract"]["system_instruction"]
    train_examples = preprocess_incident_records(train, train_truth, tokenizer, 768, system)
    val_examples = preprocess_incident_records(validation, val_truth, tokenizer, 768, system)
    model = attach_lora(load_quantized_base(_resolved_sft_config(json.loads((Path("data/incident_diagnosis_training") / "resolved_training_manifest.json").read_text()), output_dir)), _resolved_sft_config(json.loads((Path("data/incident_diagnosis_training") / "resolved_training_manifest.json").read_text()), output_dir))
    devices = sorted({str(parameter.device) for parameter in model.parameters()}); trainable, logical = model.get_nb_trainable_parameters(); packed = sum(parameter.numel() for parameter in model.parameters()); target_presence = {target: any(name.endswith("." + target) for name, _ in model.named_modules()) for target in TARGET_MODULES}; trainable_names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]; bad_trainable = [name for name in trainable_names if "lora_" not in name]
    if trainable != EXPECTED_TRAINABLE or logical != EXPECTED_LOGICAL or any(not value for value in target_presence.values()) or bad_trainable or any(not str(device).startswith("cuda") for device in devices) or not torch.cuda.is_bf16_supported():
        raise RuntimeError(json.dumps({"trainable": trainable, "logical": logical, "packed_parameter_numel": packed, "target_presence": target_presence, "unexpected_trainable": bad_trainable, "devices": devices, "bf16_supported": torch.cuda.is_bf16_supported()}))
    preflight.update({"model_weights_loaded": True, "training_started": False, "parameter_structure": {"trainable": trainable, "logical": logical, "trainable_percent": trainable / logical * 100, "packed_parameter_numel": packed, "target_presence": target_presence, "base_frozen": not bad_trainable, "lora_trainable": len(trainable_names) > 0, "devices": devices, "bf16_supported": torch.cuda.is_bf16_supported()}}); _write_json(output_dir / "preflight.json", preflight)
    summary = _train(model, tokenizer, train_examples, train, manifest=json.loads((Path("data/incident_diagnosis_training") / "resolved_training_manifest.json").read_text()), val_records=validation, val_truth=val_truth, val_examples=val_examples, system_instruction=system, eval_config=eval_config, output_dir=output_dir)
    selected_path = checkpoint_path(output_dir, summary["best_checkpoint_step"])
    adapter_meta = json.loads((selected_path / "checkpoint_metadata.json").read_text()); _write_json(output_dir / "selected_adapter_metadata.json", {"selected_path": str(selected_path), "checkpoint_metadata": adapter_meta, "adapter_parameter_count_in_training_model": adapter_parameter_count(model)})
    del model, train_examples, val_examples; gc.collect(); torch.cuda.empty_cache();
    fresh_base = load_quantized_base(_resolved_sft_config(json.loads((Path("data/incident_diagnosis_training") / "resolved_training_manifest.json").read_text()), output_dir)); reloaded = load_adapter(fresh_base, str(selected_path)); reloaded.eval(); reload_ok = adapter_parameter_count(reloaded) > 0
    _write_json(output_dir / "reload_verification.json", {"status": "pass" if reload_ok else "fail", "base_model": "Qwen/Qwen3-4B", "adapter_path": str(selected_path), "adapter_parameter_count": adapter_parameter_count(reloaded), "checkpoint_step": summary["best_checkpoint_step"], "train_fingerprint": EXPECTED_TRAIN, "validation_fingerprint": EXPECTED_VALIDATION, "benchmark_fingerprint": EXPECTED_BENCHMARK, "evaluation_fingerprint": EXPECTED_EVALUATION, "selected_validation_metrics": adapter_meta["validation_metrics"], "lora_config": json.loads((selected_path / "adapter_config.json").read_text())})
    benchmark_inputs, benchmark_truth_rows, benchmark_manifest = read_benchmark("data/incident_diagnosis"); assert benchmark_manifest["benchmark_fingerprint"] == EXPECTED_BENCHMARK
    truth_by_id = {truth["incident_id"]: truth for truth in benchmark_truth_rows}; all_inputs = [item for split in ("standard", "hard", "transfer") for item in benchmark_inputs[split]]; all_truth = [truth_by_id[item["incident_id"]] for item in all_inputs]
    raw = _generate(reloaded, tokenizer, all_inputs, system, eval_config["evaluation_contract"]["generation"]["max_new_tokens"], eval_config["evaluation_contract"]["generation"]["batch_size"])
    _write_jsonl(output_dir / "tuned_predictions.jsonl", [{"incident_id": item["incident_id"], "raw_output": raw[item["incident_id"]]} for item in all_inputs])
    tuned_splits = {}; all_pre = preprocess_incident_records(all_inputs, all_truth, tokenizer, 768, system)
    for split in ("standard", "hard", "transfer"):
        rows = benchmark_inputs[split]; truths = [truth_by_id[item["incident_id"]] for item in rows]; examples = [example for example in all_pre if example.example_id in {item["incident_id"] for item in rows}]; metrics = evaluate_incidents(rows, {truth["incident_id"]: truth for truth in truths}, {item["incident_id"]: raw[item["incident_id"]] for item in rows}); metrics["teacher_forced_loss"] = _teacher_loss(reloaded, tokenizer, examples)["loss"]; tuned_splits[split] = metrics
    tuned_splits["all"] = evaluate_incidents(all_inputs, truth_by_id, raw)
    _write_json(output_dir / "tuned_metrics.json", {"benchmark_fingerprint": EXPECTED_BENCHMARK, "evaluation_fingerprint": EXPECTED_EVALUATION, "prediction_count": len(raw), "splits": tuned_splits, "failure_patterns": failure_patterns_for_splits(benchmark_inputs, tuned_splits, all_inputs)})
    base = json.loads(Path("results/incident_diagnosis_base.json").read_text()); base_predictions = {row["incident_id"]: row for row in _read_jsonl(Path("outputs/incident_diagnosis_base/predictions.jsonl"))}; transitions = Counter(); transition_rows = []
    tuned_predictions = {row["incident_id"]: row for row in tuned_splits["all"]["predictions"]}
    for item in all_inputs:
        bid = base_predictions[item["incident_id"]]; tuned_row = tuned_predictions[item["incident_id"]]
        row_transitions = {}
        for metric_name, base_field, tuned_field in (("diagnosis", "diagnosis_exact", "diagnosis_exact"), ("resolution", "resolution_exact", "resolution_exact"), ("action", "action_correct", "action_correct")):
            base_correct = bool(bid[base_field]); tuned_correct = bool(tuned_row[tuned_field]); key = f"base_{'correct' if base_correct else 'wrong'} -> tuned_{'correct' if tuned_correct else 'wrong'}"; transitions[f"{metric_name}:{key}"] += 1; row_transitions[f"{metric_name}_transition"] = key
        transition_rows.append({"incident_id": item["incident_id"], "base_diagnosis_exact": bool(bid["diagnosis_exact"]), "tuned_diagnosis_exact": bool(tuned_row["diagnosis_exact"]), **row_transitions})
    family_comparison = {}
    for family, base_family in base["splits"]["all"]["per_failure_family"].items():
        tuned_family = tuned_splits["all"]["per_failure_family"][family]
        family_comparison[family] = {"base": base_family, "tuned": tuned_family, "delta_pp": {"diagnosis_exact_match": (tuned_family["diagnosis_exact_match"] - base_family["diagnosis_exact_match"]) * 100, "resolution_exact_match": (tuned_family["resolution_exact_match"] - base_family["resolution_exact_match"]) * 100, "culprit_accuracy": (tuned_family["culprit_accuracy"] - base_family["culprit_accuracy"]) * 100, "failure_mode_accuracy": (tuned_family["failure_mode_accuracy"] - base_family["failure_mode_accuracy"]) * 100, "action_accuracy": (tuned_family["action_accuracy"] - base_family["action_accuracy"]) * 100, "json_compliance": (tuned_family["json_compliance"] - base_family["json_compliance"]) * 100}}
    comparison = {"base": base["splits"], "tuned": {split: {key: value for key, value in metrics.items() if key != "predictions"} for split, metrics in tuned_splits.items()}, "failure_patterns": {"base": failure_patterns(all_inputs, base["splits"]["all"]), "tuned": failure_patterns(all_inputs, tuned_splits["all"])}, "transitions": dict(transitions), "transition_rows": transition_rows, "per_family": family_comparison, "training": summary, "selected_checkpoint": summary["best_checkpoint_step"]}; _write_json(output_dir / "base_vs_tuned.json", comparison)
    base_all = base["splits"]["all"]; tuned_all = tuned_splits["all"]
    site_summary = {"experiment": "production_incident_diagnosis_specialization", "model": "Qwen/Qwen3-4B", "base_diagnosis_exact": base_all["diagnosis_exact_match"]["rate"], "tuned_diagnosis_exact": tuned_all["diagnosis_exact_match"]["rate"], "diagnosis_gain_pp": (tuned_all["diagnosis_exact_match"]["rate"] - base_all["diagnosis_exact_match"]["rate"]) * 100, "base_resolution_exact": base_all["resolution_exact_match"]["rate"], "tuned_resolution_exact": tuned_all["resolution_exact_match"]["rate"], "resolution_gain_pp": (tuned_all["resolution_exact_match"]["rate"] - base_all["resolution_exact_match"]["rate"]) * 100, "base_hard": base["splits"]["hard"]["diagnosis_exact_match"]["rate"], "tuned_hard": tuned_splits["hard"]["diagnosis_exact_match"]["rate"], "hard_gain_pp": (tuned_splits["hard"]["diagnosis_exact_match"]["rate"] - base["splits"]["hard"]["diagnosis_exact_match"]["rate"]) * 100, "base_transfer": base["splits"]["transfer"]["diagnosis_exact_match"]["rate"], "tuned_transfer": tuned_splits["transfer"]["diagnosis_exact_match"]["rate"], "transfer_gain_pp": (tuned_splits["transfer"]["diagnosis_exact_match"]["rate"] - base["splits"]["transfer"]["diagnosis_exact_match"]["rate"]) * 100, "base_action_accuracy": base_all["recommended_action_accuracy"]["rate"], "tuned_action_accuracy": tuned_all["recommended_action_accuracy"]["rate"], "action_gain_pp": (tuned_all["recommended_action_accuracy"]["rate"] - base_all["recommended_action_accuracy"]["rate"]) * 100, "max_optimizer_steps": 600, "actual_stop_step": summary["actual_stop_step"], "best_checkpoint_step": summary["best_checkpoint_step"], "earliest_near_best_step": summary["earliest_near_best"]["earliest_step"], "training_seconds": summary["wall_clock_training_seconds"], "validation_seconds": summary["validation_wall_clock_seconds"], "peak_allocated_vram_gib": summary["peak_vram"]["allocated_gib"], "peak_reserved_vram_gib": summary["peak_vram"]["reserved_gib"]}
    _write_json(output_dir / "site_ready_summary.json", site_summary)
    final_summary = {"experiment": "02B.2 Production Incident Diagnosis QLoRA Specialization", "fingerprints": {"train": EXPECTED_TRAIN, "validation": EXPECTED_VALIDATION, "benchmark": EXPECTED_BENCHMARK, "evaluation": EXPECTED_EVALUATION}, "training": summary, "reload": json.loads((output_dir / "reload_verification.json").read_text()), "benchmark_evaluations": {"tuned_generation_passes": 1, "tuned_prediction_count": len(raw), "selected_checkpoint": summary["best_checkpoint_step"]}, "total_experiment_seconds": time.perf_counter() - experiment_started}
    _write_json(output_dir / "final_experiment_summary.json", final_summary)
    print(json.dumps({"status": "complete", "summary": summary, "reload": json.loads((output_dir / "reload_verification.json").read_text()), "output_dir": str(output_dir)}, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
