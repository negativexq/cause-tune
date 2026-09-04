#!/usr/bin/env python3
"""Fresh-process CUDA execution audit for incident-diagnosis 02B.2.

This diagnostic deliberately has no optimizer.step(), save_pretrained(), or
frozen-benchmark path.  Each model-bearing stage is run in a new interpreter
so a poisoned CUDA context cannot be mistaken for a healthy one.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "incident_diagnosis_training"
OUTPUT = ROOT / "outputs" / "incident_diagnosis_02b2"
EXPECTED = {
    "train": "8e660f9380d482b9043a360e4bcbdfe45559f63279882460774c6a6a68756cc0",
    "validation": "1459fdbe699aa98acbe2c79a9efc3d9b22afb3d77dc53781861eb853aed0cb52",
    "benchmark": "5e9a6d74ba881af7aa1146a23f4b837e139a4e8e5a77dba7fe40c3bb2c9c0a94",
    "evaluation": "d2dea87fd26a3a8c78a2a6a1b4b6fc583b27070ed2de7e691b53c17e59231cf8",
}
TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def memory(torch: Any) -> dict[str, float]:
    return {
        "allocated_gib": torch.cuda.memory_allocated() / 2**30,
        "reserved_gib": torch.cuda.memory_reserved() / 2**30,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def sync(torch: Any) -> None:
    torch.cuda.synchronize()


def model_setup() -> tuple[Any, Any, dict[str, Any], Any]:
    import torch
    from causetune.config import LoraConfig, QuantizationConfig, SFTConfig
    from causetune.model import attach_lora, load_quantized_base, load_tokenizer_for_model

    manifest = json.loads((DATA / "resolved_training_manifest.json").read_text(encoding="utf-8"))
    train = manifest["training"]
    config = SFTConfig(
        model_id=manifest["model_id"], seed=manifest["seeds"]["train"],
        dataset_path=manifest["training_sources"]["train"], max_examples=2400,
        train_ratio=1.0, validation_ratio=0.0, test_ratio=0.0,
        max_sequence_length=train["max_sequence_length"], micro_batch_size=train["micro_batch_size"],
        gradient_accumulation_steps=train["gradient_accumulation_steps"], learning_rate=train["learning_rate"],
        num_epochs=manifest["max_epochs"], quantization=QuantizationConfig(**manifest["quantization"]),
        lora=LoraConfig(rank=manifest["lora"]["rank"], alpha=manifest["lora"]["alpha"], dropout=manifest["lora"]["dropout"], target_modules=tuple(manifest["lora"]["target_modules"])),
        gradient_checkpointing=train["gradient_checkpointing"], gradient_checkpointing_use_reentrant=train["gradient_checkpointing_use_reentrant"], output_dir=str(OUTPUT),
    )
    tokenizer = load_tokenizer_for_model(config.model_id)
    base = load_quantized_base(config)
    model = attach_lora(base, config)
    return model, tokenizer, manifest, torch


def model_report(model: Any, torch: Any) -> dict[str, Any]:
    trainable, logical = model.get_nb_trainable_parameters()
    packed = sum(parameter.numel() for parameter in model.parameters())
    target_presence = {target: any(name.endswith("." + target) for name, _ in model.named_modules()) for target in TARGETS}
    lora_names = [name for name, parameter in model.named_parameters() if "lora_" in name and parameter.requires_grad]
    unexpected = [name for name, parameter in model.named_parameters() if parameter.requires_grad and "lora_" not in name]
    return {
        "trainable": trainable,
        "logical": logical,
        "trainable_percent": trainable / logical * 100,
        "packed_parameter_numel": packed,
        "target_presence": target_presence,
        "lora_trainable_parameter_names": len(lora_names),
        "unexpected_trainable_names": unexpected,
        "base_frozen": not unexpected,
        "lora_trainable": bool(lora_names),
        "devices": sorted({str(parameter.device) for parameter in model.parameters()}),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        "use_cache": bool(getattr(model.config, "use_cache", True)),
        "training_mode": bool(model.training),
    }


def one_example(tokenizer: Any, manifest: dict[str, Any], index: int = 0) -> tuple[Any, Any]:
    from causetune.incident_training import preprocess_incident_records
    from causetune.evaluation import collate_preprocessed

    train = read_jsonl(DATA / "train.jsonl")
    truth = read_jsonl(DATA / "ground_truth_train.jsonl")
    order = list(range(len(train)))
    random.Random(manifest["seeds"]["train"]).shuffle(order)
    record = [train[order[index]]]
    example = preprocess_incident_records(record, truth, tokenizer, 768, json.loads((ROOT / "configs" / "incident_diagnosis_eval.json").read_text())['evaluation_contract']['system_instruction'])[0]
    return record[0], collate_preprocessed([example], tokenizer.pad_token_id)


def batch_on_device(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in batch.items()}


def grad_report(model: Any, torch: Any) -> tuple[bool, bool, float]:
    finite = True
    nonzero = False
    squared = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            grad = parameter.grad.detach()
            finite = finite and bool(torch.isfinite(grad).all())
            nonzero = nonzero or bool(torch.any(grad != 0))
            squared += float((grad.float() ** 2).sum().item())
    return finite, nonzero, squared**0.5


def discard_gradients(model: Any, torch: Any) -> None:
    for parameter in model.parameters():
        parameter.grad = None
    gc.collect()
    torch.cuda.empty_cache()
    sync(torch)


def stage_basic() -> dict[str, Any]:
    import torch
    start = time.perf_counter()
    torch.cuda.init()
    torch.cuda.reset_peak_memory_stats()
    x = torch.randn((1024, 1024), device="cuda", dtype=torch.bfloat16, requires_grad=True)
    sync(torch)
    forward_start = time.perf_counter()
    loss = (x @ x).mean()
    sync(torch)
    forward = time.perf_counter() - forward_start
    backward_start = time.perf_counter()
    loss.backward()
    sync(torch)
    backward = time.perf_counter() - backward_start
    result = {"status": "pass", "duration_seconds": time.perf_counter() - start, "forward_seconds": forward, "backward_seconds": backward, "grad_norm": float(x.grad.float().norm().item()), "memory": memory(torch)}
    del x, loss
    gc.collect(); torch.cuda.empty_cache(); sync(torch)
    return result


def stage_load() -> dict[str, Any]:
    import torch
    torch.cuda.init(); torch.cuda.reset_peak_memory_stats()
    model, _, _, torch = model_setup()
    sync(torch)
    report = model_report(model, torch)
    report["memory"] = memory(torch)
    report["status"] = "pass" if report["trainable"] == 33_030_144 and report["logical"] == 4_055_498_240 and all(report["target_presence"].values()) and report["base_frozen"] and report["lora_trainable"] and all(device.startswith("cuda") for device in report["devices"]) else "fail"
    del model; gc.collect(); torch.cuda.empty_cache(); sync(torch)
    return report


def stage_forward() -> dict[str, Any]:
    import torch
    from causetune.incident_training import preprocess_incident_records
    from causetune.evaluation import collate_preprocessed
    torch.cuda.init(); model, tokenizer, manifest, torch = model_setup()
    record, batch = one_example(tokenizer, manifest)
    device = next(model.parameters()).device
    start = time.perf_counter(); batch = batch_on_device(batch, device); sync(torch)
    forward_start = time.perf_counter(); result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]); sync(torch)
    duration = time.perf_counter() - forward_start
    loss = float(result.loss.detach().cpu())
    output = {"status": "pass", "incident_id": record["incident_id"], "sequence_length": int(batch["input_ids"].shape[-1]), "loss": loss, "duration_seconds": duration, "total_stage_seconds": time.perf_counter() - start, "memory": memory(torch), "training_mode": bool(model.training), "use_cache": bool(model.config.use_cache)}
    del result, batch, model; gc.collect(); torch.cuda.empty_cache(); sync(torch)
    return output


def stage_backward() -> dict[str, Any]:
    import torch
    torch.cuda.init(); model, tokenizer, manifest, torch = model_setup()
    model.train()
    record, batch = one_example(tokenizer, manifest)
    device = next(model.parameters()).device
    print("TRAIN_DIAG: input_ready", flush=True)
    batch = batch_on_device(batch, device); sync(torch)
    print("TRAIN_DIAG: forward_start", flush=True)
    forward_start = time.perf_counter(); result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]); sync(torch); forward = time.perf_counter() - forward_start
    print("TRAIN_DIAG: forward_complete", flush=True)
    print("TRAIN_DIAG: backward_start", flush=True)
    backward_start = time.perf_counter(); result.loss.backward(); sync(torch); backward = time.perf_counter() - backward_start
    print("TRAIN_DIAG: backward_complete", flush=True)
    finite, nonzero, norm = grad_report(model, torch)
    output = {"status": "pass" if finite and nonzero else "fail", "incident_id": record["incident_id"], "sequence_length": int(batch["input_ids"].shape[-1]), "loss": float(result.loss.detach().cpu()), "forward_seconds": forward, "backward_seconds": backward, "gradient_finite": finite, "gradient_nonzero": nonzero, "lora_grad_norm": norm, "memory": memory(torch), "optimizer_step_called": False}
    discard_gradients(model, torch); del result, batch, model; gc.collect(); torch.cuda.empty_cache(); sync(torch)
    return output


def stage_accumulation() -> dict[str, Any]:
    import torch
    torch.cuda.init(); model, tokenizer, manifest, torch = model_setup(); model.train()
    from causetune.incident_training import preprocess_incident_records
    from causetune.evaluation import collate_preprocessed
    train = read_jsonl(DATA / "train.jsonl"); truth = read_jsonl(DATA / "ground_truth_train.jsonl")
    order = list(range(len(train))); random.Random(manifest["seeds"]["train"]).shuffle(order)
    records = [train[index] for index in order[:8]]
    examples = preprocess_incident_records(records, truth, tokenizer, 768, json.loads((ROOT / "configs" / "incident_diagnosis_eval.json").read_text())['evaluation_contract']['system_instruction'])
    device = next(model.parameters()).device; rows = []
    for microbatch, example in enumerate(examples, 1):
        print(f"TRAIN_DIAG: input_ready microbatch={microbatch}", flush=True)
        batch = batch_on_device(collate_preprocessed([example], tokenizer.pad_token_id), device); sync(torch)
        print(f"TRAIN_DIAG: forward_start microbatch={microbatch}", flush=True)
        result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]); sync(torch)
        print(f"TRAIN_DIAG: forward_complete microbatch={microbatch}", flush=True)
        print(f"TRAIN_DIAG: backward_start microbatch={microbatch}", flush=True)
        (result.loss / 8).backward(); sync(torch)
        print(f"TRAIN_DIAG: backward_complete microbatch={microbatch}", flush=True)
        rows.append({"microbatch": microbatch, "incident_id": example.example_id, "sequence_length": len(example.input_ids), "loss": float(result.loss.detach().cpu()), "memory": memory(torch)})
        del result, batch
    finite, nonzero, norm = grad_report(model, torch)
    output = {"status": "pass" if finite and nonzero else "fail", "microbatches": rows, "gradient_finite": finite, "gradient_nonzero": nonzero, "accumulated_lora_grad_norm": norm, "optimizer_step_called": False, "memory_after_accumulation": memory(torch)}
    discard_gradients(model, torch); del examples, model; gc.collect(); torch.cuda.empty_cache(); sync(torch)
    return output


def stage_validation() -> dict[str, Any]:
    import torch
    torch.cuda.init(); model, tokenizer, _, torch = model_setup()
    from causetune.incident_evaluation import evaluate_incidents
    import importlib.util
    spec = importlib.util.spec_from_file_location("incident_runner", ROOT / "scripts" / "run_incident_qlora.py")
    runner = importlib.util.module_from_spec(spec); assert spec.loader is not None; spec.loader.exec_module(runner)
    records = read_jsonl(DATA / "validation.jsonl"); truths = read_jsonl(DATA / "ground_truth_validation.jsonl")
    config = json.loads((ROOT / "configs" / "incident_diagnosis_eval.json").read_text())
    examples = runner.preprocess_incident_records(records, truths, tokenizer, 768, config["evaluation_contract"]["system_instruction"])
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=2e-4)
    timeline = {"model_loaded": memory(torch), "optimizer_created": {**memory(torch), "optimizer_state_entries": len(optimizer.state)}, "gradient_checkpointing_after_load": bool(getattr(model, "is_gradient_checkpointing", False)), "use_cache_after_load": bool(model.config.use_cache), "model_mode_before_validation": bool(model.training)}
    validation_start = time.perf_counter()
    raw_metrics, raw = runner._eval_records(model, tokenizer, records, truths, examples, config["evaluation_contract"]["system_instruction"], config)
    sync(torch); timeline["after_generation_validation"] = {**memory(torch), "raw_output_count": len(raw), "raw_outputs_are_strings": all(isinstance(value, str) for value in raw.values())}
    aggregation = evaluate_incidents([{"incident_id": item["incident_id"], "slice": item["metadata"]["difficulty"], "incident_packet": item["incident_packet"], "metadata": item["metadata"]} for item in records], {item["incident_id"]: item for item in truths}, raw)
    del raw_metrics, raw, aggregation
    gc.collect(); torch.cuda.empty_cache(); sync(torch)
    timeline["after_generation_cleanup"] = memory(torch)
    teacher = runner._teacher_loss(model, tokenizer, examples)
    sync(torch); timeline["after_teacher_loss"] = {**memory(torch), "teacher_loss": float(teacher["loss"]), "teacher_tokens": int(teacher["tokens"])}
    mode_after_validation = bool(model.training)
    del teacher
    gc.collect(); torch.cuda.empty_cache(); sync(torch)
    timeline["after_all_aggregation"] = {**memory(torch), "model_mode_after_validation": mode_after_validation, "optimizer_state_entries": len(optimizer.state)}
    model.train(); model.config.use_cache = False; optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache(); sync(torch)
    timeline["after_cleanup_before_backward"] = {**memory(torch), "model_training_mode": bool(model.training), "use_cache": bool(model.config.use_cache)}
    record, batch = one_example(tokenizer, json.loads((DATA / "resolved_training_manifest.json").read_text()))
    batch = batch_on_device(batch, next(model.parameters()).device); sync(torch)
    print("TRAIN_DIAG: post_validation input_ready", flush=True)
    forward_start = time.perf_counter(); result = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"]); sync(torch); forward = time.perf_counter() - forward_start
    print("TRAIN_DIAG: post_validation forward_complete", flush=True)
    backward_start = time.perf_counter(); result.loss.backward(); sync(torch); backward = time.perf_counter() - backward_start
    print("TRAIN_DIAG: post_validation backward_complete", flush=True)
    finite, nonzero, norm = grad_report(model, torch)
    timeline["post_validation_backward"] = {"status": "pass" if finite and nonzero else "fail", "forward_seconds": forward, "backward_seconds": backward, "gradient_finite": finite, "gradient_nonzero": nonzero, "lora_grad_norm": norm, "memory": memory(torch), "incident_id": record["incident_id"]}
    timeline["validation_duration_seconds"] = time.perf_counter() - validation_start
    discard_gradients(model, torch); del result, batch, examples, optimizer, model; gc.collect(); torch.cuda.empty_cache(); sync(torch)
    # _eval_records and _teacher_loss intentionally leave eval mode; the
    # training loop restores train mode immediately after _validation.  The
    # post-cleanup backward is the execution proof that this lifecycle works.
    return {"status": "pass" if timeline["post_validation_backward"]["status"] == "pass" else "fail", "timeline": timeline, "validation_count": len(records), "optimizer_step_called": False, "mode_restored_before_post_backward": timeline["after_cleanup_before_backward"]["model_training_mode"]}


def child_main(stage: str) -> None:
    try:
        import torch
        result = {"stage": stage, "result": globals()[f"stage_{stage}"]()}
    except Exception as exc:
        result = {"stage": stage, "result": {"status": "fail", "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(), "optimizer_step_called": False}}
    print("__AUDIT_JSON__" + json.dumps(result, sort_keys=True), flush=True)


def environment() -> dict[str, Any]:
    values: dict[str, Any] = {"python": sys.version, "cwd": str(ROOT), "env": {key: os.environ.get(key) for key in ("CUDA_HOME", "LD_LIBRARY_PATH", "BNB_CUDA_VERSION", "PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF", "TORCH_LOGS")}}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "triton"):
        try: values[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: values[package] = None
    for command, name in ((["nvidia-smi", "--query-gpu=driver_version,name,temperature.gpu,clocks.sm,utilization.gpu,memory.used,memory.total", "--format=csv,noheader"], "nvidia_smi"), (["uname", "-a"], "uname"), (["cat", "/proc/version"], "proc_version")):
        try: values[name] = subprocess.run(command, capture_output=True, text=True, check=False, timeout=15).stdout.strip()
        except Exception as exc: values[name] = f"error: {exc}"
    return values


def run_parent() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--stage"); args = parser.parse_args()
    if args.stage:
        child_main(args.stage); return
    stages = ["basic", "load", "forward", "backward", "accumulation", "validation"]
    report: dict[str, Any] = {"audit": "incident_diagnosis_02b2_cuda_execution", "environment_before": environment(), "allocator_policy": {"unset_for_diagnostics": True, "original_pytorch_alloc_conf": os.environ.get("PYTORCH_ALLOC_CONF"), "original_pytorch_cuda_alloc_conf": os.environ.get("PYTORCH_CUDA_ALLOC_CONF")}, "stages": {}, "identified_root_cause": None, "fixes_applied": ["runner restores model training/eval state after validation", "inference temporarily enables KV cache and restores training use_cache state", "diagnostic processes unset expandable-segment allocator settings"], "real_training": {"optimizer_steps": 0, "started": False, "adapter_created": False, "benchmark_evaluated": False}}
    child_env = dict(os.environ)
    for key in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        child_env.pop(key, None)
    child_env.update({"CUDA_LAUNCH_BLOCKING": "1", "TORCH_SHOW_CPP_STACKTRACES": "1", "PYTHONPATH": str(ROOT / "src") + os.pathsep + child_env.get("PYTHONPATH", "")})
    for stage in stages:
        if stage == "validation":
            child_env.pop("CUDA_LAUNCH_BLOCKING", None)
        command = [sys.executable, str(Path(__file__).resolve()), "--stage", stage]
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=child_env, capture_output=True, text=True, check=False, timeout=3600)
        parsed = None
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith("__AUDIT_JSON__"):
                parsed = json.loads(line[len("__AUDIT_JSON__"):]); break
        if parsed is None:
            parsed = {"stage": stage, "result": {"status": "fail", "error": "child produced no audit result", "returncode": completed.returncode}}
        parsed["duration_seconds"] = time.perf_counter() - started
        parsed["returncode"] = completed.returncode
        parsed["stdout_tail"] = completed.stdout[-4000:]
        parsed["stderr_tail"] = completed.stderr[-4000:]
        report["stages"][stage] = parsed
        if completed.returncode != 0 or parsed["result"].get("status") != "pass":
            report["identified_root_cause"] = f"{stage} stage failed"
            break
    if report["identified_root_cause"] is None:
        report["identified_root_cause"] = "no failure reproduced in diagnostic ladder; prior stall was not reproduced by no-step smoke tests"
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cuda_execution_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "audit_path": str(OUTPUT / "cuda_execution_audit.json"), "stages": {key: value["result"].get("status") for key, value in report["stages"].items()}, "root_cause": report["identified_root_cause"], "optimizer_steps": 0}, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_parent()
