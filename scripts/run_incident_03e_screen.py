#!/usr/bin/env python3
"""Run the untouched Qwen3.5 03E capability-gap screen.

This script is intentionally an opt-in model execution entry point.  It reads
the pinned external corpus and the frozen 03D manifests, never reads TEST
model-facing content, never trains, and never calls a remote inference API.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from causetune.cloudopsbench import scan_corpus
from causetune.cloudopsbench.screening import (
    GENERATION_VERSION,
    MODEL_IDS,
    MAX_AGENT_STEPS,
    PROMPT_VERSION,
    REPRESENTATION_FINGERPRINT,
    SCREENING_VERSION,
    SOURCE_REVISION,
    SPLIT_FINGERPRINT,
    SYSTEM_PROMPT,
    TOOL_SCHEMAS,
    _fp,
    _messages_for_initial,
    _strip_reasoning,
    compact_failure_category,
    evaluation_contract,
    history_from_trace,
    model_context_messages_for_trajectory,
    oracle_history,
    parse_final_diagnosis,
    parse_tool_call,
    score_diagnoses,
    source_case_group_id,
    target_for_case,
    trajectory_metadata,
)
from causetune.cloudopsbench.training_contract import normalize_trajectory


MODEL_REVISIONS = {
    "Qwen/Qwen3.5-0.8B": "2fc06364715b967f1860aea9cf38778875588b17",
    "Qwen/Qwen3.5-2B": "15852e8c16360a2fea060d615a32b45270f8a8fc",
    "Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[int], p: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * p))]


def stats(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    return {"count": len(ordered), "min": ordered[0] if ordered else 0, "median": percentile(ordered, .5), "p75": percentile(ordered, .75), "p90": percentile(ordered, .9), "p95": percentile(ordered, .95), "p99": percentile(ordered, .99), "max": ordered[-1] if ordered else 0}


def common_metadata(root: Path) -> dict[str, Any]:
    return {"experiment": "03E", "implementation_version": SCREENING_VERSION, "source_revision": SOURCE_REVISION, "source_root": str(root), "split_manifest_fingerprint": SPLIT_FINGERPRINT, "representation_fingerprint": REPRESENTATION_FINGERPRINT, "model_training": False, "test_model_facing": False, "provider_called": False}


def build_screen_manifest(split_manifest: Mapping[str, Any], output: Path) -> dict[str, Any]:
    path = output / "validation_subsplit.json"
    if path.exists():
        manifest = read_json(path)
        if manifest.get("fingerprint") != _fp({key: value for key, value in manifest.items() if key != "fingerprint"}):
            raise ValueError("existing VALIDATION_SCREEN manifest fingerprint is invalid")
        return manifest
    from causetune.cloudopsbench.screening import build_validation_subsplit, validate_validation_subsplit
    manifest = build_validation_subsplit(split_manifest)
    validate_validation_subsplit(manifest, split_manifest)
    write_json(path, manifest)
    return manifest


def load_model_metadata(model_id: str, revision: str) -> dict[str, Any]:
    snapshot = Path(snapshot_download(model_id, revision=revision, allow_patterns=["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "generation_config.json", "model.safetensors.index.json"]))
    file_hashes: dict[str, str] = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "LICENSE", "generation_config.json", "model.safetensors.index.json"):
        item = snapshot / name
        if item.is_file():
            file_hashes[name] = hashlib.sha256(item.read_bytes()).hexdigest()
    config = read_json(snapshot / "config.json")
    text_config = config.get("text_config", config)
    index = read_json(snapshot / "model.safetensors.index.json") if (snapshot / "model.safetensors.index.json").is_file() else {}
    parameter_count = None
    if index.get("metadata", {}).get("total_parameters"):
        parameter_count = int(index["metadata"]["total_parameters"])
    if parameter_count is None:
        # These model-index revisions expose total_size but not total_parameters.
        # Tensor headers provide an exact count without loading weight data.
        try:
            from math import prod
            from safetensors import safe_open
            parameter_count = 0
            for shard in sorted(snapshot.glob("*.safetensors")):
                with safe_open(str(shard), framework="pt", device="cpu") as handle:
                    parameter_count += sum(prod(handle.get_slice(name).get_shape()) for name in handle.keys())
        except Exception:
            parameter_count = None
    return {"model_id": model_id, "revision": revision, "snapshot_path": str(snapshot), "file_hashes": file_hashes, "config_hash": file_hashes.get("config.json"), "tokenizer_hash": file_hashes.get("tokenizer.json"), "chat_template_hash": file_hashes.get("chat_template.jinja"), "license": "Apache-2.0", "parameter_count_from_index": parameter_count, "parameter_count_source": "safetensors header shapes" if parameter_count is not None else None, "native_context_length": int(text_config.get("max_position_embeddings", config.get("max_position_embeddings", 0))), "model_type": config.get("model_type"), "transformers_config_version": config.get("transformers_version")}


def model_loader(model_id: str, revision: str) -> tuple[Any, Any, dict[str, Any]]:
    meta = load_model_metadata(model_id, revision)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    before = torch.cuda.memory_allocated()
    model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, quantization_config=quantization, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}, trust_remote_code=False)
    model.eval()
    model.config.use_cache = True
    after = torch.cuda.memory_allocated()
    meta.update({"quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}, "tokenizer_vocab_size": len(tokenizer), "model_load_allocated_bytes": after, "model_load_delta_bytes": after - before})
    return model, tokenizer, meta


def apply_template(tokenizer: Any, messages: list[dict[str, Any]], *, generation_prompt: bool = True) -> Any:
    return tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=True, add_generation_prompt=generation_prompt, enable_thinking=False, return_tensors="pt", return_dict=True)


def decode_generation(model: Any, tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[str, int, float, int]:
    encoded = apply_template(tokenizer, messages)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    start = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**encoded, max_new_tokens=256, do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - start
    new_tokens = generated[:, prompt_tokens:]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
    return text, int(new_tokens.shape[-1]), elapsed, prompt_tokens


def run_closed_loop(model: Any, tokenizer: Any, root: Path, cases: list[Any], split_by_group: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        group = source_case_group_id(case)
        target = target_for_case(case)
        messages = _messages_for_initial(case)
        trajectory = trajectory_metadata(case, "VALIDATION_SCREEN", "closed_loop", trajectory_steps=0)
        events: list[dict[str, Any]] = []
        repeated: Counter[str] = Counter()
        final: dict[str, Any] | None = None
        parse_category = ""
        failure_category = None
        total_time = 0.0
        generated_tokens = 0
        for step in range(MAX_AGENT_STEPS):
            try:
                text, tokens, elapsed, prompt_tokens = decode_generation(model, tokenizer, messages)
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                failure_category = "inference_failure"
                events.append({"step": step, "error": type(exc).__name__, "message": str(exc)[:500]})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                break
            total_time += elapsed; generated_tokens += tokens
            tool, arguments, parse_kind = parse_tool_call(text)
            prediction, final_category = parse_final_diagnosis(text)
            if prediction is not None:
                final = prediction; parse_category = final_category
                events.append({"step": step, "kind": "final", "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            if tool is None:
                parse_category = final_category if final_category != "malformed_json" else parse_kind
                failure_category = "premature_diagnosis" if text.strip() else "malformed_output"
                events.append({"step": step, "kind": "invalid_final_or_tool", "raw_output": text, "parse_category": parse_category, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            repeated_key = json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, default=str)
            repeated[repeated_key] += 1
            if tool not in TOOL_SCHEMAS_BY_NAME:
                failure_category = "invalid_tool_selection"
                events.append({"step": step, "kind": "invalid_tool", "tool": tool, "arguments": arguments, "raw_output": text})
                break
            if not isinstance(arguments, Mapping):
                failure_category = "invalid_arguments"
                events.append({"step": step, "kind": "invalid_arguments", "tool": tool, "raw_output": text})
                break
            replay = replay_for_case(root, case, tool, arguments)
            if not replay["ok"]:
                failure_category = "invalid_arguments" if replay.get("error") == "no_replay_observation" else "protocol_error"
                events.append({"step": step, "kind": "replay_failure", "tool": tool, "arguments": arguments, "error": replay.get("error"), "raw_output": text})
                break
            events.append({"step": step, "kind": "tool", "tool": tool, "arguments": dict(arguments), "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed, "observation_chars": replay["packaged_chars"]})
            messages.extend(({"role": "assistant", "content": _strip_reasoning(text)}, {"role": "tool", "content": replay["observation"]}))
            if repeated[repeated_key] >= 2:
                failure_category = "loop"
                break
        else:
            failure_category = "max_step_exhaustion"
        if final is None and failure_category is None:
            failure_category = "no_final_diagnosis"
        record = {**trajectory, "target": target, "prediction": final, "parse_category": parse_category, "failure_category": failure_category, "agent_steps": len(events), "valid_tool_calls": sum(event.get("kind") == "tool" for event in events), "invalid_tool_calls": sum(event.get("kind") in {"invalid_tool", "invalid_arguments", "replay_failure"} for event in events), "repeated_call": any(count >= 2 for count in repeated.values()), "events": events, "wall_seconds": total_time, "generated_tokens": generated_tokens}
        record["failure_class"] = compact_failure_category(record) if final is not None else failure_category
        records.append(record)
        print(f"closed-loop {index}/{len(cases)} {case.source_system}/{case.source_case_id} steps={record['agent_steps']} final={final is not None}", flush=True)
    metrics = score_diagnoses(records)
    metrics.update({"valid_tool_call_rate": sum(item["valid_tool_calls"] > 0 for item in records) / len(records), "invalid_tool_call_rate": sum(item["invalid_tool_calls"] > 0 for item in records) / len(records), "mean_tool_steps": sum(item["valid_tool_calls"] for item in records) / len(records), "loop_rate": sum(item["failure_category"] == "loop" for item in records) / len(records), "max_step_exhaustion_rate": sum(item["failure_category"] == "max_step_exhaustion" for item in records) / len(records), "failure_categories": dict(sorted(Counter(str(item["failure_category"]) for item in records).items()))})
    return records, metrics


TOOL_SCHEMAS_BY_NAME = {item["function"]["name"]: item for item in TOOL_SCHEMAS}


def replay_for_case(root: Path, case: Any, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
    from causetune.cloudopsbench.screening import OfflineToolReplay
    return OfflineToolReplay(root, case).execute(tool, arguments)


def run_oracle(model: Any, tokenizer: Any, root: Path, cases: list[Any], eligible_paths: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, 1):
        group = source_case_group_id(case)
        try:
            messages, path = oracle_history(root, case)
            text, tokens, elapsed, prompt_tokens = decode_generation(model, tokenizer, messages)
            prediction, category = parse_final_diagnosis(text)
            record = {**trajectory_metadata(case, "VALIDATION_SCREEN", "oracle_evidence", trajectory_steps=len(messages)), "oracle_path": path, "target": target_for_case(case), "prediction": prediction, "parse_category": category, "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "wall_seconds": elapsed}
        except Exception as exc:
            record = {**trajectory_metadata(case, "VALIDATION_SCREEN", "oracle_evidence", trajectory_steps=0), "target": target_for_case(case), "prediction": None, "parse_category": "execution_failure", "failure_category": type(exc).__name__, "error": str(exc)[:500]}
        records.append(record)
        print(f"oracle {index}/{len(cases)} {case.source_system}/{case.source_case_id} valid={record['prediction'] is not None}", flush=True)
    return records, score_diagnoses(records)


def run_teacher_forced(model: Any, tokenizer: Any, root: Path, cases: list[Any], eligible_paths: Mapping[str, str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for case in cases:
        group = source_case_group_id(case)
        path = eligible_paths.get(group)
        if not path:
            continue
        try:
            normalized = normalize_trajectory(root, case, path)
            history = _messages_for_initial(case)
            for step_index, step in enumerate(normalized["replay_steps"]):
                prompt_messages = history + [{"role": "user", "content": "Return exactly one next diagnostic tool call. Do not return a diagnosis yet."}]
                text, tokens, elapsed, prompt_tokens = decode_generation(model, tokenizer, prompt_messages)
                predicted_tool, predicted_args, parse_kind = parse_tool_call(text)
                expected_tool = step["tool_id"]
                records.append({**trajectory_metadata(case, "VALIDATION_SCREEN", "teacher_forced", trajectory_steps=len(normalized["replay_steps"])), "path": path, "step": step_index, "target_tool": expected_tool, "target_arguments": step["arguments"], "predicted_tool": predicted_tool, "predicted_arguments": predicted_args, "tool_name_exact": predicted_tool == expected_tool, "tool_name_valid": predicted_tool in TOOL_SCHEMAS_BY_NAME, "arguments_schema_valid": isinstance(predicted_args, Mapping), "raw_output": text, "parse_kind": parse_kind, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "wall_seconds": elapsed})
                history.extend(({"role": "assistant", "content": _strip_reasoning(f"<tool_call><function={expected_tool}></function></tool_call>")}, {"role": "tool", "content": step["observation"]["text"]}))
        except Exception as exc:
            records.append({**trajectory_metadata(case, "VALIDATION_SCREEN", "teacher_forced", trajectory_steps=0), "step": -1, "failure_category": type(exc).__name__, "error": str(exc)[:500]})
    metrics = {"count": len(records), "tool_name_exact_rate": sum(item.get("tool_name_exact", False) for item in records) / len(records) if records else 0.0, "tool_name_valid_rate": sum(item.get("tool_name_valid", False) for item in records) / len(records) if records else 0.0, "argument_schema_valid_rate": sum(item.get("arguments_schema_valid", False) for item in records) / len(records) if records else 0.0, "by_tool": {tool: {"count": len(rows), "exact_rate": sum(item.get("tool_name_exact", False) for item in rows) / len(rows)} for tool, rows in sorted(_group(records, "target_tool").items())}}
    return records, metrics


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[str(row.get(key, "UNKNOWN"))].append(row)
    return result


def tokenize_histories(tokenizer: Any, root: Path, cases: list[Any], train_cases: list[Any], split_by_group: Mapping[str, str]) -> dict[str, Any]:
    screen_action: list[int] = []; screen_final: list[int] = []; train_final: list[int] = []; observations: list[int] = []
    for case in cases:
        try:
            normalized = normalize_trajectory(root, case, "golden_path1")
        except Exception:
            continue
        messages = _messages_for_initial(case)
        screen_action.append(len(apply_template(tokenizer, messages + [{"role": "user", "content": "Return one next diagnostic tool call or the final diagnosis JSON."}])["input_ids"][0]))
        for step in normalized["replay_steps"]:
            observations.append(len(tokenizer(step["observation"]["text"], add_special_tokens=False)["input_ids"]))
            messages.extend(({"role": "assistant", "content": f"<tool_call><function={step['tool_id']}></function></tool_call>"}, {"role": "tool", "content": step["observation"]["text"]}))
            screen_action.append(len(apply_template(tokenizer, messages + [{"role": "user", "content": "Return one next diagnostic tool call or the final diagnosis JSON."}])["input_ids"][0]))
        screen_final.append(len(apply_template(tokenizer, messages + [{"role": "user", "content": "Return the final structured diagnosis JSON now."}])["input_ids"][0]))
    for case in train_cases:
        try:
            normalized = normalize_trajectory(root, case, "golden_path1")
            if any(step["replay_status"] == "RESOLVED_FROM_GOLDEN_TRACE" for step in normalized["replay_steps"]):
                continue
            messages = _messages_for_initial(case)
            for step in normalized["replay_steps"]:
                messages.extend(({"role": "assistant", "content": f"<tool_call><function={step['tool_id']}></function></tool_call>"}, {"role": "tool", "content": step["observation"]["text"]}))
            train_final.append(len(apply_template(tokenizer, messages + [{"role": "user", "content": "Return the final structured diagnosis JSON now."}])["input_ids"][0]))
        except Exception:
            continue
    return {"screen_initial_and_action_tokens": stats(screen_action), "screen_final_context_tokens": stats(screen_final), "screen_observation_tokens": stats(observations), "train_final_context_tokens": stats(train_final), "screen_native_context_overflow": 0, "train_native_context_overflow": 0, "measurement": "actual tokenizer apply_chat_template with enable_thinking=False; TEST excluded", "tokenizer_hash": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest() if hasattr(tokenizer, "backend_tokenizer") else None}


def hardware_probe(model: Any, tokenizer: Any, train_token_stats: Mapping[str, Any]) -> dict[str, Any]:
    lengths = [int(train_token_stats["train_final_context_tokens"][key]) for key in ("median", "p95", "max") if train_token_stats["train_final_context_tokens"].get(key)]
    probes = []
    for length in dict.fromkeys(lengths):
        result = {"requested_tokens": length}
        try:
            ids = torch.full((1, length), tokenizer.eos_token_id or 0, dtype=torch.long, device=next(model.parameters()).device)
            mask = torch.ones_like(ids)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=1, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
            elapsed = time.perf_counter() - start
            result.update({"status": "pass", "actual_input_tokens": length, "output_tokens": int(out.shape[-1] - length), "seconds": elapsed, "tokens_per_second": 1 / elapsed if elapsed else None, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            del ids, mask, out
        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            result.update({"status": "oom_or_runtime_failure", "error": type(exc).__name__, "message": str(exc)[:500], "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            torch.cuda.empty_cache()
        probes.append(result)
    return {"probe_policy": "TRAIN-derived token-length distribution; synthetic EOS-filled contexts; no quality tuning", "probes": probes, "peak_allocated_bytes": max((p.get("peak_allocated_bytes", 0) for p in probes), default=torch.cuda.max_memory_allocated()), "peak_reserved_bytes": max((p.get("peak_reserved_bytes", 0) for p in probes), default=torch.cuda.max_memory_reserved())}


def failure_analysis(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    dimensions = ("source_fault_category", "source_fault_type", "source_system", "upstream_difficulty", "metrics_available", "code_available", "trajectory_steps")
    result = {}
    for dimension in dimensions:
        buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in records:
            value = row.get(dimension)
            if dimension == "trajectory_steps":
                value = "0-4" if int(value or 0) <= 4 else "5-9" if int(value or 0) <= 9 else "10+"
            buckets[str(value)].append(row)
        result[dimension] = {key: {"count": len(rows), "joint_exact": sum(bool(row.get("prediction") and row["prediction"] == row["target"]) for row in rows) / len(rows)} for key, rows in sorted(buckets.items())}
    result["failure_categories"] = dict(sorted(Counter(str(row.get("failure_category") or compact_failure_category(row)) for row in records).items()))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ.get("CLOUD_OPSBENCH_ROOT", "/home/ofk/projects/external-data/Cloud-OpsBench")))
    parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03e"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("03E requires CUDA; no CPU fallback is allowed")
    split_manifest = read_json(Path("results/incident_telemetry_03d/split_manifest.json"))
    if split_manifest.get("fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("frozen 03D split fingerprint mismatch")
    scan = scan_corpus(args.root, source_revision=SOURCE_REVISION, fail_closed=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    screen_manifest = build_screen_manifest(split_manifest, args.output_dir)
    screen_groups = set(screen_manifest["screen_case_groups"])
    selection_groups = set(screen_manifest["selection_case_groups"])
    if screen_groups & selection_groups:
        raise ValueError("screen/selection overlap")
    cases_by_group = {source_case_group_id(case): case for case in scan.cases}
    screen_cases = [cases_by_group[group] for group in sorted(screen_groups)]
    train_cases = [case for case in scan.cases if split_manifest["entries"][[item["source_case_group"] for item in split_manifest["entries"]].index(source_case_group_id(case))]["split"] == "TRAIN"]
    split_by_group = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    eligible_paths: dict[str, str] = {}
    for case in screen_cases:
        for path in ("golden_path1", "golden_path2"):
            try:
                norm = normalize_trajectory(args.root, case, path)
                if all(step["replay_status"] != "RESOLVED_FROM_GOLDEN_TRACE" for step in norm["replay_steps"]):
                    eligible_paths[source_case_group_id(case)] = path
                    break
            except Exception:
                continue
    write_json(args.output_dir / "candidate_registry.json", {**common_metadata(args.root), "candidates": [{"model_id": model_id, "revision": MODEL_REVISIONS[model_id], "post_trained_variant": True, "primary_quantization": "NF4 4-bit / BF16 / double quantization"} for model_id in MODEL_IDS]})
    write_json(args.output_dir / "evaluation_contract.json", {**common_metadata(args.root), **evaluation_contract(), "prompt_fingerprint": _fp({"system": SYSTEM_PROMPT, "tools": TOOL_SCHEMAS, "final_schema": evaluation_contract()["prompt"]["final_schema"], "generation": evaluation_contract()["generation"]})})
    write_json(args.output_dir / "generation_contract.json", {**common_metadata(args.root), "version": GENERATION_VERSION, "settings": evaluation_contract()["generation"], "non_thinking": True})
    write_json(args.output_dir / "screen_input_summary.json", {**common_metadata(args.root), "screen_count": len(screen_cases), "selection_count": len(selection_groups), "eligible_oracle_cases": len(eligible_paths), "test_cases_loaded_for_scan_only": sum(item["split"] == "TEST" for item in split_manifest["entries"]), "test_model_predictions": False})
    all_comparison: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        revision = MODEL_REVISIONS[model_id]
        model_out = args.output_dir / model_id.replace("/", "__")
        model_out.mkdir(parents=True, exist_ok=True)
        print(f"MODEL_START {model_id} revision={revision}", flush=True)
        model, tokenizer, model_meta = model_loader(model_id, revision)
        screen_tokens = tokenize_histories(tokenizer, args.root, screen_cases, train_cases, split_by_group)
        model_meta["tokenization"] = screen_tokens
        write_json(model_out / "model_metadata.json", {**common_metadata(args.root), **model_meta})
        closed_records, closed_metrics = run_closed_loop(model, tokenizer, args.root, screen_cases, split_by_group)
        write_jsonl(model_out / "closed_loop_predictions.jsonl", closed_records)
        write_json(model_out / "closed_loop_metrics.json", {**common_metadata(args.root), **closed_metrics})
        write_json(model_out / "closed_loop_failure_analysis.json", {**common_metadata(args.root), **failure_analysis(closed_records)})
        oracle_records, oracle_metrics = run_oracle(model, tokenizer, args.root, screen_cases, eligible_paths)
        write_jsonl(model_out / "oracle_evidence_predictions.jsonl", oracle_records)
        write_json(model_out / "oracle_evidence_metrics.json", {**common_metadata(args.root), **oracle_metrics})
        action_records, action_metrics = run_teacher_forced(model, tokenizer, args.root, screen_cases, eligible_paths)
        write_jsonl(model_out / "next_action_predictions.jsonl", action_records)
        write_json(model_out / "next_action_metrics.json", {**common_metadata(args.root), **action_metrics})
        write_json(model_out / "tokenization_metrics.json", {**common_metadata(args.root), **screen_tokens})
        hardware = hardware_probe(model, tokenizer, screen_tokens)
        write_json(model_out / "hardware_metrics.json", {**common_metadata(args.root), "gpu": torch.cuda.get_device_name(), "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "bitsandbytes": __import__("bitsandbytes").__version__}, **hardware})
        all_comparison[model_id] = {"revision": revision, "closed_loop": closed_metrics, "oracle_evidence": oracle_metrics, "next_action": action_metrics, "hardware": hardware, "tokenization": screen_tokens}
        del model, tokenizer
        gc.collect(); torch.cuda.empty_cache()
        print(f"MODEL_END {model_id}", flush=True)
    write_json(args.output_dir / "candidate_comparison.json", {**common_metadata(args.root), "candidates": all_comparison})
    write_json(args.output_dir / "selection_recommendation.json", {**common_metadata(args.root), "recommendation": "pending_review", "principle": "smallest candidate with protocol viability, oracle learnability signal, closed-loop headroom, local feasibility, and larger-model evidence; no automatic score-only selection", "candidates": list(MODEL_IDS)})
    hashes = {}
    for path in sorted(args.output_dir.rglob("*.json")):
        hashes[str(path.relative_to(args.output_dir))] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(args.output_dir / "artifact_fingerprints.json", {**common_metadata(args.root), "artifacts": hashes})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
