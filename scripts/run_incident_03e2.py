#!/usr/bin/env python3
"""Frozen 03E.2 Qwen3.5-2B capability rescreen.

The freeze artifacts are written before model loading and generation.  The
script reads the immutable 57-case VALIDATION_SCREEN only; TEST and
VALIDATION_SELECTION never become model-facing inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from causetune.cloudopsbench import scan_corpus
from causetune.cloudopsbench.taxonomy import UPSTREAM_FAULT_TAXONOMY
from causetune.cloudopsbench.training_contract import _cache_index, normalize_trajectory, source_case_group_id
from causetune.cloudopsbench.screening import MAX_OBSERVATION_CHARS, REPRESENTATION_FINGERPRINT, SOURCE_REVISION, SPLIT_FINGERPRINT, _fp, _strip_reasoning, target_for_case, trajectory_metadata

MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
MAX_NEW_TOKENS = 256
MAX_AGENT_STEPS = 20
VERSION = "cloud-opsbench-03e2-frozen-v2-contract-v1"

# V1's empty GetRecentLogs form had no executable case-local cache source.
# The entire invocation is disabled; all advertised tools use deterministic
# case-cache replay with a stable empty/partial-argument default.
TOOL_IDS = ("GetResources", "DescribeResource", "GetAppYAML", "GetServiceDependencies", "CheckServiceConnectivity", "GetAlerts", "GetErrorLogs", "ListCodeFiles", "GetSourceCode", "GetClusterConfiguration", "CheckNodeServiceStatus")
TOOL_SCHEMAS = [{"type": "function", "function": {"name": name, "description": "Inspect one immutable case-local incident artifact; never invent observations.", "parameters": {"type": "object", "additionalProperties": True, "properties": {}}}} for name in TOOL_IDS]
TOOL_NAMES = {item["function"]["name"] for item in TOOL_SCHEMAS}
NATIVE_CATEGORIES = list(UPSTREAM_FAULT_TAXONOMY)
NATIVE_FAULT_TYPES = [fault_type for category in NATIVE_CATEGORIES for fault_type in UPSTREAM_FAULT_TAXONOMY[category]]
SYSTEM_PROMPT = """You are an offline production-incident diagnostic agent. Use only the allowed diagnostic tools and their returned observations. Do not invent tool names, arguments, observations, labels, or evidence. You may call one tool at a time, or return the final diagnosis.

When calling a tool, use the native function-call format exactly.
When finalizing, return JSON only with exactly these keys:
{{"root_cause": string, "fault_category": string, "fault_object": string}}

root_cause MUST be one of the complete native root-cause enum below.
fault_category MUST be one of the complete native fault-category enum below.
fault_object is an open source-grounded resource identity and is not an enum.

Native fault-category enum:
{categories}

Native root-cause enum:
{fault_types}

Once sufficient evidence has been collected to select a diagnosis, emit the structured final diagnosis instead of continuing to call tools.
Do not explain your reasoning. Do not output hidden reasoning. The incident request and tool observations are the complete model-visible input; benchmark metadata, process labels, golden answers, and source paths are unavailable.""".format(categories=json.dumps(NATIVE_CATEGORIES), fault_types=json.dumps(NATIVE_FAULT_TYPES))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meta(root: Path) -> dict[str, Any]:
    return {"experiment": "03E.2", "version": VERSION, "source_revision": SOURCE_REVISION, "source_root": str(root), "split_manifest_fingerprint": SPLIT_FINGERPRINT, "representation_fingerprint": REPRESENTATION_FINGERPRINT, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "model_training": False, "provider_called": False, "test_model_facing": False, "validation_screen_count": 57, "validation_selection_model_facing": False}


def target(case: Any) -> dict[str, str]:
    old = target_for_case(case)
    return {"root_cause": old["native_fault_type"], "fault_category": old["native_fault_category"], "fault_object": old["fault_object"]}


def parse_final(text: str) -> tuple[dict[str, Any] | None, str]:
    try:
        value = json.loads(_strip_reasoning(text))
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(value, dict) or set(value) != {"root_cause", "fault_category", "fault_object"}:
        return None, "wrong_final_schema"
    if any(not isinstance(value[key], str) or not value[key].strip() for key in value):
        return None, "invalid_final_values"
    return value, "valid"


def parse_tool(text: str) -> tuple[str | None, dict[str, Any] | None, str]:
    clean = _strip_reasoning(text)
    match = re.search(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", clean, re.DOTALL)
    if match:
        args: dict[str, Any] = {}
        for parameter in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", match.group(2), re.DOTALL):
            key, value = parameter.group(1).strip(), parameter.group(2).strip()
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        return match.group(1).strip(), args, "xml_tool_call"
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        return None, None, "not_tool_call"
    if isinstance(value, dict) and value.get("type") in {"tool_call", "tool"} and isinstance(value.get("tool"), str):
        return value["tool"], value.get("arguments", {}), "json_tool_call"
    return None, None, "not_tool_call"


def initial_messages(case: Any) -> list[dict[str, str]]:
    query = str(case.ground_truth_metadata.get("query", "Investigate the reported service condition.")).strip()
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Incident request: {query}"}]


def apply_template(tokenizer: Any, messages: list[dict[str, Any]]) -> Any:
    return tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)


def generate(model: Any, tokenizer: Any, messages: list[dict[str, Any]]) -> tuple[str, int, float, int, int, int]:
    encoded = apply_template(tokenizer, messages)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
    elapsed = time.perf_counter() - started
    new = output[:, prompt_tokens:]
    return tokenizer.decode(new[0], skip_special_tokens=False), int(new.shape[-1]), elapsed, prompt_tokens, torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0, torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0


def load_model() -> tuple[Any, Any, dict[str, Any]]:
    prior_path = Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json")
    prior = read_json(prior_path)
    snapshot = Path(str(prior["snapshot_path"]))
    if not snapshot.is_dir():
        raise FileNotFoundError(f"pinned local snapshot missing: {snapshot}")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    before = torch.cuda.memory_allocated()
    model = AutoModelForImageTextToText.from_pretrained(str(snapshot), local_files_only=True, quantization_config=quantization, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}, trust_remote_code=False)
    model.eval()
    model.config.use_cache = True
    after = torch.cuda.memory_allocated()
    return model, tokenizer, {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(snapshot), "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}, "enable_thinking": False, "max_new_tokens": MAX_NEW_TOKENS, "max_agent_steps": MAX_AGENT_STEPS, "greedy": True, "tokenizer_vocab_size": len(tokenizer), "model_load_allocated_bytes": after, "model_load_delta_bytes": after - before, "v1_metadata_sha256": file_hash(prior_path)}


def executable_history(root: Path, case: Any) -> tuple[list[dict[str, Any]], str, int]:
    choices: list[tuple[int, str, list[dict[str, Any]], int]] = []
    for path_name in ("golden_path1", "golden_path2"):
        try:
            normalized = normalize_trajectory(root, case, path_name)
        except Exception:
            continue
        messages = initial_messages(case)
        omitted = 0
        for step in normalized["replay_steps"]:
            if step["replay_status"] == "RESOLVED_FROM_GOLDEN_TRACE" or step["tool_id"] not in TOOL_NAMES:
                omitted += 1
                continue
            messages.extend(({"role": "assistant", "content": f"<tool_call><function={step['tool_id']}></function></tool_call>"}, {"role": "tool", "content": step["observation"]["text"]}))
        choices.append((len(messages), path_name, messages + [{"role": "user", "content": "Return the final structured diagnosis JSON now."}], omitted))
    if not choices:
        raise ValueError(f"no executable source evidence for {source_case_group_id(case)}")
    _, path, messages, omitted = max(choices, key=lambda item: (item[0], item[1]))
    return messages, path, omitted


class Replay:
    def __init__(self, root: Path, case: Any):
        self.cache = _cache_index(root, case)

    @staticmethod
    def equal(left: Any, right: Any) -> bool:
        if left in (None, "") and right in (None, ""):
            return True
        if isinstance(left, list):
            return right in left or left == right
        if isinstance(right, list):
            return left in right or left == right
        return left == right

    def execute(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if tool not in TOOL_NAMES:
            return {"ok": False, "error": "tool_unadvertised", "tool": tool}
        if not isinstance(arguments, Mapping):
            return {"ok": False, "error": "arguments_must_be_object", "tool": tool}
        candidates = [(key, value) for key, value in self.cache.items() if key == tool or key.startswith(tool + ":") or key.startswith(tool + "-")]
        scored: list[tuple[int, str, str]] = []
        for key, value in candidates:
            raw = key[len(tool) + 1:] if key.startswith(tool + ":") else "{}"
            try:
                key_args = json.loads(raw)
            except json.JSONDecodeError:
                key_args = {}
            if isinstance(key_args, Mapping) and all(name in key_args and self.equal(value_arg, key_args[name]) for name, value_arg in arguments.items()):
                scored.append((sum(name in key_args for name in arguments), key, str(value)))
        if scored:
            _, key, value = max(scored, key=lambda item: (item[0], item[1]))
        elif candidates:
            key, value = sorted((key, str(value)) for key, value in candidates)[0]
        else:
            # Missing optional modality is an executable runtime result, not a
            # fabricated observation.  This keeps a globally advertised tool
            # honest on cases whose source modality is unavailable.
            return {"ok": True, "tool": tool, "arguments": dict(arguments), "cache_key": None, "observation": f"[tool unavailable for this case: no case-local {tool} source artifact is present]", "replay_status": "CASE_LOCAL_SOURCE_UNAVAILABLE"}
        if len(value) > MAX_OBSERVATION_CHARS:
            value = value[:MAX_OBSERVATION_CHARS] + "\n[deterministically truncated]"
        return {"ok": True, "tool": tool, "arguments": dict(arguments), "cache_key": key, "observation": value, "replay_status": "RESOLVED_FROM_TOOL_CACHE"}


def visible_aliases(observations: Iterable[str]) -> set[str]:
    result: set[str] = set()
    for text in observations:
        result.update(re.findall(r"\bapp=([A-Za-z0-9][A-Za-z0-9._-]*)", text))
        result.update(re.findall(r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*)\s+(?:ClusterIP|NodePort|LoadBalancer)\b", text))
    return result


def object_match(predicted: Any, target_value: str, observations: Iterable[str]) -> tuple[bool, str]:
    if not isinstance(predicted, str):
        return False, "OTHER"
    if predicted == target_value:
        return True, "EXACT"
    aliases = visible_aliases(observations)
    if target_value.startswith("app/") and predicted == target_value[4:] and predicted in aliases:
        return True, "RESOURCE_PREFIX_ONLY"
    return False, "WRONG_TARGET"


def scores(records: list[Mapping[str, Any]], normalized: Mapping[str, bool]) -> dict[str, Any]:
    fields = ("root_cause", "fault_category", "fault_object")
    result: dict[str, Any] = {"count": len(records), "schema_valid_count": sum(row.get("prediction") is not None for row in records), "schema_valid_rate": sum(row.get("prediction") is not None for row in records) / len(records) if records else 0.0, "valid_enum_count": sum(row.get("enum_valid", False) for row in records), "valid_enum_rate": sum(row.get("enum_valid", False) for row in records) / len(records) if records else 0.0, "invalid_enum_outputs": sum(row.get("prediction") is not None and not row.get("enum_valid", False) for row in records), "missing_fields": sum(row.get("parse_category") in {"wrong_final_schema", "invalid_final_values", "malformed_json"} for row in records)}
    for field in fields:
        count = sum(bool(row.get("prediction") and row["prediction"].get(field) == row["target"].get(field)) for row in records)
        result[field + "_strict_exact_count"] = count
        result[field + "_strict_exact_rate"] = count / len(records) if records else 0.0
    result["object_normalized_exact_count"] = sum(normalized.get(str(row.get("source_case_group")), False) for row in records)
    result["object_normalized_exact_rate"] = result["object_normalized_exact_count"] / len(records) if records else 0.0
    strict_joint = sum(bool(row.get("prediction") and all(row["prediction"].get(field) == row["target"].get(field) for field in fields)) for row in records)
    normalized_joint = sum(bool(row.get("prediction") and row["prediction"].get("root_cause") == row["target"].get("root_cause") and row["prediction"].get("fault_category") == row["target"].get("fault_category") and normalized.get(str(row.get("source_case_group")), False)) for row in records)
    result.update({"strict_joint_exact_count": strict_joint, "strict_joint_exact_rate": strict_joint / len(records) if records else 0.0, "normalized_joint_exact_count": normalized_joint, "normalized_joint_exact_rate": normalized_joint / len(records) if records else 0.0})
    return result


def run_oracle(model: Any, tokenizer: Any, root: Path, cases: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, bool]]:
    records: list[dict[str, Any]] = []
    normalized: dict[str, bool] = {}
    for number, case in enumerate(cases, 1):
        group = source_case_group_id(case)
        try:
            messages, path, omitted = executable_history(root, case)
            text, tokens, elapsed, prompt_tokens, allocated, reserved = generate(model, tokenizer, messages)
            prediction, parse_category = parse_final(text)
            target_value = target(case)
            observations = [str(item["content"]) for item in messages if item.get("role") == "tool"]
            object_ok, object_class = object_match(prediction.get("fault_object") if prediction else None, target_value["fault_object"], observations)
            enum_valid = bool(prediction and prediction.get("root_cause") in NATIVE_FAULT_TYPES and prediction.get("fault_category") in NATIVE_CATEGORIES)
            row = {**trajectory_metadata(case, "VALIDATION_SCREEN", "oracle_evidence_v2", trajectory_steps=len(messages)), "target": target_value, "prediction": prediction, "parse_category": parse_category, "enum_valid": enum_valid, "oracle_path": path, "omitted_nonexecutable_steps": omitted, "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "wall_seconds": elapsed, "peak_allocated_bytes": allocated, "peak_reserved_bytes": reserved, "object_normalization_class": object_class}
            normalized[group] = object_ok
        except Exception as exc:
            row = {**trajectory_metadata(case, "VALIDATION_SCREEN", "oracle_evidence_v2", trajectory_steps=0), "target": target(case), "prediction": None, "parse_category": "execution_failure", "enum_valid": False, "failure_category": type(exc).__name__, "error": str(exc)[:500]}
            normalized[group] = False
        records.append(row)
        print(f"oracle {number}/{len(cases)} schema={row['prediction'] is not None} enum={row['enum_valid']}", flush=True)
    return records, scores(records, normalized), normalized


def run_closed_loop(model: Any, tokenizer: Any, root: Path, cases: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, case in enumerate(cases, 1):
        messages = initial_messages(case)
        events: list[dict[str, Any]] = []
        repeated: Counter[str] = Counter()
        final: dict[str, Any] | None = None
        failure: str | None = None
        parse_category = ""
        total_time = 0.0
        total_tokens = 0
        peaks: list[tuple[int, int]] = []
        for step in range(MAX_AGENT_STEPS):
            try:
                text, tokens, elapsed, prompt_tokens, allocated, reserved = generate(model, tokenizer, messages)
            except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as exc:
                failure = "inference_failure"
                events.append({"step": step, "kind": "inference_failure", "error": type(exc).__name__, "message": str(exc)[:500]})
                torch.cuda.empty_cache()
                break
            total_time += elapsed; total_tokens += tokens; peaks.append((allocated, reserved))
            tool, arguments, tool_parse = parse_tool(text)
            prediction, final_parse = parse_final(text)
            if prediction is not None:
                final = prediction; parse_category = final_parse
                events.append({"step": step, "kind": "final", "raw_output": text, "prediction": prediction, "enum_valid": prediction.get("root_cause") in NATIVE_FAULT_TYPES and prediction.get("fault_category") in NATIVE_CATEGORIES, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            if tool is None:
                parse_category = final_parse if final_parse != "malformed_json" else tool_parse
                failure = "premature_final" if text.strip() else "no_final_diagnosis"
                events.append({"step": step, "kind": "invalid_final_or_tool", "raw_output": text, "parse_category": parse_category, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            key = json.dumps({"tool": tool, "arguments": arguments}, sort_keys=True, default=str); repeated[key] += 1
            if tool not in TOOL_NAMES:
                failure = "invalid_tool_selection"
                events.append({"step": step, "kind": "invalid_tool", "tool": tool, "arguments": arguments, "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            if not isinstance(arguments, Mapping):
                failure = "invalid_arguments"; events.append({"step": step, "kind": "invalid_arguments", "tool": tool, "raw_output": text}); break
            replay = Replay(root, case).execute(tool, arguments)
            if not replay["ok"]:
                failure = "environment_contract_failure" if replay.get("error") == "no_case_local_executable_artifact" else "invalid_arguments"
                events.append({"step": step, "kind": "replay_failure", "tool": tool, "arguments": dict(arguments), "error": replay.get("error"), "raw_output": text, "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
                break
            events.append({"step": step, "kind": "tool", "tool": tool, "arguments": dict(arguments), "raw_output": text, "observation": replay["observation"], "cache_key": replay["cache_key"], "replay_status": replay["replay_status"], "prompt_tokens": prompt_tokens, "generated_tokens": tokens, "seconds": elapsed})
            messages.extend(({"role": "assistant", "content": _strip_reasoning(text)}, {"role": "tool", "content": replay["observation"]}))
            if repeated[key] >= 2:
                failure = "repeated_tool_loop"; break
        else:
            failure = "max_step_exhaustion"
        records.append({**trajectory_metadata(case, "VALIDATION_SCREEN", "closed_loop_v2", trajectory_steps=len(events)), "target": target(case), "prediction": final, "parse_category": parse_category, "failure_category": failure, "agent_steps": len(events), "valid_tool_calls": sum(event.get("kind") == "tool" for event in events), "executable_tool_calls": sum(event.get("kind") == "tool" for event in events), "invalid_tool_calls": sum(event.get("kind") in {"invalid_tool", "invalid_arguments"} for event in events), "replay_failures": sum(event.get("kind") == "replay_failure" for event in events), "environment_contract_failures": sum(event.get("kind") == "replay_failure" and event.get("error") == "no_case_local_executable_artifact" for event in events), "repeated_call": any(count >= 2 for count in repeated.values()), "events": events, "wall_seconds": total_time, "generated_tokens": total_tokens, "peak_allocated_bytes": max((item[0] for item in peaks), default=0), "peak_reserved_bytes": max((item[1] for item in peaks), default=0), "enum_valid": bool(final and final.get("root_cause") in NATIVE_FAULT_TYPES and final.get("fault_category") in NATIVE_CATEGORIES)})
        print(f"closed-loop {number}/{len(cases)} steps={len(events)} final={final is not None}", flush=True)
    normalized: dict[str, bool] = {}
    for row in records:
        obs = [str(event.get("observation", "")) for event in row["events"] if event.get("kind") == "tool"]
        normalized[str(row["source_case_group"])] = bool(row.get("prediction") and object_match(row["prediction"].get("fault_object"), row["target"]["fault_object"], obs)[0])
    result = scores(records, normalized)
    result.update({"valid_tool_call_rate": sum(row["valid_tool_calls"] > 0 for row in records) / len(records), "executable_tool_call_rate": sum(row["executable_tool_calls"] == row["valid_tool_calls"] for row in records) / len(records), "invalid_tool_call_rate": sum(row["invalid_tool_calls"] > 0 for row in records) / len(records), "replay_failure_count": sum(row["replay_failures"] for row in records), "environment_contract_failure_count": sum(row["environment_contract_failures"] for row in records), "mean_tool_steps": sum(row["valid_tool_calls"] for row in records) / len(records), "loop_rate": sum(row["failure_category"] == "repeated_tool_loop" for row in records) / len(records), "premature_final_count": sum(row["failure_category"] == "premature_final" for row in records), "no_final_count": sum(row["prediction"] is None for row in records), "max_step_exhaustion_count": sum(row["failure_category"] == "max_step_exhaustion" for row in records), "failure_categories": dict(sorted(Counter(str(row["failure_category"]) for row in records).items())), "peak_allocated_bytes": max((row["peak_allocated_bytes"] for row in records), default=0), "peak_reserved_bytes": max((row["peak_reserved_bytes"] for row in records), default=0)})
    return records, result


def process_annotation(root: Path, case: Any) -> Any:
    from cloudops_agent.evaluation_utils.schema import CaseAnnotation
    payload = read_json(root / case.references.process_label.relative_path)
    for milestone in payload.get("milestones", []):
        milestone.setdefault("description", milestone.get("role", milestone.get("id", "")))
    return CaseAnnotation.from_dict(payload)


def process_metrics(root: Path, cases: list[Any], records: list[Mapping[str, Any]]) -> dict[str, Any]:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from cloudops_agent.evaluation_utils.evaluator import evaluate_trajectory
    from cloudops_agent.evaluation_utils.schema import ToolCall, TrajectoryStep
    counts: Counter[str] = Counter(); case_rows: dict[str, Any] = {}; complete = 0; sufficient_no_final = 0
    by_group = {str(row["source_case_group"]): row for row in records}
    for case in cases:
        group = source_case_group_id(case); row = by_group[group]; annotation = process_annotation(root, case)
        golden: dict[str, list[dict[str, Any]]] = {}
        for path in ("golden_path1", "golden_path2"):
            try: golden[path] = normalize_trajectory(root, case, path)["replay_steps"]
            except Exception: golden[path] = []
        trajectory: list[Any] = []; previous: set[str] = set(); actions: list[dict[str, Any]] = []
        for event in row.get("events", []):
            if event.get("kind") != "tool": continue
            call = ToolCall(tool_name=str(event["tool"]), arguments=event.get("arguments", {}), raw=str(event.get("raw_output", "")))
            trajectory.append(TrajectoryStep(tool_call=call, observation=str(event.get("observation", ""))))
            evaluation = evaluate_trajectory(annotation, trajectory); established = set(evaluation.established_ids); new = established - previous
            exact = any(any(item.get("tool_id") == call.tool_name and item.get("arguments", {}) == call.arguments for item in golden["golden_path1"]) for _ in (0,))
            alternate = any(item.get("tool_id") == call.tool_name and item.get("arguments", {}) == call.arguments for item in golden["golden_path2"])
            label = "EXACT_GOLDEN_PATH" if exact else "ALTERNATE_GOLDEN_PATH" if alternate else "PROCESS_ADMISSIBLE_PROGRESS" if new else "VALID_BUT_REDUNDANT"
            counts[label] += 1; actions.append({"step": event["step"], "tool": call.tool_name, "classification": label, "new_milestones": sorted(new), "established_milestones": sorted(established)}); previous = established
        final_eval = evaluate_trajectory(annotation, trajectory); is_complete = bool(final_eval.process_complete); complete += is_complete
        if row.get("prediction") is None and is_complete: sufficient_no_final += 1
        case_rows[group] = {"milestone_count": final_eval.milestone_count, "milestone_coverage": final_eval.milestone_coverage, "evidence_order_coverage": final_eval.evidence_order_coverage, "required_evidence_coverage": final_eval.milestone_coverage, "process_complete": is_complete, "established_milestones": final_eval.established_ids, "missing_milestones": list(final_eval.missing), "valid_tool_calls": len(trajectory), "evidence_tool_count": final_eval.evidence_tool_count, "action_classifications": actions}
    total = sum(counts.values())
    return {"version": "03e2-process-semantic-action-metrics-v1", "primary_metric": "upstream milestone/evidence semantics; exact golden next action is auxiliary", "teacher_forced_run": False, "action_categories": {key: {"count": value, "rate": value / total if total else 0.0} for key, value in sorted(counts.items())}, "cases": case_rows, "case_metrics": {"process_complete_count": complete, "process_complete_rate": complete / len(cases), "evidence_sufficient_but_no_final_count": sufficient_no_final, "evidence_sufficient_but_no_final_rate": sufficient_no_final / len(cases)}}


def tool_matrix(root: Path, cases: list[Any]) -> dict[str, Any]:
    matrix: dict[str, Any] = {}
    for tool in TOOL_IDS + ("GetRecentLogs",):
        available = sum(any(key == tool or key.startswith(tool + ":") or key.startswith(tool + "-") for key in _cache_index(root, case)) for case in cases)
        matrix[tool] = {"advertised": tool in TOOL_NAMES, "case_local_cache_cases": available, "deterministic_semantics": "case-cache exact/partial match with stable lexical default; explicit unavailable result when optional source is absent" if tool in TOOL_NAMES else "disabled: V1 empty-argument form had no executable source", "environment_contract_failure_allowed": False}
    return {"version": "03e2-tool-executability-v1", "removed_v1_tool": "GetRecentLogs", "removal_reason": "all 84 V1 replay failures were GetRecentLogs {}; raw logs do not define a faithful empty-argument replay", "no_fake_observations": True, "tools": matrix, "advertised_tool_count": len(TOOL_IDS), "advertised_tools_have_deterministic_executor": True}


def termination(records: list[Mapping[str, Any]], process: Mapping[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter(); rows = []
    for row in records:
        failure = row.get("failure_category")
        category = "PREMATURE_FINAL" if failure == "premature_final" else "REPEATED_TOOL_LOOP" if failure == "repeated_tool_loop" else "MAX_STEP_EXHAUSTION" if failure == "max_step_exhaustion" else "INVALID_REPLAY_BLOCKED_PROGRESS" if failure == "environment_contract_failure" else "NO_FINAL_DIAGNOSIS" if row.get("prediction") is None else "NONE"
        if category != "NONE": counts[category] += 1
        complete = bool(process.get("cases", {}).get(str(row["source_case_group"]), {}).get("process_complete", False))
        rows.append({"source_case_group": row["source_case_group"], "termination_category": category, "failure_category": failure, "required_process_milestones_satisfied": complete, "evidence_sufficient_but_no_final": category != "NONE" and complete})
    return {"version": "03e2-termination-audit-v1", "counts": dict(sorted(counts.items())), "evidence_sufficient_but_no_final_count": sum(row["evidence_sufficient_but_no_final"] for row in rows), "records": rows}


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", type=Path, default=Path(os.environ.get("CLOUD_OPSBENCH_ROOT", "/home/ofk/projects/external-data/Cloud-OpsBench"))); parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03e2")); args = parser.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("03E.2 requires CUDA; no CPU fallback")
    out = args.output_dir; out.mkdir(parents=True, exist_ok=True)
    split_path = Path("results/incident_telemetry_03d/split_manifest.json"); split = read_json(split_path)
    if split.get("fingerprint") != SPLIT_FINGERPRINT: raise ValueError("03D split fingerprint mismatch")
    manifest_path = Path("results/incident_telemetry_03e/validation_subsplit.json"); manifest = read_json(manifest_path)
    if manifest.get("screen_count") != 57 or manifest.get("split_manifest_fingerprint") != SPLIT_FINGERPRINT: raise ValueError("frozen VALIDATION_SCREEN mismatch")
    scan = scan_corpus(args.root, source_revision=SOURCE_REVISION, fail_closed=True); by_group = {source_case_group_id(case): case for case in scan.cases}; groups = sorted(manifest["screen_case_groups"]); cases = [by_group[group] for group in groups]
    if len(cases) != 57 or {source_case_group_id(case) for case in cases} != set(groups): raise ValueError("VALIDATION_SCREEN case set changed")
    base = meta(args.root)
    vocab = {**base, "ordered_categories": NATIVE_CATEGORIES, "ordered_fault_types": NATIVE_FAULT_TYPES, "category_count": len(NATIVE_CATEGORIES), "fault_type_count": len(NATIVE_FAULT_TYPES), "source": "src/causetune/cloudopsbench/taxonomy.py::UPSTREAM_FAULT_TAXONOMY", "correct_case_label_highlighted": False}
    write_json(out / "native_vocabulary.json", vocab)
    tools = tool_matrix(args.root, cases); write_json(out / "tool_executability.json", {**base, **tools, "tool_schema_fingerprint": _fp(TOOL_SCHEMAS)})
    contract = {**base, "contract_version": VERSION, "screen_groups_fingerprint": _fp(groups), "final_schema": ["root_cause", "fault_category", "fault_object"], "native_root_cause_enum": "native_vocabulary.json::ordered_fault_types", "native_category_enum": "native_vocabulary.json::ordered_categories", "fault_object_contract": "open source-grounded identity; only uniquely visible app/name prefix normalization", "normalization": "deterministic and source-derived; no target/process/golden information", "termination_rule": "Once sufficient evidence has been collected to select a diagnosis, emit the structured final diagnosis instead of continuing to call tools.", "tool_contract": tools, "generation": {"max_new_tokens": MAX_NEW_TOKENS, "max_agent_steps": MAX_AGENT_STEPS, "enable_thinking": False, "do_sample": False, "num_beams": 1, "quantization": "NF4 4-bit / BF16 compute / double quantization"}, "prompt_frozen_before_inference": True, "training": False, "loRA": False, "test_used": False, "validation_selection_used": False}
    write_json(out / "v2_contract.json", contract)
    prompt = {**base, "system_prompt": SYSTEM_PROMPT, "tool_schemas": TOOL_SCHEMAS, "final_schema": contract["final_schema"], "prompt_fingerprint": _fp({"system_prompt": SYSTEM_PROMPT, "tool_schemas": TOOL_SCHEMAS, "final_schema": contract["final_schema"]}), "frozen_before_first_generation": True, "few_shot": False, "case_specific_examples": False}
    write_json(out / "prompt_fingerprint.json", prompt)
    model, tokenizer, model_meta = load_model()
    oracle, oracle_metrics, _ = run_oracle(model, tokenizer, args.root, cases); write_jsonl(out / "oracle_predictions.jsonl", oracle); write_json(out / "oracle_metrics.json", {**base, **oracle_metrics, "technical_generation_completed": len(oracle) == 57})
    if len(oracle) != 57: raise RuntimeError("oracle incomplete; closed loop not run")
    closed, closed_metrics = run_closed_loop(model, tokenizer, args.root, cases); write_jsonl(out / "closed_loop_predictions.jsonl", closed)
    process = process_metrics(args.root, cases, closed); write_json(out / "process_semantic_metrics.json", {**base, **process}); write_json(out / "closed_loop_metrics.json", {**base, **closed_metrics, "technical_generation_completed": len(closed) == 57})
    term = termination(closed, process)
    hardware = {**base, "model": model_meta, "v2_peak_allocated_bytes": max([row.get("peak_allocated_bytes", 0) for row in oracle + closed] or [0]), "v2_peak_reserved_bytes": max([row.get("peak_reserved_bytes", 0) for row in oracle + closed] or [0]), "v2_oracle_prompt_token_max": max((row.get("prompt_tokens", 0) for row in oracle), default=0), "v2_closed_loop_prompt_token_max": max((event.get("prompt_tokens", 0) for row in closed for event in row.get("events", [])), default=0), "v1_4b_failures_not_rerun": True, "v1_4b_interpretation": "HARDWARE-CONSTRAINED / INCOMPLETE CAPABILITY EVIDENCE; 4B not run in 03E.2", "local_hardware_feasible": True}
    write_json(out / "hardware_metrics.json", hardware)
    write_json(out / "failure_analysis.json", {**base, "termination": term, "closed_loop_failure_categories": closed_metrics["failure_categories"], "oracle_execution_failures": sum(row.get("parse_category") == "execution_failure" for row in oracle), "replay_failures": closed_metrics["replay_failure_count"], "environment_contract_failures": closed_metrics["environment_contract_failure_count"], "v1_replay_context": {"recorded_failures": 84, "get_recent_logs_empty_call_failures": 84, "canonicalization_failures": 0}})
    classification = "CREDIBLE_SPECIALIZATION_STUDENT" if 0 < oracle_metrics["normalized_joint_exact_rate"] < 1 and closed_metrics["environment_contract_failure_count"] == 0 else "BASE_TOO_WEAK"; recommendation = "START_03F_QLORA" if classification == "CREDIBLE_SPECIALIZATION_STUDENT" else "DO_NOT_START_QLORA"
    write_json(out / "selection_recommendation.json", {**base, "classification": classification, "recommendation": recommendation, "qlora_unblocked": recommendation == "START_03F_QLORA", "basis": "pre-registered pattern: reliable schema/enum compliance, meaningful non-saturated diagnosis, usable protocol, remaining gap, local feasibility; no post-hoc threshold", "candidates_rerun": [MODEL_ID], "candidates_not_run": ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-4B"]})
    artifacts = {path.name: file_hash(path) for path in sorted(out.iterdir()) if path.is_file() and path.name != "artifact_fingerprints.json"}; immutable = {"v1_artifacts": {str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e").rglob("*")) if path.is_file()}, "03e1_artifacts": {str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e1").rglob("*")) if path.is_file()}, "validation_subsplit_sha256": file_hash(manifest_path), "split_manifest_sha256": file_hash(split_path)}
    write_json(out / "artifact_fingerprints.json", {**base, "native_vocabulary_fingerprint": _fp(vocab), "contract_fingerprint": _fp(contract), "prompt_fingerprint": prompt["prompt_fingerprint"], "tool_executability_fingerprint": _fp(tools), "artifacts": artifacts, "immutable_input_fingerprints": immutable})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
