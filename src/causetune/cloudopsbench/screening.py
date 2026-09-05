"""Experiment 03E: model-independent screening contracts and evaluation helpers.

The module deliberately keeps the frozen 03D source split intact.  It exposes
only TRAIN/VALIDATION_SCREEN material to model-facing callers; TEST is rejected
at the boundary.  Model loading and generation live in the script so the
contracts remain CPU-testable.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causetune.incident_telemetry.fingerprints import canonical_json

from .models import CloudOpsBenchCase
from .representation_integrity import _uniform_line_package
from .training_contract import (
    MAX_OBSERVATION_CHARS,
    _argument_pattern,
    _cache_index,
    _calling_arguments,
    _normalize_text,
    _trace,
    _resolve_cache_key,
    normalize_trajectory,
    sha256_value,
    source_case_group_id,
)


SOURCE_REVISION = "03c415e5709297432282fbbfd499f1bca0f8c347"
SPLIT_FINGERPRINT = "0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76"
REPRESENTATION_FINGERPRINT = "f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf"
SCREENING_VERSION = "cloud-opsbench-03e-screening-v1"
SCREEN_MANIFEST_VERSION = "cloud-opsbench-03e-validation-screen-v1"
PROMPT_VERSION = "cloud-opsbench-03e-primary-prompt-v1"
GENERATION_VERSION = "cloud-opsbench-03e-deterministic-generation-v1"
MAX_AGENT_STEPS = 20
MODEL_IDS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B")

TOOL_IDS = (
    "GetResources", "DescribeResource", "GetAppYAML", "GetServiceDependencies",
    "CheckServiceConnectivity", "GetAlerts", "GetRecentLogs", "GetErrorLogs",
    "ListCodeFiles", "GetSourceCode", "GetClusterConfiguration", "CheckNodeServiceStatus",
)

TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": name, "description": "Inspect one immutable incident artifact; never invent observations.", "parameters": {"type": "object", "additionalProperties": True, "properties": {}}}}
    for name in TOOL_IDS
]

SYSTEM_PROMPT = """You are an offline production-incident diagnostic agent. Use only the allowed diagnostic tools and their returned observations. Do not invent tool names, arguments, observations, labels, or evidence. You may call one tool at a time, or return the final diagnosis.

When calling a tool, use the native function-call format exactly.
When finalizing, return JSON only with exactly these keys:
{\"native_fault_type\": string, \"native_fault_category\": string, \"fault_object\": string}

Do not explain your reasoning. Do not output hidden reasoning. The incident request and tool observations are the complete model-visible input; benchmark metadata, process labels, golden answers, and source paths are unavailable.
"""


def _fp(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def build_validation_subsplit(split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Select one deterministic validation case per native type.

    The frozen VALIDATION partition is never changed.  The remaining cases are
    selection-only material and are intentionally not passed to model calls.
    """

    entries = [item for item in split_manifest["entries"] if item["split"] == "VALIDATION"]
    by_type: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in entries:
        by_type[str(item["source_fault_type"])].append(item)
    selected: list[Mapping[str, Any]] = []
    for fault_type in sorted(by_type):
        selected.append(sorted(by_type[fault_type], key=lambda item: item["source_case_group"])[0])
    selected_ids = {item["source_case_group"] for item in selected}
    selection = [item for item in entries if item["source_case_group"] not in selected_ids]
    payload = {
        "version": SCREEN_MANIFEST_VERSION,
        "source_revision": SOURCE_REVISION,
        "split_manifest_fingerprint": split_manifest.get("fingerprint"),
        "selection_policy": "one lexicographically first case per native fault type from frozen VALIDATION; no result-dependent selection",
        "screen_split": "VALIDATION_SCREEN",
        "selection_split": "VALIDATION_SELECTION",
        "screen_case_groups": sorted(item["source_case_group"] for item in selected),
        "selection_case_groups": sorted(item["source_case_group"] for item in selection),
        "screen_count": len(selected),
        "selection_count": len(selection),
        "screen_fault_types": sorted({item["source_fault_type"] for item in selected}),
        "selection_model_predictions": False,
        "test_model_predictions": False,
    }
    payload["fingerprint"] = _fp(payload)
    return payload


def validate_validation_subsplit(manifest: Mapping[str, Any], split_manifest: Mapping[str, Any]) -> None:
    if manifest.get("split_manifest_fingerprint") != split_manifest.get("fingerprint"):
        raise ValueError("validation subsplit does not reference the frozen split")
    groups = set(manifest.get("screen_case_groups", [])) | set(manifest.get("selection_case_groups", []))
    validation = {item["source_case_group"] for item in split_manifest["entries"] if item["split"] == "VALIDATION"}
    if groups != validation or set(manifest.get("screen_case_groups", [])) & set(manifest.get("selection_case_groups", [])):
        raise ValueError("validation sub-split is not a disjoint complete partition")
    if manifest.get("screen_count") != len(manifest.get("screen_case_groups", [])):
        raise ValueError("screen count mismatch")
    if manifest.get("screen_fault_types") != sorted({item["source_fault_type"] for item in split_manifest["entries"] if item["source_case_group"] in set(manifest["screen_case_groups"])}):
        raise ValueError("screen label coverage changed")


def tool_contract() -> dict[str, Any]:
    return {
        "version": "cloud-opsbench-03e-offline-agent-tool-contract-v1",
        "mode": "deterministic offline replay from immutable case-local artifacts",
        "tools": TOOL_SCHEMAS,
        "unknown_tool_policy": "protocol failure; no repair",
        "invalid_argument_policy": "protocol failure; no repair",
        "max_agent_steps": MAX_AGENT_STEPS,
    }


def evaluation_contract() -> dict[str, Any]:
    return {
        "version": SCREENING_VERSION,
        "primary_mode": "CLOSED_LOOP_AGENT",
        "secondary_modes": ["ORACLE_EVIDENCE_DIAGNOSIS", "TEACHER_FORCED_NEXT_ACTION", "STATIC_EVIDENCE_DIAGNOSIS"],
        "source_revision": SOURCE_REVISION,
        "split_fingerprint": SPLIT_FINGERPRINT,
        "representation_fingerprint": REPRESENTATION_FINGERPRINT,
        "thinking": {"enabled": False, "contract": "explicit enable_thinking=False"},
        "prompt": {"system": SYSTEM_PROMPT, "tools": TOOL_SCHEMAS, "final_schema": ["native_fault_type", "native_fault_category", "fault_object"], "few_shot": False},
        "generation": {"version": GENERATION_VERSION, "do_sample": False, "temperature": None, "top_p": None, "max_new_tokens": 256, "max_agent_steps": MAX_AGENT_STEPS, "retry_wrong_diagnosis": False, "output_repair": False},
        "targets": ["native_fault_type", "native_fault_category", "fault_object"],
        "test_policy": "TEST is sealed and model-facing content is forbidden",
    }


def _strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Qwen chat-template control tokens are serialization delimiters, not
    # model content.  ``decode(..., skip_special_tokens=False)`` intentionally
    # preserves them for auditability, so parsers remove only the known control
    # token form before validating the frozen output contract.
    text = re.sub(r"<\|(?:im_start|im_end|endoftext|assistant|user|system|tool)\|>", "", text)
    return text.strip()


def parse_final_diagnosis(text: str) -> tuple[dict[str, Any] | None, str]:
    clean = _strip_reasoning(text)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(value, dict) or set(value) != {"native_fault_type", "native_fault_category", "fault_object"}:
        return None, "wrong_final_schema"
    if any(not isinstance(value[key], str) or not value[key].strip() for key in value):
        return None, "invalid_final_values"
    return value, "valid"


def parse_tool_call(text: str) -> tuple[str | None, dict[str, Any] | None, str]:
    clean = _strip_reasoning(text)
    match = re.search(r"<tool_call>\s*<function=([^>]+)>(.*?)</function>\s*</tool_call>", clean, re.DOTALL)
    if match:
        name = match.group(1).strip()
        args: dict[str, Any] = {}
        for parameter in re.finditer(r"<parameter=([^>]+)>\s*(.*?)\s*</parameter>", match.group(2), re.DOTALL):
            key, value = parameter.group(1).strip(), parameter.group(2).strip()
            try:
                args[key] = json.loads(value)
            except json.JSONDecodeError:
                args[key] = value
        return name, args, "xml_tool_call"
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        return None, None, "not_tool_call"
    if isinstance(value, dict) and value.get("type") in {"tool_call", "tool"} and isinstance(value.get("tool"), str):
        return value["tool"], value.get("arguments", {}), "json_tool_call"
    return None, None, "not_tool_call"


def _cache_key_args(key: str, tool: str) -> dict[str, Any]:
    if not key.startswith(tool + ":"):
        return {}
    try:
        value = json.loads(key[len(tool) + 1:])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _args_equal(left: Any, right: Any) -> bool:
    if left in (None, "") and right in (None, ""):
        return True
    if isinstance(left, list):
        return right in left or left == right
    if isinstance(right, list):
        return left in right or left == right
    return left == right


class OfflineToolReplay:
    """Case-local replay that never reads target/process/golden answer fields."""

    def __init__(self, root: Path, case: CloudOpsBenchCase):
        self.root = root
        self.case = case
        self.cache = _cache_index(root, case)

    def execute(self, tool: str, arguments: Mapping[str, Any]) -> dict[str, Any]:
        if tool not in TOOL_IDS:
            return {"ok": False, "error": "unknown_tool", "tool": tool}
        if not isinstance(arguments, Mapping):
            return {"ok": False, "error": "arguments_must_be_object", "tool": tool}
        candidates = [(key, value) for key, value in self.cache.items() if key == tool or key.startswith(tool + ":") or key.startswith(tool + "-")]
        scored: list[tuple[int, str, str]] = []
        for key, value in candidates:
            key_args = _cache_key_args(key, tool)
            if all(name in key_args and _args_equal(value_arg, key_args[name]) for name, value_arg in arguments.items()):
                scored.append((sum(name in key_args for name in arguments), key, value))
        if not scored:
            # Empty/partially specified calls use a stable tool default.  This
            # is replay semantics, not target-aware selection.
            if candidates:
                key, value = sorted(candidates)[0]
            else:
                return {"ok": False, "error": "no_replay_observation", "tool": tool}
        else:
            _, key, value = max(scored, key=lambda item: (item[0], item[1]))
        packaged = _uniform_line_package(value, MAX_OBSERVATION_CHARS)
        return {"ok": True, "tool": tool, "arguments": dict(arguments), "cache_key": key, "observation": packaged["text"], "packaged_chars": packaged["packaged_chars"], "original_chars": packaged["original_chars"], "omitted": packaged["omitted"], "replay_status": "RESOLVED_FROM_TOOL_CACHE"}


def _model_case_lookup(cases: Iterable[CloudOpsBenchCase], split_manifest: Mapping[str, Any], allowed: set[str]) -> dict[str, CloudOpsBenchCase]:
    split_by_group = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    result = {}
    for case in cases:
        group = source_case_group_id(case)
        split = split_by_group.get(group)
        if group in allowed:
            if split != "VALIDATION":
                raise ValueError("model-facing screen contains non-validation case")
            result[group] = case
        elif split == "TEST" and group in allowed:
            raise ValueError("TEST is sealed")
    return result


def _initial_query(case: CloudOpsBenchCase) -> str:
    query = str(case.ground_truth_metadata.get("query", "Investigate the reported service condition.")).strip()
    if not query:
        raise ValueError("empty sanitized incident query")
    return query


def _messages_for_initial(case: CloudOpsBenchCase) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Incident request: {_initial_query(case)}"}]


def _assistant_tool_message(raw: str) -> dict[str, Any]:
    return {"role": "assistant", "content": _strip_reasoning(raw)}


def _tool_message(observation: str) -> dict[str, Any]:
    return {"role": "tool", "content": observation}


def _target(case: CloudOpsBenchCase) -> dict[str, Any]:
    values = case.ground_truth_metadata
    target = {"native_fault_type": values.get("fault_type"), "native_fault_category": values.get("fault_category"), "fault_object": values.get("component")}
    if any(not isinstance(value, str) or not value for value in target.values()):
        raise ValueError(f"missing deterministic target for {source_case_group_id(case)}")
    return target


def _metric(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_diagnoses(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    valid = [item for item in records if item.get("prediction") is not None]
    fields = ("native_fault_type", "native_fault_category", "fault_object")
    result: dict[str, Any] = {"count": len(records), "schema_valid_count": len(valid), "schema_valid_rate": _metric([bool(item.get("prediction")) for item in records])}
    for field in fields:
        result[field + "_exact"] = _metric([bool(item.get("prediction") and item["prediction"].get(field) == item["target"].get(field)) for item in records])
    result["joint_exact"] = _metric([bool(item.get("prediction") and all(item["prediction"].get(field) == item["target"].get(field) for field in fields)) for item in records])
    result["by_fault_type"] = _group_diagnosis(records, "fault_type")
    result["by_fault_category"] = _group_diagnosis(records, "fault_category")
    result["by_system"] = _group_diagnosis(records, "system")
    return result


def _group_diagnosis(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in records:
        buckets[str(item.get(key) or item.get("target", {}).get("native_fault_type", "UNKNOWN"))].append(item)
    return {name: {"count": len(rows), "joint_exact": _metric([bool(row.get("prediction") and all(row["prediction"].get(field) == row["target"].get(field) for field in ("native_fault_type", "native_fault_category", "fault_object"))) for row in rows])} for name, rows in sorted(buckets.items())}


def history_from_trace(root: Path, case: CloudOpsBenchCase, path_name: str, *, include_final_schema: bool = True) -> list[dict[str, Any]]:
    normalized = normalize_trajectory(root, case, path_name)
    messages = _messages_for_initial(case)
    for step in normalized["replay_steps"]:
        if step["replay_status"] == "UNRESOLVED":
            raise ValueError("unresolved replay step used in model-facing history")
        messages.append({"role": "assistant", "content": f"<tool_call>\n<function={step['tool_id']}>\n</function>\n</tool_call>"})
        messages.append(_tool_message(step["observation"]["text"]))
    if include_final_schema:
        messages.append({"role": "user", "content": "Return the final structured diagnosis JSON now."})
    return messages


def oracle_history(root: Path, case: CloudOpsBenchCase) -> tuple[list[dict[str, Any]], str]:
    for path_name in ("golden_path1", "golden_path2"):
        try:
            return history_from_trace(root, case, path_name), path_name
        except ValueError:
            continue
    raise ValueError(f"no executable golden history for {source_case_group_id(case)}")


def static_history(root: Path, case: CloudOpsBenchCase) -> list[dict[str, Any]]:
    # 03D's already-frozen static view is the deterministic first replayable
    # observation stream, never the raw snapshot and never target-aware.
    return oracle_history(root, case)[0]


def model_context_messages_for_trajectory(root: Path, case: CloudOpsBenchCase, path_name: str) -> list[list[dict[str, Any]]]:
    normalized = normalize_trajectory(root, case, path_name)
    histories: list[list[dict[str, Any]]] = []
    messages = _messages_for_initial(case)
    for step in normalized["replay_steps"]:
        histories.append(messages + [{"role": "user", "content": "Return one next diagnostic tool call or the final diagnosis JSON."}])
        messages = messages + [_assistant_tool_message(f"<tool_call><function={step['tool_id']}></function></tool_call>"), _tool_message(step["observation"]["text"])]
    histories.append(messages + [{"role": "user", "content": "Return the final structured diagnosis JSON now."}])
    return histories


def target_for_case(case: CloudOpsBenchCase) -> dict[str, Any]:
    return _target(case)


def compact_failure_category(record: Mapping[str, Any]) -> str:
    if record.get("parse_category") in {"malformed_json", "wrong_final_schema", "invalid_final_values"}:
        return "schema_failure"
    prediction = record.get("prediction")
    if not prediction:
        return str(record.get("failure_category") or "no_final_diagnosis")
    target = record["target"]
    if prediction.get("native_fault_type") != target.get("native_fault_type") and prediction.get("fault_object") != target.get("fault_object"):
        return "fault_type_and_object_wrong"
    if prediction.get("native_fault_type") != target.get("native_fault_type"):
        return "fault_type_wrong"
    if prediction.get("fault_object") != target.get("fault_object"):
        return "fault_object_wrong"
    if prediction.get("native_fault_category") != target.get("native_fault_category"):
        return "category_wrong"
    return "correct"


def trajectory_metadata(case: CloudOpsBenchCase, split: str, path: str, *, trajectory_steps: int) -> dict[str, Any]:
    return {"source_revision": SOURCE_REVISION, "source_case_group": source_case_group_id(case), "source_case_id": case.source_case_id, "split": split, "source_system": case.source_system, "source_fault_category": case.source_fault_category, "source_fault_type": case.source_fault_type, "upstream_difficulty": case.upstream_difficulty, "code_available": case.references.code is not None, "metrics_available": case.references.metrics is not None, "source_fingerprint": case.source_fingerprint, "path": path, "trajectory_steps": trajectory_steps}
