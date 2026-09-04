"""Experiment 03D training contract and protected split construction.

The module produces references and deterministic message specifications, not a
copy of Cloud-OpsBench.  Source metadata supplies targets; golden traces supply
tool/action supervision; raw observations remain bounded, auditable inputs.
"""

from __future__ import annotations

import hashlib
import json
import re
import ast
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from causetune.incident_telemetry.fingerprints import canonical_json

from .models import CloudOpsBenchCase
from .scanner import CorpusScan


TRAINING_CONTRACT_VERSION = "cloud-opsbench-training-contract-v1"
PACKAGING_VERSION = "cloud-opsbench-evidence-packaging-v1"
TARGET_CONTRACT_VERSION = "cloud-opsbench-target-v1"
TOOL_CONTRACT_VERSION = "cloud-opsbench-offline-tool-replay-v1"
NORMALIZATION_VERSION = "cloud-opsbench-trajectory-normalization-v1"
SUPERVISION_POLICY_VERSION = "cloud-opsbench-supervision-v1"
FREEZE_VERSION = "cloud-opsbench-03d-freeze-v1"
MAX_OBSERVATION_CHARS = 32768

TOOL_IDS = (
    "GetResources", "DescribeResource", "GetAppYAML", "GetServiceDependencies",
    "CheckServiceConnectivity", "GetAlerts", "GetRecentLogs", "GetErrorLogs",
    "ListCodeFiles", "GetSourceCode", "GetClusterConfiguration", "CheckNodeServiceStatus",
)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _case_group(case: CloudOpsBenchCase) -> str:
    """03D identity for a source case, including its full benchmark path."""

    return f"{case.source_revision}:{case.source_system}:{case.source_fault_category}:{case.source_case_id}"


def source_case_group_id(case: CloudOpsBenchCase) -> str:
    """Return the immutable 03D source-case grouping identity."""

    return _case_group(case)


def source_manifest_fingerprint(scan: CorpusScan) -> str:
    return sha256_value({"source_revision": scan.source_revision, "cases": [{"case_id": _case_group(case), "source_fingerprint": case.source_fingerprint} for case in scan.cases]})


def target_contract() -> dict[str, Any]:
    return {
        "version": TARGET_CONTRACT_VERSION,
        "fields": {
            "native_fault_type": {"source": "metadata.result.root_cause", "required": True},
            "native_fault_category": {"source": "metadata.result.fault_taxonomy", "required": True},
            "fault_object": {"source": "metadata.result.fault_object", "required": True},
        },
        "native_taxonomy_authoritative": True,
        "confidence": "not included",
        "reasoning_target": "not included",
        "derivation": "mechanical extraction only; no generated labels",
    }


def tool_contract() -> dict[str, Any]:
    return {
        "version": TOOL_CONTRACT_VERSION,
        "mode": "offline replay from immutable case-local tool_cache.json and trajectory observations",
        "live_kubernetes": False,
        "tools": [{"tool_id": tool, "source_cache_key_rule": f"case-local keys beginning with {tool}"} for tool in TOOL_IDS],
        "unknown_tools": "fail closed",
        "argument_policy": "preserve normalized calling text; do not infer labels from arguments",
        "observation_policy": "resolve by exact cached output when possible; otherwise retain the immutable golden-trace observation and flag the fallback",
    }


def model_visible_schema() -> dict[str, Any]:
    return {
        "version": "cloud-opsbench-model-visible-input-v1",
        "initial_context": {"incident_request": "sanitized metadata.query after leakage review", "source_labels": False, "source_paths": False, "case_id": False, "process_labels": False, "golden_trajectories": False},
        "tool_action": {"tool_id": "allowlisted tool ID", "arguments": "normalized source calling text"},
        "tool_observation": {"observation": "bounded deterministic package", "source_target": False, "packaging_metadata": "provenance only"},
        "final_diagnosis": "supervision-only structured target; not part of initial input",
        "forbidden_input_fields": ["metadata.result.*", "fault taxonomy/category", "fault object", "source path", "process-label", "golden final answer", "source fingerprint"],
    }


def packaging_contract() -> dict[str, Any]:
    return {
        "version": PACKAGING_VERSION,
        "primary_input": "bounded diagnostic interaction; full raw snapshot rejected",
        "max_observation_chars": MAX_OBSERVATION_CHARS,
        "selection": "full output when within bound; otherwise deterministic structure-aware line sampling with omission flag",
        "label_aware_selection": False,
        "target_aware_selection": False,
        "process_label_aware_selection": False,
        "truncation": "never silent; original/package sizes and status are recorded",
        "required_evidence_policy": "cannot be certified by target labels; over-budget observations require review",
        "modalities": {
            "metrics": ["MODALITY_NOT_AVAILABLE", "MODALITY_AVAILABLE_BUT_EMPTY", "AVAILABLE"],
            "code": ["MODALITY_NOT_AVAILABLE", "AVAILABLE"],
            "all_other_core_snapshot_files": ["AVAILABLE", "PACKAGING_FAILED"],
        },
    }


def _load_json(root: Path, relative_path: str) -> Any:
    return json.loads((root / relative_path).read_text(encoding="utf-8"))


def _sanitized_query(case: CloudOpsBenchCase) -> str:
    query = str(case.ground_truth_metadata.get("query", "Investigate the reported service condition.")).strip()
    forbidden = (case.source_fault_type, case.source_fault_category, str(case.ground_truth_metadata.get("component", "")), case.source_case_id)
    lowered = query.casefold()
    if any(value and value.casefold() in lowered for value in forbidden):
        raise ValueError(f"{_case_group(case)} query contains an explicit target/path label")
    return query


def _trace(root: Path, case: CloudOpsBenchCase, path_name: str) -> list[dict[str, Any]]:
    reference = getattr(case.references, path_name)
    if reference is None:
        return []
    payload = _load_json(root, reference.relative_path)
    trace = payload.get("diagnostic_trace") if isinstance(payload, Mapping) else None
    if not isinstance(trace, list) or any(not isinstance(item, Mapping) for item in trace):
        raise ValueError(f"{_case_group(case)}/{path_name} has invalid diagnostic_trace")
    return [dict(item) for item in trace]


def _normalize_text(value: str) -> str:
    value = re.sub(r"\s+", " ", value.lower()).strip()
    return re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", value)


def _argument_pattern(value: str) -> str:
    return _normalize_text(re.sub(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", "<value>", value))


def _trajectory_signatures(trace: list[dict[str, Any]], raw_payload: Any) -> dict[str, str]:
    exact = sha256_value(raw_payload)
    normalized_records = [{"tool_name": item.get("tool_name"), "calling": _normalize_text(str(item.get("calling", ""))), "output": _normalize_text(str(item.get("output", "")))} for item in trace]
    sequence = [str(item.get("tool_name", "")) for item in trace]
    argument_pattern = [_argument_pattern(str(item.get("calling", ""))) for item in trace]
    return {
        "exact_serialized": exact,
        "normalized_trajectory": sha256_value(normalized_records),
        "tool_call_sequence": sha256_value(sequence),
        "tool_call_argument_pattern": sha256_value(argument_pattern),
    }


def _uniform_line_package(text: str, max_chars: int = MAX_OBSERVATION_CHARS) -> dict[str, Any]:
    if len(text) <= max_chars:
        return {"text": text, "status": "PACKAGED", "omitted": False, "original_chars": len(text), "packaged_chars": len(text), "version": PACKAGING_VERSION}
    lines = text.splitlines()
    if not lines:
        return {"text": "", "status": "PACKAGING_FAILED", "omitted": True, "original_chars": len(text), "packaged_chars": 0, "version": PACKAGING_VERSION}
    head = min(32, len(lines))
    tail = min(32, max(0, len(lines) - head))
    selected = list(lines[:head])
    middle_budget = max(0, max_chars - sum(len(line) + 1 for line in lines[:head]) - sum(len(line) + 1 for line in lines[-tail:] if tail))
    middle = lines[head:len(lines) - tail if tail else len(lines)]
    if middle and middle_budget > 0:
        step = max(1, len(middle) // max(1, middle_budget // 80))
        selected.extend(middle[::step])
    if tail:
        selected.extend(lines[-tail:])
    packaged = "\n".join(selected)
    if len(packaged) > max_chars:
        packaged = packaged[:max_chars]
    return {"text": packaged, "status": "PACKAGED_WITH_OMISSION", "omitted": True, "original_chars": len(text), "packaged_chars": len(packaged), "version": PACKAGING_VERSION, "selection": "deterministic head/tail plus uniform middle line sampling"}


def _cache_index(root: Path, case: CloudOpsBenchCase) -> dict[str, str]:
    payload = _load_json(root, case.references.tool_cache.relative_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{_case_group(case)} tool cache must be an object")
    values: dict[str, str] = {}
    for key, value in payload.items():
        if isinstance(value, str):
            values[str(key)] = value
        else:
            values[str(key)] = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return values


def _calling_arguments(calling: str) -> dict[str, Any]:
    match = re.search(r"arguments\s*=\s*\{(.*)\}\s*$", calling)
    if not match:
        return {}
    expression = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r"\1'\2':", "{" + match.group(1) + "}")
    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _cache_key_arguments(key: str, tool_name: str) -> dict[str, Any]:
    if not key.startswith(tool_name + ":"):
        return {}
    try:
        value = json.loads(key[len(tool_name) + 1:])
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _argument_matches(expected: Any, actual: Any) -> bool:
    if expected is None:
        return actual in (None, "")
    if isinstance(expected, list):
        return actual in expected or expected == actual
    return expected == actual


def _resolve_cache_key(cache: Mapping[str, str], tool_name: str, output: str, calling: str = "") -> tuple[str | None, list[str]]:
    exact = sorted(key for key, value in cache.items() if value == output and (key == tool_name or key.startswith(tool_name + ":") or key.startswith(tool_name + "-")))
    candidates = sorted(key for key in cache if key == tool_name or key.startswith(tool_name + ":") or key.startswith(tool_name + "-"))
    if not exact:
        try:
            decoded_output = json.loads(output)
        except json.JSONDecodeError:
            decoded_output = None
        if isinstance(decoded_output, str):
            exact = sorted(key for key, value in cache.items() if value == decoded_output and (key == tool_name or key.startswith(tool_name + ":") or key.startswith(tool_name + "-")))
    if exact:
        return exact[0], candidates
    arguments = _calling_arguments(calling)
    if arguments:
        matching: list[tuple[int, str]] = []
        for key in candidates:
            key_arguments = _cache_key_arguments(key, tool_name)
            if key_arguments and all(name in key_arguments and _argument_matches(value, key_arguments[name]) for name, value in arguments.items()):
                exact_fields = sum(key_arguments[name] == value for name, value in arguments.items() if name in key_arguments)
                matching.append((-exact_fields, key))
        if matching:
            return sorted(matching)[0][1], candidates
    return None, candidates


def normalize_trajectory(root: Path, case: CloudOpsBenchCase, path_name: str) -> dict[str, Any]:
    trace = _trace(root, case, path_name)
    cache = _cache_index(root, case)
    messages: list[dict[str, Any]] = [{"role": "user", "content": _sanitized_query(case)}]
    replay_steps = []
    for index, item in enumerate(trace):
        tool_name = str(item.get("tool_name", ""))
        if tool_name not in TOOL_IDS:
            raise ValueError(f"unknown golden tool: {tool_name}")
        calling = str(item.get("calling", ""))
        output = str(item.get("output", ""))
        cache_key, candidates = _resolve_cache_key(cache, tool_name, output, calling)
        package = _uniform_line_package(output)
        replay_status = "RESOLVED_FROM_TOOL_CACHE" if cache_key else "RESOLVED_FROM_GOLDEN_TRACE"
        replay_steps.append({"step": index, "tool_id": tool_name, "arguments": _argument_pattern(calling), "observation": package, "tool_cache_key": cache_key, "tool_cache_candidates": candidates[:20], "replay_status": replay_status})
        messages.append({"role": "assistant", "action": {"tool_id": tool_name, "arguments": _argument_pattern(calling)}})
        messages.append({"role": "tool", "tool_id": tool_name, "observation": package["text"], "source": "tool_cache.json" if cache_key else "golden diagnostic_trace (immutable fallback)", "replay_status": replay_status})
    target = {"native_fault_type": case.ground_truth_metadata.get("fault_type"), "native_fault_category": case.ground_truth_metadata.get("fault_category"), "fault_object": case.ground_truth_metadata.get("component")}
    messages.append({"role": "assistant", "diagnosis": target})
    return {"path": path_name, "messages": messages, "replay_steps": replay_steps, "target": target, "reasoning_disposition": "DROP", "reasoning_fields_present": False, "normalization_version": NORMALIZATION_VERSION}


def _union_find(items: Iterable[str], links: Iterable[tuple[str, str]]) -> list[list[str]]:
    parent = {item: item for item in items}
    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item
    for left, right in links:
        if left not in parent or right not in parent:
            continue
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root
    groups: dict[str, list[str]] = defaultdict(list)
    for item in sorted(parent):
        groups[find(item)].append(item)
    return [members for members in groups.values() if len(members) > 1]


def build_duplicate_report(root: Path, cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    values = tuple(cases)
    records = []
    for case in values:
        for path_name in ("golden_path1", "golden_path2"):
            reference = getattr(case.references, path_name)
            trace = _trace(root, case, path_name)
            payload = _load_json(root, reference.relative_path) if reference else None
            signatures = _trajectory_signatures(trace, payload)
            records.append({"trajectory_id": f"{_case_group(case)}:{path_name}", "source_case": _case_group(case), "path": path_name, **signatures, "step_count": len(trace)})
    groups = {}
    for signature_name in ("exact_serialized", "normalized_trajectory", "tool_call_sequence", "tool_call_argument_pattern"):
        buckets: dict[str, list[str]] = defaultdict(list)
        for record in records:
            buckets[record[signature_name]].append(record["trajectory_id"])
        groups[signature_name] = [{"signature": signature, "members": sorted(members)} for signature, members in sorted(buckets.items()) if len(members) > 1]
    contamination_links = []
    normalized_buckets: dict[str, list[str]] = defaultdict(list)
    for record in records:
        normalized_buckets[record["normalized_trajectory"]].append(record["source_case"])
    for members in normalized_buckets.values():
        unique = sorted(set(members))
        contamination_links.extend((unique[index], unique[index + 1]) for index in range(len(unique) - 1))
    return {"version": "cloud-opsbench-duplicate-audit-v1", "trajectory_count": len(records), "records": records, "groups": groups, "contamination_case_groups": _union_find({_case_group(case) for case in values}, contamination_links), "common_tool_sequences_are_not_contamination_groups": True, "near_duplicate_method": "normalized whitespace/literal-number signatures only; embeddings deferred"}


def _assign_groups(cases: tuple[CloudOpsBenchCase, ...], contamination_groups: list[list[str]]) -> dict[str, str]:
    case_ids = {_case_group(case) for case in cases}
    links = [(left, right) for group in contamination_groups for left, right in zip(sorted(group), sorted(group)[1:])]
    components = _union_find(case_ids, links)
    grouped_ids = {case_id for group in components for case_id in group}
    groups = [tuple(group) for group in components] + [(case_id,) for case_id in sorted(case_ids - grouped_ids)]
    ordered = sorted(groups, key=lambda group: (-len(group), hashlib.sha256("|".join(group).encode()).hexdigest()))
    by_id = {_case_group(case): case for case in cases}
    label_totals = Counter(case.source_fault_type for case in cases)
    assigned: dict[str, str] = {}
    counts = Counter()
    labels_by_split = {split: Counter() for split in ("TRAIN", "VALIDATION", "TEST")}
    target = {"TRAIN": 0.70 * len(cases), "VALIDATION": 0.15 * len(cases), "TEST": 0.15 * len(cases)}
    for group in ordered:
        members = [by_id[case_id] for case_id in group]
        labels = Counter(case.source_fault_type for case in members)
        options: list[tuple[float, float, int, str]] = []
        for split in ("TRAIN", "VALIDATION", "TEST"):
            missing_coverage = sum(1 for label in labels if label_totals[label] >= 3 and not labels_by_split[split][label])
            overflow = max(0.0, counts[split] + len(group) - target[split])
            options.append((-missing_coverage, overflow, counts[split], split))
        split = min(options)[3]
        for case_id in group:
            assigned[case_id] = split
        counts[split] += len(group)
        for label, count in labels.items():
            labels_by_split[split][label] += count

    # A deterministic repair pass guarantees coverage for labels with at least
    # three independent source cases, while keeping contamination groups atomic.
    for label, total in sorted(label_totals.items()):
        if total < 3:
            continue
        for wanted in ("TRAIN", "VALIDATION", "TEST"):
            if any(assigned[_case_group(case)] == wanted and case.source_fault_type == label for case in cases):
                continue
            candidates = []
            for group in ordered:
                members = [by_id[case_id] for case_id in group]
                source = assigned[group[0]]
                if source == wanted:
                    continue
                label_count = sum(case.source_fault_type == label for case in members)
                if label_count and sum(assigned[_case_group(case)] == source and case.source_fault_type == label for case in cases) > label_count:
                    candidates.append((abs((counts[wanted] + len(group)) - target[wanted]), abs(counts[source] - target[source]), group))
            if not candidates:
                continue
            group = min(candidates, key=lambda item: (item[0], item[1], item[2]))[2]
            source = assigned[group[0]]
            members = [by_id[case_id] for case_id in group]
            counts[source] -= len(group)
            counts[wanted] += len(group)
            for member in members:
                assigned[_case_group(member)] = wanted
    return assigned


def build_split_manifest(scan: CorpusScan, duplicate_report: Mapping[str, Any]) -> dict[str, Any]:
    cases = tuple(scan.cases)
    contamination_groups = duplicate_report.get("contamination_case_groups", [])
    assignments = _assign_groups(cases, contamination_groups)
    entries = []
    for case in sorted(cases, key=_case_group):
        case_group = _case_group(case)
        split = assignments[case_group]
        labels = case.ground_truth_metadata
        slices = [case.upstream_difficulty.upper() if case.upstream_difficulty else "UNSPECIFIED", "CODE_DEFECT" if case.source_fault_category == "Application_Code_Defect" else None, "METRICS_AVAILABLE" if case.references.metrics else "METRICS_UNAVAILABLE", "ONLINE_BOUTIQUE" if case.source_system == "boutique" else "TRAIN_TICKET"]
        entries.append({"source_case_group": case_group, "source_system": case.source_system, "source_case_id": case.source_case_id, "source_fault_category": case.source_fault_category, "source_fault_type": case.source_fault_type, "upstream_difficulty": case.upstream_difficulty, "code_available": case.references.code is not None, "metrics_available": case.references.metrics is not None, "split": split, "slices": [item for item in slices if item], "source_fingerprint": case.source_fingerprint, "path1_present": case.references.golden_path1 is not None, "path2_present": case.references.golden_path2 is not None, "target_fingerprint": sha256_value({"native_fault_type": labels.get("fault_type"), "native_fault_category": labels.get("fault_category"), "fault_object": labels.get("component")})})
    manifest = {"version": "cloud-opsbench-protected-split-v1", "source_revision": scan.source_revision, "assignment_policy": "case-grouped deterministic 70/15/15 approximation; contamination groups atomic; labels with >=3 cases covered where feasible", "grouping_unit": "immutable source case plus all derived artifacts", "test_protected": True, "test_consumption": "only final explicitly defined evaluation", "selection_uses_test": False, "slices": {"EASY": "upstream_difficulty == easy", "MEDIUM": "upstream_difficulty == medium", "HARD": "upstream_difficulty == hard", "CODE_DEFECT": "code/ is present", "METRICS_AVAILABLE": "metrics.csv is present", "METRICS_UNAVAILABLE": "metrics.csv is absent by source design", "ONLINE_BOUTIQUE": "source_system == boutique", "TRAIN_TICKET": "source_system == trainticket", "CROSS_SYSTEM": "exploratory only; label-confounded", "UNSEEN_FAULT_TYPE": "exploratory only and defined relative to a future training partition"}, "entries": entries}
    manifest["fingerprint"] = sha256_value(manifest)
    return manifest


def validate_split_manifest(manifest: Mapping[str, Any], duplicate_report: Mapping[str, Any], expected_case_ids: Iterable[str]) -> dict[str, Any]:
    entries = list(manifest.get("entries", []))
    expected = set(expected_case_ids)
    seen = [str(item.get("source_case_group")) for item in entries]
    errors = []
    if set(seen) != expected or len(seen) != len(set(seen)):
        errors.append("each source case must occur exactly once")
    if any(item.get("split") not in {"TRAIN", "VALIDATION", "TEST"} for item in entries):
        errors.append("unknown protected split")
    split_by_case = {item["source_case_group"]: item["split"] for item in entries}
    for group in duplicate_report.get("contamination_case_groups", []):
        assigned = {split_by_case.get(case_id) for case_id in group}
        if len(assigned) > 1:
            errors.append(f"contamination group crosses splits: {sorted(group)}")
    return {"status": "pass" if not errors else "fail", "errors": errors, "case_count": len(entries), "split_counts": dict(sorted(Counter(item.get("split") for item in entries).items()))}


def split_distribution(manifest: Mapping[str, Any]) -> dict[str, Any]:
    entries = list(manifest["entries"])
    by_split = {}
    for split in ("TRAIN", "VALIDATION", "TEST"):
        subset = [item for item in entries if item["split"] == split]
        by_split[split] = {"count": len(subset), "native_fault_types": dict(sorted(Counter(item["source_fault_type"] for item in subset).items())), "categories": dict(sorted(Counter(item["source_fault_category"] for item in subset).items())), "systems": dict(sorted(Counter(item["source_system"] for item in subset).items())), "difficulty": dict(sorted(Counter(item["upstream_difficulty"] or "UNSPECIFIED" for item in subset).items())), "code_available": sum(item["code_available"] for item in subset), "metrics_available": sum(item["metrics_available"] for item in subset)}
    labels = {split: set(value["native_fault_types"]) for split, value in by_split.items()}
    all_labels = set().union(*labels.values())
    coverage = {label: {split: label in labels[split] for split in labels} for label in sorted(all_labels)}
    counts = Counter(item["source_fault_type"] for item in entries)
    slice_counts = Counter(slice_name for item in entries for slice_name in item["slices"])
    by_split_slices = {split: dict(sorted(Counter(slice_name for item in entries if item["split"] == split for slice_name in item["slices"]).items())) for split in by_split}
    return {"version": "cloud-opsbench-split-distribution-v1", "by_split": by_split, "slice_counts": dict(sorted(slice_counts.items())), "by_split_slice_counts": by_split_slices, "label_overlap": {"train_validation": sorted(labels["TRAIN"] & labels["VALIDATION"]), "train_test": sorted(labels["TRAIN"] & labels["TEST"]), "validation_test": sorted(labels["VALIDATION"] & labels["TEST"])}, "test_only_labels": sorted(labels["TEST"] - labels["TRAIN"]), "label_coverage": coverage, "rare_labels": {label: count for label, count in sorted(counts.items()) if count < 3}, "split_percentages": {split: round(value["count"] / max(1, len(entries)) * 100, 3) for split, value in by_split.items()}}


def build_static_view_manifest(scan: CorpusScan, split_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    records = []
    for case in sorted(scan.cases, key=_case_group):
        modalities = {}
        for name in ("k8s_states", "logs", "metrics", "alerts", "code"):
            reference = getattr(case.references, name)
            if reference is None:
                state = "MODALITY_NOT_AVAILABLE"
            elif reference.byte_size == 0:
                state = "MODALITY_AVAILABLE_BUT_EMPTY"
            else:
                state = "AVAILABLE"
            modalities[name] = {"state": state, "source_reference": reference.to_dict() if reference else None, "model_visible_path": False}
        group = _case_group(case)
        records.append({"example_id": "static_03d_" + hashlib.sha256(group.encode()).hexdigest()[:20], "source_case_group": group, "split": split_by_case[group], "initial_context": {"query_field": "sanitized metadata.query", "query_value": _sanitized_query(case)}, "modalities": modalities, "target_included": False, "packaging_version": PACKAGING_VERSION})
    return records


def build_trajectory_example_manifest(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    records = []
    for case in sorted(scan.cases, key=_case_group):
        for path_name in ("golden_path1", "golden_path2"):
            normalized = normalize_trajectory(root, case, path_name)
            group = _case_group(case)
            message_specs = [{"role": "user", "content_source": "sanitized metadata.query"}]
            for step in normalized["replay_steps"]:
                message_specs.extend(({"role": "assistant", "action": {"tool_id": step["tool_id"], "arguments": step["arguments"]}}, {"role": "tool", "observation_source": "tool_cache.json" if step["replay_status"] == "RESOLVED_FROM_TOOL_CACHE" else "golden diagnostic_trace", "step": step["step"], "replay_status": step["replay_status"], "packaging": {"original_chars": step["observation"]["original_chars"], "packaged_chars": step["observation"]["packaged_chars"], "omitted": step["observation"]["omitted"]}}))
            message_specs.append({"role": "assistant", "diagnosis_source": "authoritative metadata; supervision-only"})
            records.append({"example_id": "trajectory_03d_" + hashlib.sha256(f"{group}:{path_name}".encode()).hexdigest()[:20], "source_case_group": group, "source_system": case.source_system, "source_case_id": case.source_case_id, "source_fault_category": case.source_fault_category, "path": path_name, "split": split_by_case[group], "source_revision": case.source_revision, "source_fingerprint": case.source_fingerprint, "target_fingerprint": sha256_value(normalized["target"]), "target": normalized["target"], "target_model_visible": False, "message_specs": message_specs, "message_count": len(normalized["messages"]), "tool_step_count": len(normalized["replay_steps"]), "replay_unresolved_steps": sum(step["replay_status"] == "UNRESOLVED" for step in normalized["replay_steps"]), "replay_trace_fallback_steps": sum(step["replay_status"] == "RESOLVED_FROM_GOLDEN_TRACE" for step in normalized["replay_steps"]), "packaging_omission_steps": sum(step["observation"]["omitted"] for step in normalized["replay_steps"]), "observation_original_chars": sum(step["observation"]["original_chars"] for step in normalized["replay_steps"]), "observation_packaged_chars": sum(step["observation"]["packaged_chars"] for step in normalized["replay_steps"]), "observation_sizes": [{"original_chars": step["observation"]["original_chars"], "packaged_chars": step["observation"]["packaged_chars"], "omitted": step["observation"]["omitted"]} for step in normalized["replay_steps"]], "reasoning_disposition": normalized["reasoning_disposition"], "content_policy": "message content is reconstructed from pinned source at execution time; this manifest contains schema and provenance only"})
    return records


def _stats(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": 0, "median": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    ordered = sorted(values)
    def percentile(p: float) -> int:
        if len(ordered) == 1:
            return ordered[0]
        index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
        return ordered[index]
    return {"count": len(ordered), "min": ordered[0], "median": percentile(0.5), "p90": percentile(0.9), "p95": percentile(0.95), "p99": percentile(0.99), "max": ordered[-1]}


def build_packaging_audit(trajectory_records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    records = list(trajectory_records)
    observation_sizes = [size for item in records for size in item.get("observation_sizes", [])]
    original = [int(item["original_chars"]) for item in observation_sizes]
    packaged = [int(item["packaged_chars"]) for item in observation_sizes]
    return {"version": PACKAGING_VERSION, "trajectory_count": len(records), "observation_count": len(observation_sizes), "original_chars": _stats(original), "packaged_chars": _stats(packaged), "unresolved_replay_steps": sum(int(item["replay_unresolved_steps"]) for item in records), "golden_trace_fallback_steps": sum(int(item.get("replay_trace_fallback_steps", 0)) for item in records), "packaging_omission_steps": sum(int(item["packaging_omission_steps"]) for item in records), "full_raw_snapshot_primary": False, "policy": "deterministic bounded interaction packaging; no target-aware selection"}


def source_case_derivation_graph(scan: CorpusScan, split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    return {"version": "cloud-opsbench-source-derivation-v1", "edges": [{"source_case_group": _case_group(case), "source_fingerprint": case.source_fingerprint, "split": split_by_case[_case_group(case)], "descendants": [f"{_case_group(case)}:golden_path1", f"{_case_group(case)}:golden_path2", f"{_case_group(case)}:static_packaged_view", f"{_case_group(case)}:future_renderer_augmentation"], "inheritance": ["source_revision", "source_case_group", "split", "target_fingerprint"]} for case in sorted(scan.cases, key=_case_group)]}


def supervision_policy() -> dict[str, Any]:
    return {"version": SUPERVISION_POLICY_VERSION, "supervise": ["assistant tool/action calls", "assistant deterministic structured final diagnosis"], "do_not_supervise": ["user initial context", "tool observations", "source metadata", "provenance", "free-form reasoning/scratchpad"], "reasoning_fields": {"disposition": "DROP", "justification": "golden traces expose tool calls and observations; no unrestricted reasoning is needed for the primary contract"}, "tokenizer_masking": "deferred; message-level policy only"}


def freeze_contract(scan: CorpusScan, split_manifest: Mapping[str, Any], duplicate_report: Mapping[str, Any]) -> dict[str, Any]:
    source_fp = source_manifest_fingerprint(scan)
    components = {"source_revision": scan.source_revision, "source_manifest_fingerprint": source_fp, "split_manifest_fingerprint": split_manifest["fingerprint"], "target_contract": target_contract(), "leakage_policy": {"metadata.result.*": "SUPERVISION_ONLY", "paths/process-label/golden-trajectory": "PROVENANCE_OR_SUPERVISION_ONLY", "sanitized_query": "MODEL_VISIBLE_WITH_TRANSFORM", "raw_observations": "MODEL_VISIBLE_SAFE_AFTER_PACKAGING", "source_revision_and_fingerprint": "PROVENANCE_ONLY"}, "packaging_contract": packaging_contract(), "tool_contract": tool_contract(), "trajectory_normalization": {"version": NORMALIZATION_VERSION, "reasoning_disposition": "DROP"}, "supervision_policy": supervision_policy()}
    return {"version": FREEZE_VERSION, "experiment": "03D", "source_revision": scan.source_revision, "source_manifest_fingerprint": source_fp, "split_manifest_fingerprint": split_manifest["fingerprint"], "component_fingerprint": sha256_value(components), "components": components, "case_count": len(scan.cases), "test_policy": {"protected": True, "selection_or_tuning_allowed": False, "first_consumption": "final evaluation only"}, "model_loaded": False, "model_evaluated": False, "training_executed": False, "provider_called": False}
