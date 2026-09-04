"""03D.1 replay-parity and cumulative-context integrity audits.

All functions are deterministic and provider/model free.  The packager uses
only source observations; process labels are consulted only by the separate
evidence-retention auditor.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from causetune.incident_telemetry.fingerprints import canonical_json

from .models import CloudOpsBenchCase
from .scanner import CorpusScan
from .training_contract import (
    MAX_OBSERVATION_CHARS,
    _argument_pattern,
    _cache_index,
    _resolve_cache_key,
    _stats,
    _trace,
    _uniform_line_package,
    _calling_arguments,
    source_case_group_id,
    sha256_value,
)


REPRESENTATION_VERSION = "cloud-opsbench-03d1-representation-v1"
FALLBACK_REASONS = (
    "NORMALIZATION_MISMATCH",
    "ARGUMENT_CANONICALIZATION_MISMATCH",
    "TOOL_ALIAS_MISMATCH",
    "TOOL_VERSION_MISMATCH",
    "CACHE_SCHEMA_MISMATCH",
    "SOURCE_ARTIFACT_DERIVABLE",
    "GOLDEN_ONLY_NOT_REPLAYABLE",
    "MALFORMED_OR_UNKNOWN",
)
PARITY_CLASSES = (
    "EXACT_REPLAY",
    "CANONICALLY_EQUIVALENT_REPLAY",
    "SAFE_SOURCE_DERIVED_REPLAY",
    "GOLDEN_ONLY_FALLBACK",
    "UNRESOLVED",
)


def _parse_value(value: str) -> Any:
    candidates: list[Any] = [value]
    try:
        decoded = json.loads(value)
        candidates.insert(0, decoded)
        if isinstance(decoded, str):
            try:
                candidates.insert(0, json.loads(decoded))
            except json.JSONDecodeError:
                pass
    except json.JSONDecodeError:
        pass
    try:
        candidates.insert(0, ast.literal_eval(value))
    except (SyntaxError, ValueError):
        pass
    return candidates[0]


def _equivalent(left: Any, right: Any) -> bool:
    return canonical_json(left) == canonical_json(right)


def _candidate_cache_equivalence(cache: Mapping[str, str], tool_name: str, output: str) -> tuple[str | None, str | None]:
    parsed_output = _parse_value(output)
    candidates = sorted(key for key in cache if key == tool_name or key.startswith(tool_name + ":") or key.startswith(tool_name + "-"))
    for key in candidates:
        parsed_cache = _parse_value(cache[key])
        if _equivalent(parsed_output, parsed_cache):
            return key, sha256_value(parsed_cache)
    return None, None


def _source_derived_observation(root: Path, case: CloudOpsBenchCase, tool_name: str, calling: str, output: str) -> tuple[Any | None, str | None]:
    arguments = _calling_arguments(calling)
    parsed_output = _parse_value(output)
    if tool_name == "GetRecentLogs" and case.references.logs is not None:
        logs = json.loads((root / case.references.logs.relative_path).read_text(encoding="utf-8"))
        service = arguments.get("service_name")
        lines = arguments.get("lines")
        if isinstance(logs, Mapping) and isinstance(service, str) and isinstance(lines, int) and isinstance(logs.get(service), list):
            candidates = (logs[service][:lines], logs[service][-lines:])
            for candidate in candidates:
                if _equivalent(parsed_output, candidate):
                    return candidate, "logs.json deterministic service/line selection"
    if tool_name == "GetSourceCode" and case.references.code is not None:
        file_path = arguments.get("file_path")
        if isinstance(file_path, str):
            base = root / case.references.code.relative_path
            paths = [base / file_path]
            if not paths[0].is_file():
                paths = sorted(path for path in base.rglob("*") if path.is_file() and path.name == Path(file_path).name)
            for path in paths:
                if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == output:
                    return output, "code/ deterministic file selection"
    return None, None


def classify_fallbacks(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any]) -> dict[str, Any]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    records: list[dict[str, Any]] = []
    for case in sorted(scan.cases, key=source_case_group_id):
        group = source_case_group_id(case)
        cache = _cache_index(root, case)
        for path_name in ("golden_path1", "golden_path2"):
            for step_index, item in enumerate(_trace(root, case, path_name)):
                tool_name = str(item.get("tool_name", ""))
                output = str(item.get("output", ""))
                calling = str(item.get("calling", ""))
                key, _ = _resolve_cache_key(cache, tool_name, output, calling)
                if key:
                    continue
                if tool_name not in {key.split(":", 1)[0] for key in cache} and tool_name not in {"GetRecentLogs", "GetSourceCode"}:
                    reason = "MALFORMED_OR_UNKNOWN"
                    expected = None
                    source_detail = None
                else:
                    equivalent_key, equivalent_fp = _candidate_cache_equivalence(cache, tool_name, output)
                    if equivalent_key:
                        reason = "NORMALIZATION_MISMATCH"
                        expected = _parse_value(cache[equivalent_key])
                        source_detail = f"cache:{equivalent_key}"
                    else:
                        expected, source_detail = _source_derived_observation(root, case, tool_name, calling, output)
                        reason = "SOURCE_ARTIFACT_DERIVABLE" if source_detail else "GOLDEN_ONLY_NOT_REPLAYABLE"
                records.append({
                    "source_case_group": group,
                    "source_system": case.source_system,
                    "source_case_id": case.source_case_id,
                    "source_fault_category": case.source_fault_category,
                    "source_fault_type": case.source_fault_type,
                    "upstream_difficulty": case.upstream_difficulty,
                    "split": split_by_case[group],
                    "path": path_name,
                    "step": step_index,
                    "tool_id": tool_name,
                    "normalized_arguments": _argument_pattern(calling),
                    "expected_observation_fingerprint": sha256_value(expected) if expected is not None else None,
                    "fallback_observation_fingerprint": sha256_value(output),
                    "classification": reason,
                    "source_equivalent": expected is not None,
                    "source_detail": source_detail,
                    "deterministically_reproducible_at_evaluation": reason in {"NORMALIZATION_MISMATCH", "SOURCE_ARTIFACT_DERIVABLE"},
                    "safe_for_primary_sft": reason in {"NORMALIZATION_MISMATCH", "SOURCE_ARTIFACT_DERIVABLE"},
                })
    records.sort(key=lambda item: (item["source_case_group"], item["path"], item["step"]))
    dimensions = {
        "reason": Counter(item["classification"] for item in records),
        "tool": Counter(item["tool_id"] for item in records),
        "fault_category": Counter(item["source_fault_category"] for item in records),
        "system": Counter(item["source_system"] for item in records),
        "split": Counter(item["split"] for item in records),
    }
    return {"version": "cloud-opsbench-03d1-fallback-classification-v1", "original_fallback_count": len(records), "taxonomy": list(FALLBACK_REASONS), "records": records, "counts": {name: dict(sorted(counter.items())) for name, counter in dimensions.items()}, "replay_safety": {"safe_for_primary_sft": sum(item["safe_for_primary_sft"] for item in records), "golden_only_or_unresolved": sum(not item["safe_for_primary_sft"] for item in records)}}


def replay_parity(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any], fallback_report: Mapping[str, Any]) -> dict[str, Any]:
    fallback_by_step = {(item["source_case_group"], item["path"], item["step"]): item for item in fallback_report["records"]}
    counts = Counter()
    by_tool: dict[str, Counter[str]] = defaultdict(Counter)
    for case in scan.cases:
        group = source_case_group_id(case)
        cache = _cache_index(root, case)
        for path_name in ("golden_path1", "golden_path2"):
            for step, item in enumerate(_trace(root, case, path_name)):
                tool = str(item.get("tool_name", ""))
                key, _ = _resolve_cache_key(cache, tool, str(item.get("output", "")), str(item.get("calling", "")))
                if key:
                    parity = "EXACT_REPLAY"
                else:
                    reason = fallback_by_step[(group, path_name, step)]["classification"]
                    parity = {"NORMALIZATION_MISMATCH": "CANONICALLY_EQUIVALENT_REPLAY", "SOURCE_ARTIFACT_DERIVABLE": "SAFE_SOURCE_DERIVED_REPLAY", "GOLDEN_ONLY_NOT_REPLAYABLE": "GOLDEN_ONLY_FALLBACK"}.get(reason, "UNRESOLVED")
                counts[parity] += 1
                by_tool[tool][parity] += 1
    return {"version": REPRESENTATION_VERSION, "trajectory_observation_count": sum(counts.values()), "counts": {name: counts[name] for name in PARITY_CLASSES}, "by_tool": {tool: dict(sorted(values.items())) for tool, values in sorted(by_tool.items())}, "primary_executable_observation_count": counts["EXACT_REPLAY"] + counts["CANONICALLY_EQUIVALENT_REPLAY"] + counts["SAFE_SOURCE_DERIVED_REPLAY"], "golden_only_observation_count": counts["GOLDEN_ONLY_FALLBACK"], "unresolved_observation_count": counts["UNRESOLVED"]}


def _process_patterns(root: Path, case: CloudOpsBenchCase) -> dict[str, list[tuple[str, str]]]:
    if case.references.process_label is None:
        return {}
    value = json.loads((root / case.references.process_label.relative_path).read_text(encoding="utf-8"))
    result: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for milestone in value.get("milestones", []) if isinstance(value, Mapping) else []:
        for admissible in milestone.get("admissible_tool_uses", []) if isinstance(milestone, Mapping) else []:
            tool = admissible.get("tool_name") if isinstance(admissible, Mapping) else None
            for pattern in admissible.get("evidence_patterns", []) if isinstance(admissible, Mapping) else []:
                if isinstance(tool, str) and isinstance(pattern, Mapping) and isinstance(pattern.get("value"), str):
                    result[tool].append((str(pattern.get("kind", "unknown")), pattern["value"]))
    return result


def _pattern_matches(pattern: tuple[str, str], text: str) -> bool:
    kind, value = pattern
    if kind == "regex":
        try:
            return re.search(value, text) is not None
        except re.error:
            return False
    return value in text


def packaging_omission_audit(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any], *, max_chars: int = MAX_OBSERVATION_CHARS) -> dict[str, Any]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    records: list[dict[str, Any]] = []
    raw_required = retained_required = 0
    for case in sorted(scan.cases, key=source_case_group_id):
        patterns = _process_patterns(root, case)
        for path_name in ("golden_path1", "golden_path2"):
            for step_index, item in enumerate(_trace(root, case, path_name)):
                tool = str(item.get("tool_name", ""))
                raw = str(item.get("output", ""))
                package = _uniform_line_package(raw, max_chars)
                matched = [pattern for pattern in patterns.get(tool, []) if _pattern_matches(pattern, raw)]
                retained = [pattern for pattern in matched if _pattern_matches(pattern, package["text"])]
                raw_required += len(matched)
                retained_required += len(retained)
                if not package["omitted"]:
                    continue
                records.append({"source_case_group": source_case_group_id(case), "source_system": case.source_system, "source_case_id": case.source_case_id, "source_fault_category": case.source_fault_category, "source_fault_type": case.source_fault_type, "split": split_by_case[source_case_group_id(case)], "path": path_name, "step": step_index, "tool_id": tool, "original_chars": package["original_chars"], "packaged_chars": package["packaged_chars"], "omission_reason": package.get("selection", "deterministic bound exceeded"), "omitted_structural_fields": "middle lines outside deterministic sample; exact fields are not inferred", "required_pattern_matches_raw": len(matched), "required_pattern_matches_retained": len(retained), "required_evidence_lost": len(matched) - len(retained), "packager_target_blind": True})
    return {"version": "cloud-opsbench-03d1-packaging-omission-audit-v1", "budget_chars": max_chars, "omission_count": len(records), "records": records, "counts_by": {dimension: dict(sorted(Counter(item[dimension] for item in records).items())) for dimension in ("tool_id", "source_system", "source_fault_category", "split")}, "required_pattern_matches_raw": raw_required, "required_pattern_matches_retained": retained_required, "required_pattern_retention_rate": retained_required / raw_required if raw_required else 1.0}


def evidence_retention_summary(omission_audit: Mapping[str, Any]) -> dict[str, Any]:
    by_dimension: dict[str, dict[str, dict[str, int]]] = {}
    for dimension in ("split", "source_system", "source_fault_category", "source_fault_type"):
        values: dict[str, dict[str, int]] = defaultdict(lambda: {"omissions": 0, "required_raw": 0, "required_retained": 0, "required_lost": 0})
        for item in omission_audit["records"]:
            bucket = str(item[dimension])
            values[bucket]["omissions"] += 1
            values[bucket]["required_raw"] += int(item["required_pattern_matches_raw"])
            values[bucket]["required_retained"] += int(item["required_pattern_matches_retained"])
            values[bucket]["required_lost"] += int(item["required_evidence_lost"])
        by_dimension[dimension] = dict(sorted(values.items()))
    return {"version": "cloud-opsbench-03d1-evidence-retention-v1", "budget_chars": omission_audit["budget_chars"], "all_observations": {"required_raw": omission_audit["required_pattern_matches_raw"], "required_retained": omission_audit["required_pattern_matches_retained"], "required_lost": omission_audit["required_pattern_matches_raw"] - omission_audit["required_pattern_matches_retained"], "retention_rate": omission_audit["required_pattern_retention_rate"]}, "by_dimension": by_dimension, "interpretation": "process-label evidence patterns are used only by this offline auditor, never by the target-blind packager"}


def _context_text(parts: list[dict[str, Any]]) -> tuple[int, int]:
    text = json.dumps(parts, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return len(text), len(text.encode("utf-8"))


def _context_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        def extended(values: list[int]) -> dict[str, Any]:
            result = _stats(values)
            ordered = sorted(values)
            if ordered:
                result["p75"] = ordered[min(len(ordered) - 1, round((len(ordered) - 1) * 0.75))]
            else:
                result["p75"] = 0
            return result
        return {"count": len(rows), "characters": extended([row["characters"] for row in rows]), "bytes": extended([row["bytes"] for row in rows])}
    groups: dict[str, dict[str, Any]] = {}
    for field in ("split", "source_system", "upstream_difficulty", "source_fault_category", "tool_steps"):
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in records:
            buckets[str(row[field])].append(row)
        groups[field] = {key: summarize(value) for key, value in sorted(buckets.items())}
    return {"all": summarize(records), "by": groups}


def cumulative_context_audit(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any], *, max_chars: int = MAX_OBSERVATION_CHARS) -> dict[str, Any]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    action_rows: list[dict[str, Any]] = []
    final_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    compact_final_rows: list[dict[str, Any]] = []
    summary_final_rows: list[dict[str, Any]] = []
    for case in sorted(scan.cases, key=source_case_group_id):
        group = source_case_group_id(case)
        split = split_by_case[group]
        for path_name in ("golden_path1", "golden_path2"):
            trace = _trace(root, case, path_name)
            history = [{"role": "user", "content": str(case.ground_truth_metadata.get("query", "Investigate the reported service condition."))}]
            compact_history = [{"role": "user", "content": "sanitized incident request"}]
            summary_history = [{"role": "user", "content": "sanitized incident request"}]
            for step_index, item in enumerate(trace):
                tool = str(item.get("tool_name", ""))
                calling = str(item.get("calling", ""))
                package = _uniform_line_package(str(item.get("output", "")), max_chars)
                action = {"role": "assistant", "tool_call": {"tool_id": tool, "arguments": _argument_pattern(calling)}}
                chars, bytes_count = _context_text(history + [action])
                metadata = {"split": split, "source_system": case.source_system, "upstream_difficulty": case.upstream_difficulty or "UNSPECIFIED", "source_fault_category": case.source_fault_category, "tool_steps": len(trace)}
                action_rows.append({**metadata, "characters": chars, "bytes": bytes_count, "step": step_index, "tool_id": tool})
                history.extend((action, {"role": "tool", "tool_id": tool, "observation": package["text"]}))
                compact_history.extend((action, {"role": "tool", "tool_id": tool, "observation_ref": sha256_value(package["text"]), "packaged_chars": package["packaged_chars"], "omitted": package["omitted"]}))
                summary_history.extend(({"role": "assistant", "tool_id": tool, "arguments": _argument_pattern(calling)}, {"role": "state", "last_tool": tool, "observation_chars": package["packaged_chars"], "omitted": package["omitted"], "observation_count": step_index + 1}))
            final_schema = {"role": "assistant", "final_diagnosis_schema": ["native_fault_type", "native_fault_category", "fault_object"]}
            metadata = {"split": split, "source_system": case.source_system, "upstream_difficulty": case.upstream_difficulty or "UNSPECIFIED", "source_fault_category": case.source_fault_category, "tool_steps": len(trace)}
            chars, bytes_count = _context_text(history + [final_schema])
            final_rows.append({**metadata, "characters": chars, "bytes": bytes_count})
            compact_chars, compact_bytes = _context_text(compact_history + [final_schema])
            compact_final_rows.append({**metadata, "characters": compact_chars, "bytes": compact_bytes})
            summary_chars, summary_bytes = _context_text(summary_history + [final_schema])
            summary_final_rows.append({**metadata, "characters": summary_chars, "bytes": summary_bytes})
            trajectory_rows.append({**metadata, "characters": chars, "bytes": bytes_count})
    def threshold_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
        return {str(threshold): {"character_budget": threshold * 4, "exceeding_by_bytes": sum(row["bytes"] > threshold * 4 for row in rows), "exceeding_by_characters": sum(row["characters"] > threshold * 4 for row in rows)} for threshold in (4096, 8192, 16384, 32768, 65536)}
    return {"version": "cloud-opsbench-03d1-cumulative-context-v1", "observation_budget_chars": max_chars, "per_assistant_action": _context_stats(action_rows), "per_final_diagnosis_action": _context_stats(final_rows), "per_complete_trajectory": _context_stats(trajectory_rows), "threshold_counts": {"per_assistant_action": threshold_counts(action_rows), "per_final_diagnosis_action": threshold_counts(final_rows), "per_complete_trajectory": threshold_counts(trajectory_rows)}, "history_policy_comparison": {"FULL_HISTORY": _context_stats(final_rows), "STRUCTURED_COMPACT_HISTORY": _context_stats(compact_final_rows), "OBSERVATION_SUMMARY_STATE": _context_stats(summary_final_rows)}, "construction": "serialized initial context plus prior action/observation messages plus current action framing; no target values or process labels"}


def token_budget_projection(context_audit: Mapping[str, Any]) -> dict[str, Any]:
    def project(section: Mapping[str, Any], key: str) -> dict[str, Any]:
        chars = section["all"]["characters"]
        values = {}
        for threshold, counts in context_audit["threshold_counts"][key].items():
            values[threshold] = {"token_estimate": int(threshold), **counts}
        return {"distribution": chars, "thresholds": values}
    return {"version": "cloud-opsbench-03d1-token-projection-v1", "method": "deterministic bytes/4 token estimate; not a tokenizer measurement", "per_assistant_action": project(context_audit["per_assistant_action"], "per_assistant_action"), "per_final_diagnosis_action": project(context_audit["per_final_diagnosis_action"], "per_final_diagnosis_action"), "per_complete_trajectory": project(context_audit["per_complete_trajectory"], "per_complete_trajectory"), "note": "estimates only; no model or tokenizer is loaded"}


def tool_budget_analysis(root: Path, scan: CorpusScan, split_manifest: Mapping[str, Any], budgets: Iterable[int] = (8192, 16384, 24576, 32768)) -> dict[str, Any]:
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    results = {}
    for budget in budgets:
        affected_observations = 0
        affected_trajectories: set[tuple[str, str]] = set()
        by_category: Counter[str] = Counter()
        raw_required = retained_required = 0
        for case in scan.cases:
            patterns = _process_patterns(root, case)
            group = source_case_group_id(case)
            for path_name in ("golden_path1", "golden_path2"):
                for item in _trace(root, case, path_name):
                    tool = str(item.get("tool_name", "")); raw = str(item.get("output", "")); package = _uniform_line_package(raw, budget)
                    matched = [pattern for pattern in patterns.get(tool, []) if _pattern_matches(pattern, raw)]
                    retained = [pattern for pattern in matched if _pattern_matches(pattern, package["text"])]
                    raw_required += len(matched); retained_required += len(retained)
                    if package["omitted"]:
                        affected_observations += 1; affected_trajectories.add((group, path_name)); by_category[case.source_fault_category] += 1
        results[str(budget)] = {"budget_chars": budget, "affected_observations": affected_observations, "affected_trajectories": len(affected_trajectories), "affected_source_cases": len({case for case, _ in affected_trajectories}), "affected_by_fault_category": dict(sorted(by_category.items())), "required_pattern_matches_raw": raw_required, "required_pattern_matches_retained": retained_required, "required_evidence_lost": raw_required - retained_required, "required_evidence_retention_rate": retained_required / raw_required if raw_required else 1.0}
    return {"version": "cloud-opsbench-03d1-tool-budget-analysis-v1", "target_blind_packager": True, "budgets": results, "selection_basis": "integrity/evidence retention and deployability only; no model accuracy"}


def representation_manifest(scan: CorpusScan, split_manifest: Mapping[str, Any], fallback_report: Mapping[str, Any], *, frozen_split_fingerprint: str) -> dict[str, Any]:
    if split_manifest["fingerprint"] != frozen_split_fingerprint:
        raise ValueError("03D split fingerprint changed during 03D.1")
    by_case_path: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in fallback_report["records"]:
        by_case_path[(item["source_case_group"], item["path"])].append(item)
    split_by_case = {item["source_case_group"]: item["split"] for item in split_manifest["entries"]}
    eligible = []
    auxiliary = []
    for case in scan.cases:
        group = source_case_group_id(case)
        for path_name in ("golden_path1", "golden_path2"):
            row = {"source_case_group": group, "path": path_name, "split": split_by_case[group], "golden_only_fallback_steps": sum(item["classification"] == "GOLDEN_ONLY_NOT_REPLAYABLE" for item in by_case_path.get((group, path_name), [])), "source_fingerprint": case.source_fingerprint}
            row["primary_executable"] = row["split"] in {"TRAIN", "VALIDATION"} and row["golden_only_fallback_steps"] == 0
            (eligible if row["primary_executable"] else auxiliary).append(row)
    policy = {"version": REPRESENTATION_VERSION, "split_fingerprint": split_manifest["fingerprint"], "primary_input": "bounded hybrid tool interaction", "full_raw_snapshot_primary": False, "primary_sft_included_splits": ["TRAIN", "VALIDATION"], "primary_sft_excludes_golden_only_trajectories": True, "golden_only_policy": "auxiliary_ablation_only; do not silently retain in executable primary SFT", "test_model_facing": False, "eligible_trajectory_count": len(eligible), "auxiliary_trajectory_count": len(auxiliary), "eligible_source_case_count": len({item["source_case_group"] for item in eligible}), "auxiliary_by_split": dict(sorted(Counter(item["split"] for item in auxiliary).items())), "eligible_records": sorted(eligible, key=lambda item: (item["source_case_group"], item["path"])), "auxiliary_records": sorted(auxiliary, key=lambda item: (item["source_case_group"], item["path"]))}
    policy["representation_fingerprint"] = sha256_value(policy)
    return policy
