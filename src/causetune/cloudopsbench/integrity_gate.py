"""CPU-only evidence integrity audits for Experiment 03F.1.

The packager functions in this module deliberately have no access to labels or
process annotations.  Process labels are consumed only by the audit helpers.
No model, tokenizer, provider, or GPU dependency is used here.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from causetune.incident_telemetry.fingerprints import canonical_json

from .decomposition import PACKAGE_BUDGET_CHARS, _match_pattern, _stats
from .models import CloudOpsBenchCase
from .training_contract import _sanitized_query, _uniform_line_package, normalize_trajectory, source_case_group_id


INTEGRITY_VERSION = "cloud-opsbench-03f1-evidence-integrity-v1"
REPRESENTATION_CANDIDATES = ("R1_EXISTING_03F", "R2_DETERMINISTIC_PATH_UNION", "R3_UNION_BASELINE_OBSERVABILITY")
BUDGETS = (15000, 20000, 30000, 40000)
BASELINE_CHANNELS = (
    ("BaselineAlertSummary", "alerts", 4000),
    ("BaselineKubernetesState", "k8s_states", 10000),
    ("BaselineRecentLogs", "logs", 8000),
    ("BaselineMetrics", "metrics", 4000),
    ("BaselineCodeContext", "code", 6000),
)


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def canonicalize_text(value: str) -> str:
    """Canonicalize only Unicode/case/whitespace representation."""

    value = unicodedata.normalize("NFKC", str(value)).replace("\x1b[", "")
    return re.sub(r"\s+", " ", value).strip().casefold()


def _repair_impossible_word_boundary(pattern: str) -> str | None:
    """Return a diagnostic-only repair for a regex boundary after punctuation.

    ``<none>\b`` cannot match because both adjacent characters are non-word.
    This is classified as a matcher/schema defect, not silently used as a
    scoring improvement.
    """

    repaired = re.sub(r"(?<=[^\w\s])\\b", "", pattern)
    return repaired if repaired != pattern else None


def matcher_result(kind: str, value: Any, text: str) -> dict[str, Any]:
    """Report strict and mechanically canonical-equivalent matcher outcomes."""

    if not isinstance(value, str) or not isinstance(text, str):
        return {"strict": False, "canonical": False, "matcher_schema_gap": False, "invalid": True}
    if kind == "regex":
        try:
            strict = re.search(value, text, re.DOTALL | re.MULTILINE) is not None
        except re.error:
            strict = False
        repaired_pattern = _repair_impossible_word_boundary(value)
        repaired = False
        if repaired_pattern is not None:
            try:
                repaired = re.search(repaired_pattern, text, re.DOTALL | re.MULTILINE) is not None
            except re.error:
                repaired = False
        return {"strict": strict, "canonical": strict, "matcher_schema_gap": repaired and not strict, "invalid": False}
    canonical = canonicalize_text(value) in canonicalize_text(text)
    strict = value.casefold() in text.casefold()
    return {"strict": strict, "canonical": canonical, "matcher_schema_gap": False, "invalid": False}


def _load_json(root: Path, case: CloudOpsBenchCase, name: str) -> Any:
    reference = getattr(case.references, name)
    if reference is None:
        return None
    return json.loads((root / reference.relative_path).read_text(encoding="utf-8-sig"))


def process_patterns(root: Path, case: CloudOpsBenchCase) -> list[dict[str, Any]]:
    """Load process patterns for auditing; never called by packagers."""

    payload = _load_json(root, case, "process_label") or {}
    rows: list[dict[str, Any]] = []
    for milestone in payload.get("milestones", []) if isinstance(payload, Mapping) else []:
        for admissible in milestone.get("admissible_tool_uses", []) if isinstance(milestone, Mapping) else []:
            tool = admissible.get("tool_name") if isinstance(admissible, Mapping) else None
            for pattern in admissible.get("evidence_patterns", []) if isinstance(admissible, Mapping) else []:
                if isinstance(tool, str) and isinstance(pattern, Mapping) and isinstance(pattern.get("value"), str):
                    rows.append({"milestone": milestone.get("id"), "tool": tool, "kind": str(pattern.get("kind", "literal")), "value": pattern["value"]})
    return rows


def executable_observations(root: Path, case: CloudOpsBenchCase, path: str) -> list[dict[str, str]]:
    normalized = normalize_trajectory(root, case, path)
    return [{"tool": str(step["tool_id"]), "observation": str(step["observation"]["text"])} for step in normalized["replay_steps"] if step["replay_status"] == "RESOLVED_FROM_TOOL_CACHE"]


def all_trace_observations(root: Path, case: CloudOpsBenchCase, path: str) -> list[dict[str, str]]:
    normalized = normalize_trajectory(root, case, path)
    return [{"tool": str(step["tool_id"]), "observation": str(step["observation"]["text"])} for step in normalized["replay_steps"]]


def deduplicate_observations(observations: Iterable[Mapping[str, str]]) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in observations:
        value = (str(item.get("tool", "")), str(item.get("observation", "")))
        if value not in seen:
            seen.add(value)
            result.append({"tool": value[0], "observation": value[1]})
    return result


def _render(query: str, evidence: Sequence[Mapping[str, str]]) -> str:
    return json.dumps({"incident_request": query, "evidence": list(evidence)}, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _budget_evidence(query: str, evidence: Sequence[Mapping[str, str]], budget: int) -> tuple[list[dict[str, str]], str]:
    original = [{"tool": str(item["tool"]), "observation": str(item["observation"])} for item in evidence]
    if not original or len(_render(query, original)) <= budget:
        return original, _render(query, original)
    low, high = 0, max(len(item["observation"]) for item in original)
    while low < high:
        cap = (low + high + 1) // 2
        trial = [{"tool": item["tool"], "observation": _uniform_line_package(item["observation"], cap)["text"]} for item in original]
        if len(_render(query, trial)) <= budget:
            low = cap
        else:
            high = cap - 1
    bounded = [{"tool": item["tool"], "observation": _uniform_line_package(item["observation"], low)["text"]} for item in original]
    return bounded, _render(query, bounded)


def _code_text(root: Path, case: CloudOpsBenchCase) -> str:
    reference = case.references.code
    if reference is None:
        return ""
    base = root / reference.relative_path
    parts: list[str] = []
    for path in sorted(path for path in base.rglob("*") if path.is_file()):
        parts.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(parts)


def baseline_observations(root: Path, case: CloudOpsBenchCase) -> list[dict[str, str]]:
    """Add the same source-modality policy to every eligible case.

    Selection is based solely on fixed modality availability, never labels,
    process labels, paths, trajectories, or target-dependent content.
    """

    result: list[dict[str, str]] = []
    for tool, modality, cap in BASELINE_CHANNELS:
        if modality == "code":
            text = _code_text(root, case)
        elif modality == "metrics":
            reference = case.references.metrics
            text = (root / reference.relative_path).read_text(encoding="utf-8-sig", errors="replace") if reference is not None else ""
        else:
            value = _load_json(root, case, modality)
            if value is None:
                text = ""
            elif modality == "metrics":
                text = str(value)
            else:
                text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)
        if text:
            result.append({"tool": tool, "observation": _uniform_line_package(text, cap)["text"]})
    return result


def candidate_evidence(root: Path, case: CloudOpsBenchCase, candidate: str) -> list[dict[str, str]]:
    path1 = executable_observations(root, case, "golden_path1")
    path2 = executable_observations(root, case, "golden_path2")
    if candidate == "R1_EXISTING_03F":
        return path1 if path1 else path2
    union = deduplicate_observations([*path1, *path2])
    if candidate == "R2_DETERMINISTIC_PATH_UNION":
        return union
    if candidate == "R3_UNION_BASELINE_OBSERVABILITY":
        return [*union, *baseline_observations(root, case)]
    raise ValueError(f"unknown representation candidate: {candidate}")


def build_candidate_package(root: Path, case: CloudOpsBenchCase, candidate: str, budget: int = PACKAGE_BUDGET_CHARS) -> dict[str, Any]:
    query = _sanitized_query(case)
    evidence, rendered = _budget_evidence(query, candidate_evidence(root, case, candidate), budget)
    return {
        "model_input": {"incident_request": query, "evidence": evidence},
        "candidate": candidate,
        "budget_chars": budget,
        "package_chars": len(rendered),
        "package_bytes": len(rendered.encode("utf-8")),
        "original_evidence_steps": len(candidate_evidence(root, case, candidate)),
        "retained_evidence_steps": len(evidence),
        "observations_truncated": len(rendered) < len(_render(query, candidate_evidence(root, case, candidate))),
        "target_blind": True,
        "process_labels_in_model_input": False,
        "golden_answers_in_model_input": False,
        "source_paths_in_model_input": False,
        "provenance_in_model_input": False,
    }


def pattern_coverage_rows(patterns: Sequence[Mapping[str, Any]], evidence: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pattern in patterns:
        candidates = [item for item in evidence if str(item.get("tool")) == str(pattern["tool"])]
        outcomes = [matcher_result(str(pattern["kind"]), pattern["value"], str(item["observation"])) for item in candidates]
        rows.append({**dict(pattern), "strict_match": any(item["strict"] for item in outcomes), "canonical_match": any(item["canonical"] for item in outcomes), "matcher_schema_gap": any(item["matcher_schema_gap"] for item in outcomes), "candidate_observation_count": len(candidates)})
    return rows


def _stats_float(values: Sequence[float]) -> dict[str, Any]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "min": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    def at(percentile: float) -> float:
        return ordered[round((len(ordered) - 1) * percentile)]
    return {"count": len(ordered), "min": ordered[0], "median": at(.5), "p75": at(.75), "p90": at(.9), "p95": at(.95), "p99": at(.99), "max": ordered[-1]}


def classify_case_reason(patterns: Sequence[Mapping[str, Any]], pattern_rows: Sequence[Mapping[str, Any]], *, path1_rows: Sequence[Mapping[str, Any]], path2_rows: Sequence[Mapping[str, Any]], union_rows: Sequence[Mapping[str, Any]], source_rows: Sequence[Mapping[str, Any]]) -> str:
    if not patterns:
        return "PROCESS_LABEL_NO_MATCHABLE_PATTERNS"
    missing = [index for index, row in enumerate(pattern_rows) if not row.get("strict_match")]
    if any(source_rows[index].get("matcher_schema_gap") for index in missing):
        return "MATCHER_SCHEMA_GAP"
    if any(source_rows[index].get("strict_match") for index in missing):
        if any(path2_rows[index].get("strict_match") and not path1_rows[index].get("strict_match") for index in missing):
            return "PROCESS_EVIDENCE_ON_ALTERNATE_PATH"
        if any(path1_rows[index].get("strict_match") for index in missing):
            return "PACKAGING_OR_BUDGET_LOSS"
        if any(union_rows[index].get("strict_match") for index in missing):
            return "SELECTED_PATH_MISSES_PROCESS_EVIDENCE"
        if any(source_rows[index].get("direct_source_match") and not source_rows[index].get("trace_match") for index in missing):
            return "SOURCE_ARTIFACT_AVAILABLE_BUT_NOT_PACKAGED"
        return "SOURCE_ARTIFACT_AVAILABLE_BUT_NOT_PACKAGED"
    if any(row.get("strict_match") for row in source_rows):
        return "PACKAGING_OR_BUDGET_LOSS"
    return "PROCESS_PATTERN_UNMATCHABLE_OR_EVIDENCE_SPARSE"


def _source_text(root: Path, case: CloudOpsBenchCase) -> str:
    values: list[str] = []
    for name in ("alerts", "k8s_states", "logs", "metrics"):
        ref = getattr(case.references, name)
        if ref is not None:
            values.append((root / ref.relative_path).read_text(encoding="utf-8", errors="replace"))
    values.append(_code_text(root, case))
    return "\n".join(values)


def _pattern_rows_with_source(root: Path, case: CloudOpsBenchCase, evidence: Sequence[Mapping[str, str]], path1: Sequence[Mapping[str, str]], path2: Sequence[Mapping[str, str]], union: Sequence[Mapping[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    patterns = process_patterns(root, case)
    rows = pattern_coverage_rows(patterns, evidence)
    p1 = pattern_coverage_rows(patterns, path1)
    p2 = pattern_coverage_rows(patterns, path2)
    ur = pattern_coverage_rows(patterns, union)
    source = []
    source_text = _source_text(root, case)
    for row in patterns:
        outcome = matcher_result(str(row["kind"]), row["value"], source_text)
        source.append({**dict(row), "strict_match": outcome["strict"], "canonical_match": outcome["canonical"], "matcher_schema_gap": outcome["matcher_schema_gap"]})
    return rows, p1, p2, ur, source


def path_coverage_record(root: Path, case: CloudOpsBenchCase, candidate: str = "R2_DETERMINISTIC_PATH_UNION", budget: int = PACKAGE_BUDGET_CHARS) -> dict[str, Any]:
    path1 = executable_observations(root, case, "golden_path1")
    path2 = executable_observations(root, case, "golden_path2")
    union = deduplicate_observations([*path1, *path2])
    package = build_candidate_package(root, case, candidate, budget)
    patterns = process_patterns(root, case)
    def count(items: Sequence[Mapping[str, str]]) -> int:
        return sum(row["strict_match"] for row in pattern_coverage_rows(patterns, items))
    return {"path1": {"observations": len(path1), "patterns": len(patterns), "retained_patterns": count(path1)}, "path2": {"observations": len(path2), "patterns": len(patterns), "retained_patterns": count(path2)}, "union": {"observations": len(union), "patterns": len(patterns), "retained_patterns": count(union), "deduplicated_observations": len(union), "additional_observations_over_path1": max(0, len(union) - len(path1))}, "selected_package": {"candidate": candidate, "retained_patterns": count(package["model_input"]["evidence"]), "package_chars": package["package_chars"], "package_bytes": package["package_bytes"]}}


def coverage_class(total: int, retained: int) -> str:
    return "ZERO" if retained == 0 else "FULL" if total and retained == total else "PARTIAL"


def summarize_coverage(rows: Sequence[Mapping[str, Any]], key: str = "retained_patterns") -> dict[str, Any]:
    counts = Counter(coverage_class(int(row["patterns"]), int(row[key])) for row in rows)
    total = sum(int(row["patterns"]) for row in rows)
    retained = sum(int(row[key]) for row in rows)
    return {"cases": len(rows), "FULL": counts["FULL"], "PARTIAL": counts["PARTIAL"], "ZERO": counts["ZERO"], "total_patterns": total, "retained_patterns": retained, "missing_patterns": total - retained, "retention_rate": retained / total if total else None}


def _case_modalities(case: CloudOpsBenchCase) -> list[str]:
    return [name for name in ("k8s_states", "logs", "alerts", "metrics", "code") if getattr(case.references, name) is not None]


def build_case_audit(root: Path, case: CloudOpsBenchCase, split: str, *, screen: bool = False, budget: int = PACKAGE_BUDGET_CHARS) -> dict[str, Any]:
    path1 = executable_observations(root, case, "golden_path1")
    path2 = executable_observations(root, case, "golden_path2")
    union = deduplicate_observations([*path1, *path2])
    package = build_candidate_package(root, case, "R2_DETERMINISTIC_PATH_UNION", budget)
    package_rows, p1_rows, p2_rows, union_rows, source_rows = _pattern_rows_with_source(root, case, package["model_input"]["evidence"], path1, path2, union)
    patterns = process_patterns(root, case)
    reason = classify_case_reason(patterns, package_rows, p1_rows, p2_rows, union_rows, source_rows)
    retained = sum(row["strict_match"] for row in package_rows)
    return {"source_case_group": source_case_group_id(case), "split": split, "screen_case": screen, "system": case.source_system, "category": case.source_fault_category, "root_cause": case.source_fault_type, "difficulty": case.upstream_difficulty, "available_modalities": _case_modalities(case), "selected_golden_path_policy": "path1_then_path2_union_for_R2", "path1_executable_observations": len(path1), "path2_executable_observations": len(path2), "union_observations": len(union), "package_chars": package["package_chars"], "package_bytes": package["package_bytes"], "process_milestone_count": len((_load_json(root, case, "process_label") or {}).get("milestones", [])), "admissible_evidence_pattern_count": len(patterns), "package_evidence_pattern_match_count": retained, "missing_pattern_count": len(patterns) - retained, "coverage_class": coverage_class(len(patterns), retained), "reason": reason, "relevant_tools_referenced": sorted({str(row["tool"]) for row in patterns}), "relevant_tools_present_path1": sorted({str(row["tool"]) for row in path1}), "relevant_tools_present_union": sorted({str(row["tool"]) for row in union}), "pattern_rows": package_rows, "path1_pattern_rows": p1_rows, "path2_pattern_rows": p2_rows, "union_pattern_rows": union_rows, "source_pattern_rows": source_rows, "diagnosis_input_sufficiency": {"nonempty_executable_evidence": bool(union), "source_modalities_present": bool(_case_modalities(case)), "process_label_coverage_is_proxy": True}}
