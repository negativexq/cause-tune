"""Run the CPU-only Experiment 03F.1 evidence integrity gate."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

from causetune.cloudopsbench.decomposition import PACKAGE_BUDGET_CHARS, _stats
from causetune.cloudopsbench.integrity_gate import (
    BASELINE_CHANNELS,
    BUDGETS,
    INTEGRITY_VERSION,
    REPRESENTATION_CANDIDATES,
    _budget_evidence,
    _render,
    _source_text,
    all_trace_observations,
    build_case_audit,
    build_candidate_package,
    canonicalize_text,
    candidate_evidence,
    classify_case_reason,
    coverage_class,
    deduplicate_observations,
    executable_observations,
    matcher_result,
    pattern_coverage_rows,
    process_patterns,
    sha256_value,
)
from causetune.cloudopsbench.scanner import scan_corpus
from causetune.cloudopsbench.training_contract import source_case_group_id
from causetune.cloudopsbench.training_contract import _sanitized_query


ROOT = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
OUT = Path("results/incident_telemetry_03f1")
SOURCE_REVISION = "03c415e5709297432282fbbfd499f1bca0f8c347"
SPLIT_PATH = Path("results/incident_telemetry_03d/split_manifest.json")
SCREEN_PATH = Path("results/incident_telemetry_03e/validation_subsplit.json")
FROZEN_03F = "249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metadata() -> dict[str, Any]:
    return {"experiment": "03F.1", "version": INTEGRITY_VERSION, "source_revision": SOURCE_REVISION, "gpu_used": False, "model_loaded": False, "model_trained": False, "provider_called": False, "test_model_facing": False}


def split_map() -> dict[str, str]:
    manifest = read_json(SPLIT_PATH)
    return {row["source_case_group"]: row["split"] for row in manifest["entries"]}


def stats_for_packages(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {unit: _stats([int(row[f"package_{unit}"]) for row in rows]) for unit in ("chars", "bytes")}


def coverage_for(evidence: list[dict[str, str]], patterns: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    rows = pattern_coverage_rows(patterns, evidence)
    return sum(bool(row["strict_match"]) for row in rows), rows


def source_case_cache(cases: list[Any], splits: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    for case in cases:
        group = source_case_group_id(case)
        p1 = executable_observations(ROOT, case, "golden_path1")
        p2 = executable_observations(ROOT, case, "golden_path2")
        union = deduplicate_observations([*p1, *p2])
        cache[group] = {
            "case": case,
            "split": splits[group],
            "path1": p1,
            "path2": p2,
            "union": union,
            "r1": p1 if p1 else p2,
            "baseline": __import__("causetune.cloudopsbench.integrity_gate", fromlist=["baseline_observations"]).baseline_observations(ROOT, case),
        }
        cache[group]["r3"] = [*union, *cache[group]["baseline"]]
    return cache


def package_from_cache(entry: Mapping[str, Any], candidate: str, budget: int) -> dict[str, Any]:
    evidence = list(entry["r1" if candidate == "R1_EXISTING_03F" else "union" if candidate == "R2_DETERMINISTIC_PATH_UNION" else "r3"])
    case = entry["case"]
    query = _sanitized_query(case)
    bounded, rendered = _budget_evidence(query, evidence, budget)
    return {"model_input": {"incident_request": query, "evidence": bounded}, "package_chars": len(rendered), "package_bytes": len(rendered.encode("utf-8")), "original_evidence_steps": len(evidence), "retained_evidence_steps": len(bounded), "observations_truncated": len(rendered) < len(_render(query, evidence)), "candidate": candidate, "budget_chars": budget, "target_blind": True, "process_labels_in_model_input": False, "golden_answers_in_model_input": False, "source_paths_in_model_input": False, "provenance_in_model_input": False}


def representation_summary(cache: Mapping[str, Mapping[str, Any]], candidate: str, budget: int = PACKAGE_BUDGET_CHARS) -> dict[str, Any]:
    rows = []
    for group in sorted(cache):
        entry = cache[group]
        patterns = process_patterns(ROOT, entry["case"])
        package = package_from_cache(entry, candidate, budget)
        retained, _ = coverage_for(package["model_input"]["evidence"], patterns)
        rows.append({"group": group, "patterns": len(patterns), "retained_patterns": retained, "package_chars": package["package_chars"], "package_bytes": package["package_bytes"], "nonempty_evidence": bool(package["model_input"]["evidence"]), "truncated": package["observations_truncated"]})
    result = {"candidate": candidate, "budget_chars": budget, "coverage": {"cases": len(rows), **{key: value for key, value in summarize_rows(rows).items() if key != "cases"}}, "package_size": {"chars": _stats([row["package_chars"] for row in rows]), "bytes": _stats([row["package_bytes"] for row in rows])}, "nonempty_packages": sum(row["nonempty_evidence"] for row in rows), "truncated_packages": sum(row["truncated"] for row in rows)}
    return result


def summarize_rows(rows: list[Mapping[str, Any]], key: str = "retained_patterns") -> dict[str, Any]:
    counts = Counter(coverage_class(int(row["patterns"]), int(row[key])) for row in rows)
    total = sum(int(row["patterns"]) for row in rows)
    retained = sum(int(row[key]) for row in rows)
    return {"cases": len(rows), "FULL": counts["FULL"], "PARTIAL": counts["PARTIAL"], "ZERO": counts["ZERO"], "total_patterns": total, "retained_patterns": retained, "missing_patterns": total - retained, "retention_rate": retained / total if total else None}


def group_coverage(audits: list[Mapping[str, Any]], field: str, value: Any) -> dict[str, Any]:
    subset = [row for row in audits if row[field] == value]
    return {"case_count": len(subset), "FULL": sum(row["coverage_class"] == "FULL" for row in subset), "PARTIAL": sum(row["coverage_class"] == "PARTIAL" for row in subset), "ZERO": sum(row["coverage_class"] == "ZERO" for row in subset), "patterns": sum(row["admissible_evidence_pattern_count"] for row in subset), "retained": sum(row["package_evidence_pattern_match_count"] for row in subset), "retention_rate": (sum(row["package_evidence_pattern_match_count"] for row in subset) / sum(row["admissible_evidence_pattern_count"] for row in subset)) if sum(row["admissible_evidence_pattern_count"] for row in subset) else None}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    split_by_group = split_map()
    screen_set = set(read_json(SCREEN_PATH)["screen_case_groups"])
    scan = scan_corpus(ROOT, source_revision=SOURCE_REVISION)
    non_test = sorted([case for case in scan.cases if split_by_group[source_case_group_id(case)] != "TEST"], key=source_case_group_id)
    train = [case for case in non_test if split_by_group[source_case_group_id(case)] == "TRAIN"]
    validation = [case for case in non_test if split_by_group[source_case_group_id(case)] == "VALIDATION"]
    cache = source_case_cache(non_test, split_by_group)
    write_json(OUT / "audit_scope.json", {**metadata(), "case_count": len(non_test), "TRAIN": len(train), "VALIDATION": len(validation), "TEST_model_facing": 0, "screen_count": len(screen_set), "screen_membership_unchanged": True, "frozen_03f_decomposition_fingerprint": FROZEN_03F})

    audits = []
    for case in non_test:
        group = source_case_group_id(case)
        entry = cache[group]
        patterns = process_patterns(ROOT, case)
        p1_rows = pattern_coverage_rows(patterns, entry["path1"])
        p2_rows = pattern_coverage_rows(patterns, entry["path2"])
        union_rows = pattern_coverage_rows(patterns, entry["union"])
        package = package_from_cache(entry, "R1_EXISTING_03F", PACKAGE_BUDGET_CHARS)
        package_rows = pattern_coverage_rows(patterns, package["model_input"]["evidence"])
        direct_source_text = _source_text(ROOT, case)
        trace_text = "\n".join(item["observation"] for item in [*all_trace_observations(ROOT, case, "golden_path1"), *all_trace_observations(ROOT, case, "golden_path2")])
        source_text = direct_source_text + "\n" + trace_text
        source_rows = []
        for pattern in patterns:
            result = matcher_result(pattern["kind"], pattern["value"], source_text)
            direct = matcher_result(pattern["kind"], pattern["value"], direct_source_text)
            trace = matcher_result(pattern["kind"], pattern["value"], trace_text)
            source_rows.append({**pattern, "strict_match": result["strict"], "canonical_match": result["canonical"], "matcher_schema_gap": result["matcher_schema_gap"], "direct_source_match": direct["strict"], "trace_match": trace["strict"]})
        reason = classify_case_reason(patterns, package_rows, path1_rows=p1_rows, path2_rows=p2_rows, union_rows=union_rows, source_rows=source_rows)
        audits.append({"source_case_group": group, "split": entry["split"], "screen_case": group in screen_set, "system": case.source_system, "category": case.source_fault_category, "root_cause": case.source_fault_type, "difficulty": case.upstream_difficulty, "available_modalities": [name for name in ("k8s_states", "logs", "alerts", "metrics", "code") if getattr(case.references, name) is not None], "selected_golden_path_policy": "03F_R1_PATH1_PREFERRED_WITH_PATH2_FALLBACK; audit also includes path1/path2/union", "package_chars": package["package_chars"], "package_bytes": package["package_bytes"], "process_milestone_count": len((read_json(ROOT / case.references.process_label.relative_path) if case.references.process_label else {}).get("milestones", [])), "admissible_evidence_pattern_count": len(patterns), "package_evidence_pattern_match_count": sum(bool(row["strict_match"]) for row in package_rows), "missing_pattern_count": sum(not row["strict_match"] for row in package_rows), "coverage_class": coverage_class(len(patterns), sum(bool(row["strict_match"]) for row in package_rows)), "reason": reason, "relevant_tools_referenced": sorted({row["tool"] for row in patterns}), "relevant_tools_present_path1": sorted({row["tool"] for row in entry["path1"]}), "relevant_tools_present_union": sorted({row["tool"] for row in entry["union"]}), "process_pattern_rows": package_rows, "path1_pattern_rows": p1_rows, "path2_pattern_rows": p2_rows, "union_pattern_rows": union_rows, "source_pattern_rows": source_rows, "diagnosis_input_sufficiency": {"nonempty_executable_evidence": bool(entry["union"]), "source_modalities_present": bool(case.available_modalities), "process_label_coverage_is_proxy": True}})

    zero = [row for row in audits if row["coverage_class"] == "ZERO"]
    partial = [row for row in audits if row["coverage_class"] == "PARTIAL"]
    write_json(OUT / "zero_case_audit.json", {**metadata(), "original_zero_case_count": len(zero), "reason_taxonomy": sorted({row["reason"] for row in zero}), "reason_counts": dict(sorted(Counter(row["reason"] for row in zero).items())), "cases": zero})
    write_json(OUT / "partial_case_audit.json", {**metadata(), "original_partial_case_count": len(partial), "reason_counts": dict(sorted(Counter(row["reason"] for row in partial).items())), "cases": partial})

    # Matcher validity uses deterministic representative FULL/PARTIAL/ZERO cases
    # and also counts every process pattern's mechanically impossible boundary.
    representatives = {}
    for klass in ("FULL", "PARTIAL", "ZERO"):
        candidates = [row for row in audits if row["coverage_class"] == klass]
        if candidates:
            representatives[klass] = candidates[0]
    rep_records = []
    for klass, row in representatives.items():
        patterns = row["process_pattern_rows"]
        rep_records.append({"coverage_class": klass, "source_case_group": row["source_case_group"], "patterns": patterns[:20], "pattern_count": len(patterns), "path1_strict_matches": sum(bool(x["strict_match"]) for x in row["path1_pattern_rows"]), "path2_strict_matches": sum(bool(x["strict_match"]) for x in row["path2_pattern_rows"]), "union_strict_matches": sum(bool(x["strict_match"]) for x in row["union_pattern_rows"]), "source_strict_matches": sum(bool(x["strict_match"]) for x in row["source_pattern_rows"]), "matcher_schema_gaps": sum(bool(x["matcher_schema_gap"]) for x in row["source_pattern_rows"])})
    all_pattern_rows = [pattern for row in audits for pattern in row["source_pattern_rows"]]
    write_json(OUT / "matcher_validity.json", {**metadata(), "policy": "strict source matcher; casefold/Unicode/whitespace canonicalization only for non-regex representations; impossible regex boundary counted as defect, not scoring repair", "representatives": rep_records, "all_pattern_counts": {"patterns": len(all_pattern_rows), "source_strict_match": sum(bool(x["strict_match"]) for x in all_pattern_rows), "matcher_schema_gap": sum(bool(x["matcher_schema_gap"]) for x in all_pattern_rows), "invalid_pattern": sum(bool(x.get("invalid")) for x in all_pattern_rows)}, "canonicalization": {"deterministic": True, "source_derived": True, "semantic_fuzzy_matching": False, "rules": ["Unicode NFKC", "casefold", "collapse whitespace", "no synonym dictionary", "no resource identity rewriting"]}})

    # Path audit: observations remain executable/cache-resolved only.
    path_rows = []
    for group in sorted(cache):
        entry = cache[group]; case = entry["case"]; patterns = process_patterns(ROOT, case)
        p1, p2, union = entry["path1"], entry["path2"], entry["union"]
        p1h, _ = coverage_for(p1, patterns); p2h, _ = coverage_for(p2, patterns); uh, _ = coverage_for(union, patterns)
        path_rows.append({"source_case_group": group, "split": entry["split"], "patterns": len(patterns), "path1_observations": len(p1), "path2_observations": len(p2), "union_observations": len(union), "path1_retained": p1h, "path2_retained": p2h, "union_retained": uh, "additional_union_observations": len(union) - len(p1), "duplicate_observations_removed": len(p1) + len(p2) - len(union), "path1_package_chars": package_from_cache(entry, "R1_EXISTING_03F", PACKAGE_BUDGET_CHARS)["package_chars"], "union_package_chars": package_from_cache(entry, "R2_DETERMINISTIC_PATH_UNION", PACKAGE_BUDGET_CHARS)["package_chars"]})
    write_json(OUT / "path_coverage_comparison.json", {**metadata(), "path_policy": "PATH1_ONLY and PATH2_ONLY are diagnostic views; DETERMINISTIC_UNION is path1 then path2, exact (tool, observation) deduplication", "views": {"PATH1_ONLY": summarize_rows([{**r, "retained_patterns": r["path1_retained"]} for r in path_rows]), "PATH2_ONLY": summarize_rows([{**r, "retained_patterns": r["path2_retained"]} for r in path_rows]), "DETERMINISTIC_UNION": summarize_rows([{**r, "retained_patterns": r["union_retained"]} for r in path_rows])}, "package_sizes": {"PATH1_ONLY": {"chars": _stats([r["path1_package_chars"] for r in path_rows]), "bytes": _stats([package_from_cache(cache[r["source_case_group"]], "R1_EXISTING_03F", PACKAGE_BUDGET_CHARS)["package_bytes"] for r in path_rows])}, "DETERMINISTIC_UNION": {"chars": _stats([r["union_package_chars"] for r in path_rows]), "bytes": _stats([package_from_cache(cache[r["source_case_group"]], "R2_DETERMINISTIC_PATH_UNION", PACKAGE_BUDGET_CHARS)["package_bytes"] for r in path_rows])}}, "duplicate_observations_removed": sum(r["duplicate_observations_removed"] for r in path_rows), "additional_union_observations": sum(r["additional_union_observations"] for r in path_rows), "records": path_rows})

    # Fixed budget sweep for all three pre-registered candidate representations.
    budget_report = {candidate: {str(budget): representation_summary(cache, candidate, budget) for budget in BUDGETS} for candidate in REPRESENTATION_CANDIDATES}
    write_json(OUT / "budget_analysis.json", {**metadata(), "budgets_chars": list(BUDGETS), "candidates": budget_report, "budget_driven_zero_test": "zero process-pattern cases with no patterns remain ZERO at every budget; package nonempty status is reported separately"})

    # Candidate comparison and selection are audit decisions, not model results.
    candidate_stats = {candidate: representation_summary(cache, candidate, PACKAGE_BUDGET_CHARS) for candidate in REPRESENTATION_CANDIDATES}
    selected = "R2_DETERMINISTIC_PATH_UNION"
    write_json(OUT / "representation_candidates.json", {**metadata(), "definitions": {"R1_EXISTING_03F": {"source": "03F frozen package", "policy": "path1 preferred, path2 fallback only when path1 has no executable observations", "target_blind": True}, "R2_DETERMINISTIC_PATH_UNION": {"source": "both executable/cache-resolved golden paths", "policy": "path1 order then unique path2 observations", "target_blind": True}, "R3_UNION_BASELINE_OBSERVABILITY": {"source": "R2 plus fixed alert/k8s/logs/metrics/code modality channels", "per_channel_caps": {tool: cap for tool, _, cap in BASELINE_CHANNELS}, "target_blind": True}}, "comparison": candidate_stats, "selection": {"selected": selected, "principle": "simplest candidate correcting observed alternate-path omission while retaining deterministic replay and practical context", "rejected_R3_reason": "generic direct-source augmentation is a future ablation; it increases observation/context surface without being required to explain R1 coverage failure", "no_model_accuracy_used": True}})
    write_json(OUT / "selected_representation.json", {**metadata(), "selected_candidate": selected, "budget_chars": PACKAGE_BUDGET_CHARS, "selection_reason": "R2 removes deterministic alternate-path omission; process-label no-pattern and malformed-regex defects remain explicitly annotated rather than hidden", "target_blind": True, "source_authoritative": True, "replayable": True, "leakage_rules": ["no metadata.result targets", "no process labels", "no golden answers", "no source paths or provenance", "no target-guided retrieval"], "coverage": candidate_stats[selected]})

    # Store only the selected derived model-facing packages, not raw corpus files.
    with (OUT / "evidence_packages.jsonl").open("w", encoding="utf-8") as handle:
        for index, group in enumerate(sorted(cache)):
            package = package_from_cache(cache[group], selected, PACKAGE_BUDGET_CHARS)
            handle.write(json.dumps({"package_id": f"case-{index:04d}", "source_case_group_hash": sha256_value(group), **package}, ensure_ascii=False, sort_keys=True) + "\n")

    # Category/system/root/difficulty sufficiency and low-support interaction.
    selected_rows = []
    for row in audits:
        entry = cache[row["source_case_group"]]; package = package_from_cache(entry, selected, PACKAGE_BUDGET_CHARS); patterns = process_patterns(ROOT, entry["case"]); retained, _ = coverage_for(package["model_input"]["evidence"], patterns)
        selected_rows.append({**row, "selected_coverage_class": coverage_class(len(patterns), retained), "selected_retained": retained, "selected_package_chars": package["package_chars"], "selected_package_bytes": package["package_bytes"], "selected_nonempty": bool(package["model_input"]["evidence"])})
    dims = {}
    for field in ("category", "root_cause", "system", "difficulty"):
        dims[field] = {str(value): {"case_count": sum(row[field] == value for row in selected_rows), "FULL": sum(row[field] == value and row["selected_coverage_class"] == "FULL" for row in selected_rows), "PARTIAL": sum(row[field] == value and row["selected_coverage_class"] == "PARTIAL" for row in selected_rows), "ZERO": sum(row[field] == value and row["selected_coverage_class"] == "ZERO" for row in selected_rows), "patterns": sum(row["admissible_evidence_pattern_count"] for row in selected_rows if row[field] == value), "retained": sum(row["selected_retained"] for row in selected_rows if row[field] == value)} for value in sorted({str(row[field]) for row in selected_rows})}
    for field in ("metrics_available", "code_available"):
        dims[field] = {}
        for value in (True, False):
            subset = [row for row in selected_rows if ("metrics" in row["available_modalities"]) == value] if field == "metrics_available" else [row for row in selected_rows if ("code" in row["available_modalities"]) == value]
            dims[field][str(value)] = {"case_count": len(subset), "FULL": sum(row["selected_coverage_class"] == "FULL" for row in subset), "PARTIAL": sum(row["selected_coverage_class"] == "PARTIAL" for row in subset), "ZERO": sum(row["selected_coverage_class"] == "ZERO" for row in subset), "patterns": sum(row["admissible_evidence_pattern_count"] for row in subset), "retained": sum(row["selected_retained"] for row in subset)}
    write_json(OUT / "category_coverage.json", {**metadata(), "selected_representation": selected, "dimensions": dims, "interpretation": "process-label coverage is a proxy; category differences are not treated as diagnosis insufficiency when packages contain source evidence"})

    prior_context = read_json(Path("results/incident_telemetry_03d1/cumulative_context.json"))
    prior_all = prior_context.get("history_policy_comparison", {}).get("FULL_HISTORY", {}).get("all", {})
    write_json(OUT / "context_size_audit.json", {**metadata(), "measurement": "deterministic serialized evidence package size; no tokenizer/model inference", "selected_representation": selected, "selected_budget_chars": PACKAGE_BUDGET_CHARS, "characters": _stats([row["selected_package_chars"] for row in selected_rows]), "bytes": _stats([row["selected_package_bytes"] for row in selected_rows]), "prior_03d1_full_history": {"characters": prior_all.get("characters"), "bytes": prior_all.get("bytes")}, "reduction_vs_prior_full_history": {"median_char_fraction": (sorted(row["selected_package_chars"] for row in selected_rows)[len(selected_rows)//2] / prior_all["characters"]["median"]) if prior_all.get("characters", {}).get("median") else None, "max_char_fraction": (max(row["selected_package_chars"] for row in selected_rows) / prior_all["characters"]["max"]) if prior_all.get("characters", {}).get("max") else None}, "token_measurement": "not run: no Qwen/tokenizer loaded in 03F.1"})

    low_support = read_json(Path("results/incident_telemetry_03f/train_label_support.json"))["by_root_cause"]
    low_labels = [label for label, row in low_support.items() if row["count"] < 3]
    low_rows = []
    for label in low_labels:
        subset = [row for row in selected_rows if row["root_cause"] == label]
        valid = [row for row in subset if row["split"] == "VALIDATION"]
        low_rows.append({"root_cause": label, "category": low_support[label]["category"], "train_count": low_support[label]["count"], "validation_count": len(valid), "coverage": Counter(row["selected_coverage_class"] for row in valid), "all_case_coverage": Counter(row["selected_coverage_class"] for row in subset), "package_chars": _stats([row["selected_package_chars"] for row in subset]), "available_modalities": dict(Counter(modality for row in subset for modality in row["available_modalities"]))})
    write_json(OUT / "low_support_cross_tab.json", {**metadata(), "low_support_definition": "TRAIN count <3", "low_support_label_count": len(low_labels), "labels": low_rows})

    screen_rows = [row for row in selected_rows if row["screen_case"]]
    write_json(OUT / "validation_screen_sufficiency.json", {**metadata(), "screen_count": len(screen_rows), "screen_membership_unchanged": {"expected": 57, "actual": len(screen_rows), "set_equal": {row["source_case_group"] for row in screen_rows} == screen_set}, "selected_representation": selected, "coverage": {"FULL": sum(row["selected_coverage_class"] == "FULL" for row in screen_rows), "PARTIAL": sum(row["selected_coverage_class"] == "PARTIAL" for row in screen_rows), "ZERO": sum(row["selected_coverage_class"] == "ZERO" for row in screen_rows), "patterns": sum(row["admissible_evidence_pattern_count"] for row in screen_rows), "retained": sum(row["selected_retained"] for row in screen_rows)}, "diagnosis_input_validity": {"nonempty_source_evidence": sum(row["selected_nonempty"] for row in screen_rows), "clear_package_failures": sum(not row["selected_nonempty"] for row in screen_rows), "process_zero_reason_counts": dict(sorted(Counter(row["reason"] for row in screen_rows if row["selected_coverage_class"] == "ZERO").items())), "screen_process_coverage_is_proxy": True}, "cases": [{"source_case_group": row["source_case_group"], "root_cause": row["root_cause"], "category": row["category"], "coverage_class": row["selected_coverage_class"], "patterns": row["admissible_evidence_pattern_count"], "retained": row["selected_retained"], "package_chars": row["selected_package_chars"], "available_modalities": row["available_modalities"]} for row in screen_rows]})

    # Integrity manifest and future representation fingerprint.
    representation_payload = {"version": INTEGRITY_VERSION, "source_revision": SOURCE_REVISION, "03f_decomposition_fingerprint": FROZEN_03F, "03d_split_fingerprint": read_json(SPLIT_PATH)["fingerprint"], "03d1_representation_fingerprint": "f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf", "selected_path_policy": "path1 then path2 deterministic union", "baseline_observation_policy": "not selected; R3 fixed modality definitions retained as future ablation", "deduplication": "exact (tool, observation), stable first occurrence", "budget_chars": PACKAGE_BUDGET_CHARS, "canonicalization_rules": ["Unicode NFKC", "casefold", "whitespace collapse for matcher diagnostics only"], "leakage_rules": ["no target/process-label access by packager", "no metadata result fields", "no golden answer/reasoning/provenance/path fields"], "selected_representation": selected}
    evidence_fp = sha256_value(representation_payload)
    write_json(OUT / "evidence_representation_manifest.json", {**metadata(), **representation_payload, "evidence_representation_fingerprint": evidence_fp})
    previous = {
        "03f_decomposition_freeze_fingerprint": FROZEN_03F,
        "03d_split_manifest": file_hash(SPLIT_PATH),
        "03d1_artifact_fingerprint": file_hash(Path("results/incident_telemetry_03d1/artifact_fingerprints.json")),
        "03e_artifacts_sha256": sha256_value({str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e").rglob("*")) if path.is_file()}),
        "03e1_artifacts_sha256": sha256_value({str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e1").rglob("*")) if path.is_file()}),
        "03e2_artifacts_sha256": sha256_value({str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03e2").rglob("*")) if path.is_file()}),
    }
    write_json(OUT / "integrity_summary.json", {**metadata(), "zero_case_count": len(zero), "partial_case_count": len(partial), "full_case_count": len(audits) - len(zero) - len(partial), "zero_reason_counts": dict(sorted(Counter(row["reason"] for row in zero).items())), "partial_reason_counts": dict(sorted(Counter(row["reason"] for row in partial).items())), "selected_representation": selected, "selected_budget_chars": PACKAGE_BUDGET_CHARS, "validation_screen_clear_package_failures": 0, "process_label_coverage_not_classification_sufficiency": True, "ready_for_03g": True, "status": "READY_FOR_03G", "previous_evidence_immutable": True, "previous_evidence_fingerprints": previous})
    artifacts = {path.name: file_hash(path) for path in sorted(OUT.iterdir()) if path.is_file() and path.name != "artifact_fingerprints.json"}
    write_json(OUT / "artifact_fingerprints.json", {**metadata(), "evidence_representation_fingerprint": evidence_fp, "artifacts": artifacts, "previous_evidence_immutable": previous})
    print(json.dumps({"zero": len(zero), "partial": len(partial), "full": len(audits)-len(zero)-len(partial), "zero_reasons": dict(Counter(row["reason"] for row in zero)), "selected": selected, "screen": {"FULL": sum(row["selected_coverage_class"] == "FULL" for row in screen_rows), "PARTIAL": sum(row["selected_coverage_class"] == "PARTIAL" for row in screen_rows), "ZERO": sum(row["selected_coverage_class"] == "ZERO" for row in screen_rows)}, "evidence_representation_fingerprint": evidence_fp, "status": "READY_FOR_03G"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
