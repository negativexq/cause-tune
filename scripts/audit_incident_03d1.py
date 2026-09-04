#!/usr/bin/env python3
"""Run the provider/model-free 03D.1 representation integrity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from causetune.cloudopsbench import (
    CLOUD_OPSBENCH_REVISION,
    build_duplicate_report,
    classify_fallbacks,
    cumulative_context_audit,
    evidence_retention_summary,
    packaging_omission_audit,
    replay_parity,
    representation_manifest,
    scan_corpus,
    source_manifest_fingerprint,
    split_distribution,
    token_budget_projection,
    tool_budget_analysis,
    validate_split_manifest,
)
from causetune.cloudopsbench.scanner import CorpusScanError


EXPECTED_SPLIT_FINGERPRINT = "0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _common(scan: Any, split_manifest: dict[str, Any]) -> dict[str, Any]:
    return {"experiment": "03D.1", "implementation_version": "cloud-opsbench-03d1-integrity-v1", "source_revision": scan.source_revision, "source_manifest_fingerprint": source_manifest_fingerprint(scan), "split_manifest_fingerprint": split_manifest["fingerprint"], "source_root": str(scan.root)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="external pinned Cloud-OpsBench checkout; defaults to CLOUD_OPSBENCH_ROOT")
    parser.add_argument("--split-manifest", type=Path, default=Path("results/incident_telemetry_03d/split_manifest.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03d1"))
    args = parser.parse_args()
    try:
        split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        if split_manifest.get("fingerprint") != EXPECTED_SPLIT_FINGERPRINT:
            raise ValueError("03D split fingerprint is not the frozen value")
        scan = scan_corpus(args.root, source_revision=CLOUD_OPSBENCH_REVISION, fail_closed=True)
        duplicate = build_duplicate_report(scan.root, scan.cases)
        split_check = validate_split_manifest(split_manifest, duplicate, (f"{case.source_revision}:{case.source_system}:{case.source_fault_category}:{case.source_case_id}" for case in scan.cases))
        if split_check["status"] != "pass":
            raise ValueError("frozen split validation failed: " + "; ".join(split_check["errors"]))
        common = _common(scan, split_manifest)
        fallback = classify_fallbacks(scan.root, scan, split_manifest)
        parity = replay_parity(scan.root, scan, split_manifest, fallback)
        omission = packaging_omission_audit(scan.root, scan, split_manifest)
        retention = evidence_retention_summary(omission)
        context = cumulative_context_audit(scan.root, scan, split_manifest)
        token_projection = token_budget_projection(context)
        budgets = tool_budget_analysis(scan.root, scan, split_manifest)
        representation = representation_manifest(scan, split_manifest, fallback, frozen_split_fingerprint=EXPECTED_SPLIT_FINGERPRINT)
        source_trajectory_manifest = args.split_manifest.parent / "trajectory_example_manifest.jsonl"
        eligible_keys = {(item["source_case_group"], item["path"]) for item in representation["eligible_records"]}
        primary_trajectory_rows = [row for row in _read_jsonl(source_trajectory_manifest) if (row["source_case_group"], row["path"]) in eligible_keys]
        if any(row["split"] not in {"TRAIN", "VALIDATION"} for row in primary_trajectory_rows):
            raise ValueError("primary executable trajectory manifest contains protected TEST content")
        golden_only_records = [item for item in fallback["records"] if item["classification"] == "GOLDEN_ONLY_NOT_REPLAYABLE"]
        golden_only = {"version": "cloud-opsbench-03d1-golden-only-v1", "count": len(golden_only_records), "affected_source_cases": len({item["source_case_group"] for item in golden_only_records}), "affected_trajectories": len({(item["source_case_group"], item["path"]) for item in golden_only_records}), "by_tool": dict(sorted(Counter(item["tool_id"] for item in golden_only_records).items())), "by_fault_type": dict(sorted(Counter(item["source_fault_type"] for item in golden_only_records).items())), "by_split": dict(sorted(Counter(item["split"] for item in golden_only_records).items())), "records": golden_only_records, "policy_options": {"A_remove_unsupported_trajectory": "not selected; discards otherwise useful source supervision", "B_truncate_before_unsupported_step": "not selected; changes expert trajectory semantics", "C_auxiliary_only": "selected; exclude affected paths from executable primary SFT and retain for auxiliary analysis"}}
        history = context["history_policy_comparison"]
        history_analysis = {"version": "cloud-opsbench-03d1-history-policy-v1", "policies": history, "primary_recommendation": "FULL_HISTORY when candidate context supports it; deterministic structured compact history is a bounded fallback but is lossy because it replaces observation content with references/metadata", "full_history_findings": {"p95_final_context_characters": history["FULL_HISTORY"]["all"]["characters"]["p95"], "p99_final_context_characters": history["FULL_HISTORY"]["all"]["characters"]["p99"], "max_final_context_characters": history["FULL_HISTORY"]["all"]["characters"]["max"], "32k_estimate_exceeds_trajectories": context["threshold_counts"]["per_complete_trajectory"]["32768"]["exceeding_by_bytes"], "64k_estimate_covers_max": history["FULL_HISTORY"]["all"]["bytes"]["max"] <= 65536 * 4}, "03e_candidate_requirement": "prefer candidates with at least a 64k-token context for lossless full-history coverage under the conservative bytes/4 estimate; a 32k context leaves an explicit five-trajectory overflow tail"}
        summary = {**common, "status": "pass", "source_case_count": len(scan.cases), "trajectory_observation_count": parity["trajectory_observation_count"], "frozen_split_unchanged": split_manifest["fingerprint"] == EXPECTED_SPLIT_FINGERPRINT, "replay_parity": parity["counts"], "original_fallback_count": fallback["original_fallback_count"], "golden_only_observation_count": parity["golden_only_observation_count"], "unresolved_observation_count": parity["unresolved_observation_count"], "packaging_omission_count_at_32768": omission["omission_count"], "required_evidence_retention_at_32768": retention["all_observations"], "full_history_p95_final_context_chars": history["FULL_HISTORY"]["all"]["characters"]["p95"], "representation_fingerprint": representation["representation_fingerprint"], "primary_trajectory_manifest_count": len(primary_trajectory_rows), "primary_sft_unsupported_observations": 0, "no_model_provider_or_gpu": True, "recommendation": "READY_FOR_03E" if parity["unresolved_observation_count"] == 0 and retention["all_observations"]["required_lost"] == 0 else "NOT_READY_FOR_03E"}
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "fallback_classification.json", {**common, **fallback})
        _write(output / "replay_parity.json", {**common, **parity})
        _write(output / "golden_only_cases.json", {**common, **golden_only})
        _write(output / "packaging_omission_audit.json", {**common, **omission})
        _write(output / "evidence_retention.json", {**common, **retention})
        _write(output / "cumulative_context.json", {**common, **context})
        _write(output / "token_budget_projection.json", {**common, **token_projection})
        _write(output / "history_policy_analysis.json", {**common, **history_analysis})
        _write(output / "tool_budget_analysis.json", {**common, **budgets})
        _write(output / "representation_manifest.json", {**common, **representation})
        _write_jsonl(output / "primary_trajectory_manifest.jsonl", primary_trajectory_rows)
        _write(output / "integrity_summary.json", summary)
        _write(output / "split_integrity.json", {**common, "status": "pass", "validation": split_check, "distribution": split_distribution(split_manifest)})
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.json")) if path.name != "artifact_fingerprints.json"}
        _write(output / "artifact_fingerprints.json", {**common, "artifacts": hashes})
        print(json.dumps({"status": "pass", "recommendation": summary["recommendation"], "replay_parity": parity["counts"], "representation_fingerprint": representation["representation_fingerprint"], "output_dir": str(output)}, sort_keys=True))
        return 0
    except (CorpusScanError, FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"03D.1 audit not completed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
