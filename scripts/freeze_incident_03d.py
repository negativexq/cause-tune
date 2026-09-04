#!/usr/bin/env python3
"""Freeze the provider-free 03D Cloud-OpsBench training contract.

This command reads the pinned external checkout, writes manifests and compact
references, and never copies raw Cloud-OpsBench content into CauseTune.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from causetune.cloudopsbench import (
    CLOUD_OPSBENCH_REVISION,
    CloudOpsBenchSource,
    build_duplicate_report,
    build_packaging_audit,
    build_split_manifest,
    build_static_view_manifest,
    build_trajectory_example_manifest,
    build_leakage_field_policy,
    freeze_contract,
    model_visible_schema,
    packaging_contract,
    scan_corpus,
    source_case_derivation_graph,
    source_case_group_id,
    source_manifest_fingerprint,
    split_distribution,
    supervision_policy,
    target_contract,
    tool_contract,
    validate_split_manifest,
)
from causetune.cloudopsbench.scanner import CorpusScanError


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _source_config(path: Path) -> CloudOpsBenchSource:
    value = json.loads(path.read_text(encoding="utf-8"))
    counts = value["documented_counts"]
    source = CloudOpsBenchSource(
        official_repository_url=value["official_repository_url"],
        upstream_revision=value["upstream_revision"],
        license=value["license"],
        license_file_sha256=value.get("license_file_sha256"),
        release_identity=value["release_identity"],
        documented_case_count=counts["cases"],
        documented_fault_type_count=counts["fault_types"],
        documented_system_case_counts=counts["systems"],
        adapter_version=value["adapter_version"],
    )
    if source.upstream_revision != CLOUD_OPSBENCH_REVISION:
        raise ValueError("03D requires the pinned Cloud-OpsBench revision")
    return source


def _git_revision(root: Path) -> str:
    try:
        return subprocess.run(("git", "-C", str(root), "rev-parse", "HEAD"), check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError(f"unable to verify external checkout revision: {exc}") from exc


def _common(source: CloudOpsBenchSource, scan: Any) -> dict[str, Any]:
    return {
        "experiment": "03D",
        "implementation_version": "cloud-opsbench-03d-training-contract-v1",
        "source": source.to_dict(),
        "source_root": str(scan.root),
        "source_revision": scan.source_revision,
        "source_manifest_fingerprint": source_manifest_fingerprint(scan),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="external pinned Cloud-OpsBench checkout; defaults to CLOUD_OPSBENCH_ROOT")
    parser.add_argument("--config", type=Path, default=Path("configs/cloud_opsbench_03c.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03d"))
    args = parser.parse_args()
    try:
        source = _source_config(args.config)
        root = (args.root or Path.cwd()).expanduser().resolve()
        if not args.root:
            import os
            if os.environ.get("CLOUD_OPSBENCH_ROOT"):
                root = Path(os.environ["CLOUD_OPSBENCH_ROOT"]).expanduser().resolve()
        repo_root = Path(__file__).resolve().parents[1]
        if root == repo_root or repo_root in root.parents:
            raise ValueError("external Cloud-OpsBench checkout must be outside the CauseTune source tree")
        if _git_revision(root) != source.upstream_revision:
            raise ValueError(f"external checkout revision is not pinned: {_git_revision(root)}")
        license_path = root / "LICENSE"
        if not license_path.is_file():
            raise ValueError("external checkout has no LICENSE file")
        license_hash = hashlib.sha256(license_path.read_bytes()).hexdigest()
        if source.license_file_sha256 and license_hash != source.license_file_sha256:
            raise ValueError("external checkout LICENSE hash differs from pinned source registry")
        scan = scan_corpus(root, source_revision=source.upstream_revision, fail_closed=True)
        duplicate_report = build_duplicate_report(root, scan.cases)
        split_manifest = build_split_manifest(scan, duplicate_report)
        validation = validate_split_manifest(split_manifest, duplicate_report, (source_case_group_id(case) for case in scan.cases))
        if validation["status"] != "pass":
            raise ValueError("split validation failed: " + "; ".join(validation["errors"]))
        static_rows = build_static_view_manifest(scan, split_manifest)
        trajectory_rows = build_trajectory_example_manifest(root, scan, split_manifest)
        common = _common(source, scan)
        output = args.output_dir
        output.mkdir(parents=True, exist_ok=True)
        _write(output / "split_manifest.json", {**common, **split_manifest})
        _write(output / "split_distribution.json", {**common, **split_distribution(split_manifest), "validation": validation})
        _write(output / "target_contract.json", {**common, **target_contract()})
        _write(output / "model_visible_schema.json", {**common, **model_visible_schema()})
        _write(output / "tool_contract.json", {**common, **tool_contract()})
        _write(output / "packaging_contract.json", {**common, **packaging_contract()})
        _write(output / "trajectory_normalization.json", {**common, "version": "cloud-opsbench-trajectory-normalization-v1", "trace_schema": "diagnostic_trace[] with tool_name/calling/output", "reasoning_disposition": "DROP", "trajectory_paths_per_case": 2, "source_observations_are_not_copied_to_manifest": True})
        _write(output / "leakage_policy.json", {**common, "version": "cloud-opsbench-03d-leakage-boundary-v1", "fields": build_leakage_field_policy()["fields"], "03d_decisions": {"sanitized_query": "MODEL_VISIBLE_WITH_TRANSFORM", "metadata.result.*": "SUPERVISION_ONLY", "source_case_group": "PROVENANCE_ONLY", "target": "SUPERVISION_ONLY", "packaged_observations": "MODEL_VISIBLE_SAFE"}})
        _write(output / "duplicate_groups.json", {**common, **duplicate_report})
        _write(output / "contamination_audit.json", {**common, "version": "cloud-opsbench-contamination-audit-v1", "source_case_groups": duplicate_report["contamination_case_groups"], "common_tool_sequences_not_contamination": True, "cross_split_validation": validation})
        _write(output / "static_view_manifest.json", {"metadata": common, "record_count": len(static_rows), "records": static_rows})
        _write_jsonl(output / "trajectory_example_manifest.jsonl", trajectory_rows)
        _write(output / "supervision_policy.json", {**common, **supervision_policy()})
        _write(output / "packaging_audit.json", {**common, **build_packaging_audit(trajectory_rows)})
        _write(output / "source_case_derivation_graph.json", {**common, **source_case_derivation_graph(scan, split_manifest)})
        freeze = freeze_contract(scan, split_manifest, duplicate_report)
        _write(output / "freeze_manifest.json", {**common, **freeze, "license_hash_verified": license_hash, "external_revision_verified": True, "split_validation": validation})
        summary = {**common, "status": "pass", "case_count": len(scan.cases), "trajectory_count": len(trajectory_rows), "raw_snapshot_primary": False, "model_loaded": False, "model_evaluated": False, "training_executed": False, "provider_called": False, "split_counts": validation["split_counts"], "freeze_fingerprint": freeze["component_fingerprint"]}
        _write(output / "audit_summary.json", summary)
        artifact_hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(output.glob("*.json")) if path.name != "artifact_fingerprints.json"}
        _write(output / "artifact_fingerprints.json", {**common, "artifacts": artifact_hashes})
        print(json.dumps({"status": "pass", "cases": len(scan.cases), "trajectory_examples": len(trajectory_rows), "split_counts": validation["split_counts"], "freeze_fingerprint": freeze["component_fingerprint"], "output_dir": str(output)}, sort_keys=True))
        return 0
    except (CorpusScanError, FileNotFoundError, OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"03D freeze not completed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
