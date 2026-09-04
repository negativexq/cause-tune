#!/usr/bin/env python3
"""Run the read-only Cloud-OpsBench 03C adoption/corpus audit.

The command requires an explicit local checkout (or CLOUD_OPSBENCH_ROOT). It
never downloads or modifies the external dataset and never loads a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from causetune.cloudopsbench import (
    CloudOpsBenchSource,
    audit_corpus,
    build_context_size_audit,
    build_corpus_census,
    build_golden_trajectory_audit,
    build_real_leakage_audit,
    build_leakage_field_policy,
    build_modality_availability,
    build_split_feasibility,
    build_target_availability,
    build_taxonomy_report,
    scan_corpus,
)
from causetune.cloudopsbench.taxonomy import UPSTREAM_FAULT_TAXONOMY
from causetune.cloudopsbench.scanner import CorpusScanError


def _write(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_source(path: Path) -> CloudOpsBenchSource:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment") != "03C" or value.get("source_name") != "Cloud-OpsBench":
        raise ValueError("invalid Cloud-OpsBench 03C source config")
    counts = value.get("documented_counts", {})
    return CloudOpsBenchSource(
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="local Cloud-OpsBench checkout; defaults to CLOUD_OPSBENCH_ROOT")
    parser.add_argument("--config", type=Path, default=Path("configs/cloud_opsbench_03c.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03c_cloudops"))
    args = parser.parse_args()
    try:
        source = _load_source(args.config)
        scan = scan_corpus(args.root, source_revision=source.upstream_revision, fail_closed=True)
    except (CorpusScanError, FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"Cloud-OpsBench audit not run: {exc}", file=sys.stderr)
        return 2
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": "03C",
        "audit_version": "cloud-opsbench-audit-v1",
        "source": source.to_dict(),
        "source_root": str(scan.root),
        "case_count": len(scan.cases),
    }
    _write(output / "source_manifest.json", contract)
    _write(output / "corpus_census.json", {**contract, **build_corpus_census(scan, source)})
    _write(output / "modality_availability.json", {**contract, **build_modality_availability(scan.cases)})
    _write(output / "native_taxonomy.json", {**contract, "native_taxonomy": {category: list(values) for category, values in sorted(UPSTREAM_FAULT_TAXONOMY.items())}, "observed": build_corpus_census(scan, source)["observed"]})
    _write(output / "taxonomy_mapping.json", {**contract, **build_taxonomy_report(scan.cases)})
    _write(output / "target_availability.json", {**contract, **build_target_availability(scan.cases)})
    _write(output / "context_size_audit.json", {**contract, **build_context_size_audit(scan.root, scan.cases)})
    _write(output / "golden_trajectory_audit.json", {**contract, **build_golden_trajectory_audit(scan.root, scan.cases)})
    _write(output / "leakage_field_policy.json", {**contract, **build_leakage_field_policy()})
    _write(output / "leakage_audit.json", {**contract, **build_real_leakage_audit(scan.root, scan.cases)})
    _write(output / "split_feasibility.json", {**contract, **build_split_feasibility(scan.cases)})
    _write(output / "audit_summary.json", {**contract, **audit_corpus(scan, source)})
    fingerprints = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.glob("*.json"))
        if path.name != "artifact_fingerprints.json"
    }
    _write(output / "artifact_fingerprints.json", {**contract, "artifacts": fingerprints})
    print(json.dumps({"status": "pass", "case_count": len(scan.cases), "output_dir": str(output)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
