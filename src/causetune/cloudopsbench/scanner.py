"""Read-only Cloud-OpsBench filesystem scanner."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from causetune.incident_telemetry.fingerprints import canonical_json

from .models import (
    CLOUD_OPSBENCH_REVISION,
    CloudOpsBenchCase,
    CloudOpsBenchReferences,
    SourceReference,
    source_fingerprint,
)


@dataclass(frozen=True)
class ScanIssue:
    path: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


class CorpusScanError(ValueError):
    """Raised when fail-closed scanning finds source-contract violations."""

    def __init__(self, issues: Iterable[ScanIssue]):
        self.issues = tuple(issues)
        super().__init__("Cloud-OpsBench scan failed: " + "; ".join(item.message for item in self.issues))


@dataclass(frozen=True)
class CorpusScan:
    root: Path
    source_revision: str
    cases: tuple[CloudOpsBenchCase, ...]
    issues: tuple[ScanIssue, ...]

    def require_clean(self) -> "CorpusScan":
        errors = tuple(item for item in self.issues if item.severity == "error")
        if errors:
            raise CorpusScanError(errors)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "source_revision": self.source_revision,
            "case_count": len(self.cases),
            "issues": [item.to_dict() for item in self.issues],
            "cases": [item.to_dict() for item in self.cases],
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_reference(root: Path, path: Path, *, relative_path: str | None = None) -> SourceReference:
    if path.is_dir():
        files = sorted(item for item in path.rglob("*") if item.is_file())
        entries = []
        total = 0
        for item in files:
            data = item.read_bytes()
            total += len(data)
            entries.append({"path": item.relative_to(root).as_posix(), "sha256": _sha256_bytes(data), "bytes": len(data)})
        digest = _sha256_bytes(canonical_json(entries))
        return SourceReference(relative_path or path.relative_to(root).as_posix(), digest, total, True, len(files))
    data = path.read_bytes()
    return SourceReference(relative_path or path.relative_to(root).as_posix(), _sha256_bytes(data), len(data), True, 1)


def _read_json(path: Path, issues: list[ScanIssue]) -> Any | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(ScanIssue(str(path), f"invalid JSON: {exc}"))
        return None
    if not isinstance(value, (dict, list)):
        issues.append(ScanIssue(str(path), "JSON root must be an object or array"))
    return value


def _find_value(value: Any, names: set[str]) -> Any | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower().replace("-", "_") in names:
                return item
        for item in value.values():
            found = _find_value(item, names)
            if found is not None:
                return found
    elif isinstance(value, list):
        for item in value:
            found = _find_value(item, names)
            if found is not None:
                return found
    return None


def _metadata_targets(metadata: Any) -> dict[str, Any]:
    if not isinstance(metadata, dict):
        return {}
    fields: dict[str, Any] = {}
    for name, aliases in {
        "fault_type": {"fault_type", "faulttype", "failure_mode", "failuremode", "root_cause", "rootcause"},
        "fault_category": {"fault_category", "faultcategory", "category", "fault_taxonomy", "faulttaxonomy"},
        "component": {"faulty_component", "affected_component", "component", "resource", "fault_object", "faultobject"},
        "namespace": {"namespace"},
        "service": {"service", "service_name", "servicename"},
        "diagnosis": {"diagnosis", "ground_truth", "groundtruth", "answer"},
        "query": {"query", "question"},
        "difficulty": {"difficulty", "upstream_difficulty"},
    }.items():
        value = _find_value(metadata, aliases)
        if value is not None:
            fields[name] = value
    return fields


def _case_paths(root: Path, system: str, category: str, case_id: str) -> dict[str, Path]:
    base = root / "benchmark" / system / category / case_id
    return {
        "metadata": base / "metadata.json",
        "tool_cache": base / "tool_cache.json",
        "k8s_states": base / "raw_data" / "k8s_states.json",
        "logs": base / "raw_data" / "logs.json",
        "metrics": base / "raw_data" / "metrics.csv",
        "alerts": base / "raw_data" / "alert.json",
        "code": base / "code",
        "process_label": root / "process-label" / system / category / case_id / "milestone.json",
        "golden_path1": root / "golden-trajectory" / system / category / case_id / "path1.json",
        "golden_path2": root / "golden-trajectory" / system / category / case_id / "path2.json",
    }


def _relative_case_file_hashes(root: Path, paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in paths.values():
        if not path.exists():
            continue
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    hashes[child.relative_to(root).as_posix()] = _sha256_bytes(child.read_bytes())
        else:
            hashes[path.relative_to(root).as_posix()] = _sha256_bytes(path.read_bytes())
    return hashes


def scan_corpus(
    root: str | Path | None = None,
    *,
    source_revision: str = CLOUD_OPSBENCH_REVISION,
    fail_closed: bool = True,
) -> CorpusScan:
    """Scan a local checkout without changing or copying source data.

    ``metrics.csv`` and ``code/`` are optional per the upstream case layout;
    metadata, tool cache, core raw data, and the two diagnosis references are
    checked as required source artifacts.
    """

    configured_root = root or os.environ.get("CLOUD_OPSBENCH_ROOT")
    if not configured_root:
        raise ValueError("Cloud-OpsBench root is required; set CLOUD_OPSBENCH_ROOT or pass root")
    directory = Path(configured_root).expanduser().resolve()
    issues: list[ScanIssue] = []
    if not directory.is_dir():
        raise FileNotFoundError(f"Cloud-OpsBench root does not exist or is not a directory: {directory}")
    benchmark = directory / "benchmark"
    if not benchmark.is_dir():
        raise CorpusScanError((ScanIssue(str(benchmark), "missing benchmark/ directory"),))

    cases: list[CloudOpsBenchCase] = []
    for system_dir in sorted(item for item in benchmark.iterdir() if item.is_dir()):
        for category_dir in sorted(item for item in system_dir.iterdir() if item.is_dir()):
            for case_dir in sorted(item for item in category_dir.iterdir() if item.is_dir()):
                paths = _case_paths(directory, system_dir.name, category_dir.name, case_dir.name)
                metadata = _read_json(paths["metadata"], issues) if paths["metadata"].is_file() else None
                if metadata is None:
                    issues.append(ScanIssue(str(paths["metadata"]), "required metadata.json is missing or invalid"))
                    continue
                required = ("metadata", "tool_cache", "k8s_states", "logs", "alerts")
                missing = [name for name in required if not paths[name].is_file()]
                if missing:
                    issues.append(ScanIssue(str(case_dir), f"missing required artifacts: {missing}"))
                    continue
                fault_type = _find_value(metadata, {"fault_type", "faulttype", "failure_mode", "failuremode", "root_cause", "rootcause"})
                fault_category = _find_value(metadata, {"fault_category", "faultcategory", "category", "fault_taxonomy", "faulttaxonomy"})
                difficulty = _find_value(metadata, {"difficulty", "upstream_difficulty"})
                if not isinstance(fault_type, str) or not fault_type.strip():
                    issues.append(ScanIssue(str(paths["metadata"]), "metadata.result has no string root_cause/fault type"))
                    continue
                if not isinstance(fault_category, str) or not fault_category.strip():
                    issues.append(ScanIssue(str(paths["metadata"]), "metadata.result has no string fault_taxonomy/category"))
                    continue
                if difficulty is not None and not isinstance(difficulty, str):
                    issues.append(ScanIssue(str(paths["metadata"]), "difficulty must be a string when present"))
                    continue
                references: dict[str, SourceReference | None] = {}
                for name, path in paths.items():
                    if path.exists():
                        try:
                            references[name] = _file_reference(directory, path)
                        except OSError as exc:
                            issues.append(ScanIssue(str(path), f"cannot hash source artifact: {exc}"))
                            references[name] = None
                    else:
                        references[name] = None
                if any(references[name] is None for name in required):
                    continue
                assert all(references[name] is not None for name in required)
                source_hashes = _relative_case_file_hashes(directory, paths)
                cases.append(
                    CloudOpsBenchCase(
                        source_system=system_dir.name,
                        source_case_id=case_dir.name,
                        source_fault_category=fault_category,
                        source_fault_type=fault_type,
                        upstream_difficulty=difficulty,
                        ground_truth_metadata=_metadata_targets(metadata),
                        available_modalities=tuple(sorted(name for name, ref in references.items() if ref is not None)),
                        references=CloudOpsBenchReferences(
                            metadata=references["metadata"], tool_cache=references["tool_cache"],
                            k8s_states=references["k8s_states"], logs=references["logs"],
                            metrics=references["metrics"], alerts=references["alerts"], code=references["code"],
                            process_label=references["process_label"], golden_path1=references["golden_path1"],
                            golden_path2=references["golden_path2"],
                        ),
                        source_fingerprint=source_fingerprint(source_hashes, source_revision=source_revision),
                        source_revision=source_revision,
                    )
                )

    scan = CorpusScan(directory, source_revision, tuple(cases), tuple(issues))
    if fail_closed:
        scan.require_clean()
    return scan


def case_counts(cases: Iterable[CloudOpsBenchCase]) -> dict[str, Any]:
    values = tuple(cases)
    return {
        "total_cases": len(values),
        "by_system": dict(sorted(Counter(item.source_system for item in values).items())),
        "by_fault_category": dict(sorted(Counter(item.source_fault_category for item in values).items())),
        "by_native_fault_type": dict(sorted(Counter(item.source_fault_type for item in values).items())),
        "by_upstream_difficulty": dict(sorted(Counter(item.upstream_difficulty or "UNSPECIFIED" for item in values).items())),
    }
