"""Typed source and case representations for Cloud-OpsBench.

The source fields are intentionally distinct from future model-visible input.
In particular, labels, process annotations, and golden trajectories are source
truth/supervision artifacts and are never returned by ``model_visible_view``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

from causetune.incident_telemetry.fingerprints import canonical_json


CLOUD_OPSBENCH_REVISION = "03c415e5709297432282fbbfd499f1bca0f8c347"
CLOUD_OPSBENCH_URL = "https://github.com/LLM4Ops/Cloud-OpsBench"
SOURCE_ADAPTER_VERSION = "cloud-opsbench-adapter-v1"
SOURCE_REGISTRY_VERSION = "cloud-opsbench-source-v1"


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class SourceReference:
    """A relative source-artifact reference with content metadata only."""

    relative_path: str
    sha256: str
    byte_size: int
    present: bool = True
    file_count: int = 1

    def __post_init__(self) -> None:
        _non_empty(self.relative_path, "relative_path")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise ValueError("sha256 must be a lowercase SHA-256 digest")
        if not isinstance(self.byte_size, int) or self.byte_size < 0:
            raise ValueError("byte_size must be a non-negative integer")
        if not isinstance(self.present, bool) or self.file_count < 0:
            raise ValueError("invalid source reference presence/count")

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "present": self.present,
            "file_count": self.file_count,
        }


@dataclass(frozen=True)
class CloudOpsBenchReferences:
    metadata: SourceReference
    tool_cache: SourceReference
    k8s_states: SourceReference
    logs: SourceReference
    metrics: SourceReference | None
    alerts: SourceReference
    code: SourceReference | None
    process_label: SourceReference | None
    golden_path1: SourceReference | None
    golden_path2: SourceReference | None

    def to_dict(self) -> dict[str, Any]:
        return {
            name: (value.to_dict() if value is not None else None)
            for name, value in self.__dict__.items()
        }

    def available_modalities(self) -> tuple[str, ...]:
        names = {
            "metadata": self.metadata,
            "tool_cache": self.tool_cache,
            "kubernetes_state": self.k8s_states,
            "logs": self.logs,
            "metrics": self.metrics,
            "alerts": self.alerts,
            "code": self.code,
            "process_label": self.process_label,
            "golden_path1": self.golden_path1,
            "golden_path2": self.golden_path2,
        }
        return tuple(sorted(name for name, reference in names.items() if reference is not None and reference.present))


@dataclass(frozen=True)
class CloudOpsBenchSource:
    source_name: str = "Cloud-OpsBench"
    official_repository_url: str = CLOUD_OPSBENCH_URL
    upstream_revision: str = CLOUD_OPSBENCH_REVISION
    license: str = "MIT"
    license_file_sha256: str | None = "d19af713964259f9ab5d662bfc706d2c072ee9211495d3d6d4c3c7588f59abbc"
    release_identity: str = "main@03c415e5709297432282fbbfd499f1bca0f8c347"
    systems: tuple[str, ...] = ("boutique", "trainticket")
    documented_case_count: int = 754
    documented_fault_type_count: int = 57
    documented_system_case_counts: Mapping[str, int] = None  # type: ignore[assignment]
    expected_layout: str = "benchmark/<system>/<fault_category>/<case_id>/"
    adapter_version: str = SOURCE_ADAPTER_VERSION

    def __post_init__(self) -> None:
        _non_empty(self.source_name, "source_name")
        _non_empty(self.official_repository_url, "official_repository_url")
        if len(self.upstream_revision) != 40:
            raise ValueError("upstream_revision must be a commit SHA")
        if self.license != "MIT":
            raise ValueError("Cloud-OpsBench source registry expects the verified MIT license")
        if self.documented_case_count <= 0 or self.documented_fault_type_count <= 0:
            raise ValueError("documented source counts must be positive")
        if self.documented_system_case_counts is None:
            object.__setattr__(self, "documented_system_case_counts", {"boutique": 550, "trainticket": 204})

    def to_dict(self) -> dict[str, Any]:
        return {
            "registry_version": SOURCE_REGISTRY_VERSION,
            "source_name": self.source_name,
            "official_repository_url": self.official_repository_url,
            "upstream_revision": self.upstream_revision,
            "license": self.license,
            "license_file_sha256": self.license_file_sha256,
            "release_identity": self.release_identity,
            "systems": list(self.systems),
            "documented_case_count": self.documented_case_count,
            "documented_fault_type_count": self.documented_fault_type_count,
            "documented_system_case_counts": dict(sorted(self.documented_system_case_counts.items())),
            "expected_layout": self.expected_layout,
            "adapter_version": self.adapter_version,
        }


@dataclass(frozen=True)
class CloudOpsBenchCase:
    """Read-only case contract preserving native Cloud-OpsBench semantics."""

    source_system: str
    source_case_id: str
    source_fault_category: str
    source_fault_type: str
    upstream_difficulty: str | None
    ground_truth_metadata: Mapping[str, Any]
    available_modalities: tuple[str, ...]
    references: CloudOpsBenchReferences
    source_fingerprint: str
    source_revision: str = CLOUD_OPSBENCH_REVISION

    def __post_init__(self) -> None:
        for field in ("source_system", "source_case_id", "source_fault_category", "source_fault_type", "source_revision"):
            _non_empty(getattr(self, field), field)
        if len(self.source_revision) != 40:
            raise ValueError("source_revision must be a commit SHA")
        if self.upstream_difficulty is not None:
            _non_empty(self.upstream_difficulty, "upstream_difficulty")
        if len(self.source_fingerprint) != 64:
            raise ValueError("source_fingerprint must be SHA-256")
        if tuple(sorted(self.available_modalities)) != self.available_modalities:
            raise ValueError("available_modalities must be sorted")

    @property
    def grouped_case_id(self) -> str:
        return f"{self.source_revision}:{self.source_system}:{self.source_case_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_system": self.source_system,
            "source_case_id": self.source_case_id,
            "source_fault_category": self.source_fault_category,
            "source_fault_type": self.source_fault_type,
            "upstream_difficulty": self.upstream_difficulty,
            "ground_truth_metadata": dict(self.ground_truth_metadata),
            "available_modalities": list(self.available_modalities),
            "references": self.references.to_dict(),
            "source_fingerprint": self.source_fingerprint,
            "source_revision": self.source_revision,
        }

    def model_visible_view(self) -> dict[str, Any]:
        """Return only an input-boundary manifest, never source labels/targets."""

        visible = {name: getattr(self.references, name).to_dict() for name in ("k8s_states", "logs", "metrics", "alerts", "code") if getattr(self.references, name) is not None}
        return {
            "source_system": self.source_system,
            "source_case_id": self.source_case_id,
            "available_observations": sorted(visible),
            "observation_references": visible,
        }


def source_fingerprint(file_hashes: Mapping[str, str], *, source_revision: str) -> str:
    """Fingerprint a case inventory without copying source contents."""

    payload = {"source_revision": source_revision, "files": dict(sorted(file_hashes.items()))}
    return hashlib.sha256(canonical_json(payload)).hexdigest()
