"""Read-only Cloud-OpsBench adoption and corpus-audit contracts.

This package never downloads, mutates, renders, or trains on Cloud-OpsBench.
It preserves the upstream case taxonomy and provenance while producing small
audit summaries for the future 03D split/freeze decision.
"""

from .models import (
    CLOUD_OPSBENCH_REVISION,
    SOURCE_ADAPTER_VERSION,
    CloudOpsBenchCase,
    CloudOpsBenchReferences,
    CloudOpsBenchSource,
    SourceReference,
)
from .scanner import CorpusScan, ScanIssue, scan_corpus
from .taxonomy import (
    UPSTREAM_FAULT_TAXONOMY,
    TaxonomyClassification,
    build_taxonomy_mapping,
    build_taxonomy_report,
)
from .audit import (
    audit_corpus,
    build_context_size_audit,
    build_corpus_census,
    build_golden_trajectory_audit,
    build_real_leakage_audit,
    build_leakage_field_policy,
    build_modality_availability,
    build_split_feasibility,
    build_target_availability,
)

__all__ = [
    "CLOUD_OPSBENCH_REVISION",
    "SOURCE_ADAPTER_VERSION",
    "CloudOpsBenchCase",
    "CloudOpsBenchReferences",
    "CloudOpsBenchSource",
    "SourceReference",
    "CorpusScan",
    "ScanIssue",
    "scan_corpus",
    "UPSTREAM_FAULT_TAXONOMY",
    "TaxonomyClassification",
    "build_taxonomy_mapping",
    "build_taxonomy_report",
    "audit_corpus",
    "build_context_size_audit",
    "build_corpus_census",
    "build_golden_trajectory_audit",
    "build_real_leakage_audit",
    "build_leakage_field_policy",
    "build_modality_availability",
    "build_split_feasibility",
    "build_target_availability",
]
