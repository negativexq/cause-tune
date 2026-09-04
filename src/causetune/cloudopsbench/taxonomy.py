"""Native Cloud-OpsBench taxonomy and explicit compatibility mapping."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable

from causetune.incident_telemetry.ontology import FAILURE_MODE_SET


UPSTREAM_FAULT_TAXONOMY: dict[str, tuple[str, ...]] = {
    "Admission_Fault": (
        "namespace_cpu_quota_exceeded", "namespace_memory_quota_exceeded", "namespace_pod_quota_exceeded",
        "namespace_service_quota_exceeded", "namespace_storage_quota_exceeded", "missing_service_account",
    ),
    "Scheduling_Fault": (
        "node_cordon_mismatch", "node_affinity_mismatch", "node_selector_mismatch", "pod_anti_affinity_conflict",
        "taint_toleration_mismatch", "cpu_capacity_mismatch", "memory_capacity_mismatch", "pv_binding_occupied",
        "pvc_selector_mismatch", "pvc_storage_class_mismatch", "pvc_capacity_mismatch", "pvc_access_mode_mismatch",
    ),
    "Startup_Fault": ("volume_mount_permission_denied", "missing_secret_binding", "incorrect_image_reference", "image_registry_dns_failure", "missing_image_pull_secret"),
    "Runtime_Fault": (
        "container_memory_limit_too_low", "liveness_probe_incorrect_protocol", "liveness_probe_incorrect_port",
        "liveness_probe_incorrect_timing", "readiness_probe_incorrect_protocol", "readiness_probe_incorrect_port",
        "service_sidecar_port_conflict", "mysql_invalid_credentials", "mysql_invalid_port", "db_readonly_mode",
        "db_connection_exhaustion", "deployment_zero_replicas",
    ),
    "Service_Routing_Fault": (
        "service_selector_mismatch", "service_port_mapping_mismatch", "service_protocol_mismatch",
        "service_env_var_address_mismatch", "gateway_misrouted", "service_dns_resolution_failure",
    ),
    "Performance_Fault": ("pod_cpu_overload", "pod_network_delay", "node_network_delay", "node_network_packet_loss"),
    "Infrastructure_Fault": ("containerd_unavailable", "kubelet_unavailable", "kube_proxy_unavailable", "kube_scheduler_unavailable"),
    "Application_Code_Defect": (
        "code_missing_parameter", "code_busy_loop", "code_excessive_file_reads", "code_wrong_return",
        "code_excessive_file_writes", "code_wrong_argument_order", "code_artificial_delay", "code_memory_leak",
    ),
}


@dataclass(frozen=True)
class TaxonomyClassification:
    native_fault_type: str
    native_fault_category: str
    classification: str
    normalized_causetune_fault_id: str | None = None
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# Conservative by design: an exact semantic match is not inferred from shared
# words.  The remainder stays visibly unmapped until reviewed for 03D.
_EXPLICIT_ALIASES: dict[str, tuple[str, str, str]] = {
    "container_memory_limit_too_low": ("SOURCE_MORE_SPECIFIC", "container_memory_exhaustion", "native memory-limit fault is narrower than the frozen resource ID"),
    "db_connection_exhaustion": ("COMPATIBLE_ALIAS", "db_connection_pool_exhaustion", "native connection exhaustion matches the frozen pool fault"),
    "service_dns_resolution_failure": ("COMPATIBLE_ALIAS", "dns_resolution_failure", "native service DNS fault matches the frozen network ID"),
    "image_registry_dns_failure": ("PARTIAL_SEMANTIC_OVERLAP", "dns_resolution_failure", "registry-specific DNS failure is narrower than generic DNS failure"),
    "code_memory_leak": ("COMPATIBLE_ALIAS", "memory_leak", "native code memory leak matches the frozen resource ID"),
    "pod_cpu_overload": ("PARTIAL_SEMANTIC_OVERLAP", "cpu_throttling", "overload can cause throttling but is not identical to the frozen mechanism"),
    "readiness_probe_incorrect_protocol": ("SOURCE_MORE_SPECIFIC", "readiness_probe_failure", "probe protocol is a specific readiness failure"),
    "readiness_probe_incorrect_port": ("SOURCE_MORE_SPECIFIC", "readiness_probe_failure", "probe port is a specific readiness failure"),
    "deployment_zero_replicas": ("COMPATIBLE_ALIAS", "insufficient_replicas", "zero replicas is a concrete insufficient-capacity condition"),
}


def native_taxonomy() -> dict[str, str]:
    return {fault_type: category for category, values in UPSTREAM_FAULT_TAXONOMY.items() for fault_type in values}


def build_taxonomy_mapping() -> list[TaxonomyClassification]:
    records: list[TaxonomyClassification] = []
    for category, fault_type in sorted((category, item) for category, values in UPSTREAM_FAULT_TAXONOMY.items() for item in values):
        if fault_type in _EXPLICIT_ALIASES:
            classification, normalized, rationale = _EXPLICIT_ALIASES[fault_type]
        else:
            classification, normalized, rationale = "UNMAPPED", None, "no conservative V1 mapping approved"
        if normalized is not None and normalized not in FAILURE_MODE_SET:
            raise ValueError(f"taxonomy mapping points to unknown CauseTune ID: {normalized}")
        records.append(TaxonomyClassification(fault_type, category, classification, normalized, rationale))
    return records


def build_taxonomy_report(cases: Iterable[Any]) -> dict[str, Any]:
    mapping = build_taxonomy_mapping()
    observed = Counter((item.source_fault_type, item.source_fault_category) for item in cases)
    return {
        "native_fault_type_count": len(native_taxonomy()),
        "native_taxonomy": {category: list(values) for category, values in sorted(UPSTREAM_FAULT_TAXONOMY.items())},
        "observed_fault_type_count": len({item.source_fault_type for item in cases}),
        "mapping_classification_counts": dict(sorted(Counter(item.classification for item in mapping).items())),
        "mapping": [
            {**item.to_dict(), "observed_case_count": observed.get((item.native_fault_type, item.native_fault_category), 0)}
            for item in mapping
        ],
        "authority": "native Cloud-OpsBench labels remain authoritative; normalized IDs are optional audit metadata",
    }
