"""Small, versioned ontology for the Experiment 03A incident contract.

The ontology is deliberately data-only.  It does not generate scenarios and it
does not contain any language-model or provider integration.
"""

from __future__ import annotations

from dataclasses import dataclass


ONTOLOGY_VERSION = "incident-telemetry-ontology-v1"

FAULT_DOMAINS: tuple[str, ...] = (
    "RESOURCE",
    "DATABASE",
    "DEPENDENCY",
    "NETWORK",
    "APPLICATION",
    "MESSAGING",
    "CACHE",
    "KUBERNETES_RUNTIME",
    "SCALING",
)


@dataclass(frozen=True)
class FailureModeSpec:
    """Stable machine-readable failure-mode metadata."""

    failure_mode_id: str
    fault_domain: str


_TAXONOMY: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "RESOURCE",
        "resource",
        (
            "container_memory_exhaustion",
            "memory_leak",
            "cpu_throttling",
            "disk_io_saturation",
            "disk_pressure",
        ),
    ),
    (
        "DATABASE",
        "database",
        (
            "db_connection_pool_exhaustion",
            "db_query_regression",
            "db_capacity_saturation",
        ),
    ),
    (
        "DEPENDENCY",
        "dependency",
        (
            "downstream_dependency_timeout",
            "downstream_dependency_unavailable",
            "rate_limit_misconfiguration",
        ),
    ),
    (
        "NETWORK",
        "network",
        (
            "dns_resolution_failure",
            "connection_refused",
            "tls_certificate_expiration",
        ),
    ),
    (
        "APPLICATION",
        "application",
        (
            "thread_pool_exhaustion",
            "worker_pool_exhaustion",
            "configuration_regression",
        ),
    ),
    (
        "MESSAGING",
        "messaging",
        (
            "kafka_consumer_lag",
            "message_broker_unavailable",
        ),
    ),
    (
        "CACHE",
        "cache",
        (
            "cache_stampede",
            "cache_unavailable",
        ),
    ),
    (
        "KUBERNETES_RUNTIME",
        "kubernetes_runtime",
        (
            "readiness_probe_failure",
            "crash_loop",
            "scheduling_failure",
            "node_resource_pressure",
        ),
    ),
    (
        "SCALING",
        "scaling",
        (
            "insufficient_replicas",
            "autoscaling_failure",
            "traffic_capacity_exhaustion",
        ),
    ),
)


FAILURE_MODE_SPECS: tuple[FailureModeSpec, ...] = tuple(
    FailureModeSpec(failure_mode_id=mode, fault_domain=domain)
    for domain, _family, modes in _TAXONOMY
    for mode in modes
)
FAILURE_MODE_BY_ID = {spec.failure_mode_id: spec for spec in FAILURE_MODE_SPECS}
FAILURE_MODE_IDS = tuple(spec.failure_mode_id for spec in FAILURE_MODE_SPECS)
FAILURE_MODE_SET = frozenset(FAILURE_MODE_IDS)
ROOT_CAUSE_FAMILIES: tuple[str, ...] = tuple(family for _domain, family, _modes in _TAXONOMY)
ROOT_CAUSE_FAMILY_SET = frozenset(ROOT_CAUSE_FAMILIES)

OBSERVATION_KINDS: tuple[str, ...] = (
    "application_log",
    "runtime_event",
    "metric",
    "change",
    "dependency_health",
    "alert",
    "topology",
)
OBSERVATION_KIND_SET = frozenset(OBSERVATION_KINDS)


def validate_ontology() -> None:
    """Fail closed if the checked-in V1 ontology becomes internally invalid."""

    if len(FAILURE_MODE_IDS) != len(set(FAILURE_MODE_IDS)):
        raise ValueError("ontology contains duplicate failure-mode IDs")
    if set(FAILURE_MODE_BY_ID) != FAILURE_MODE_SET:
        raise ValueError("ontology failure-mode index is inconsistent")
    if any(not item or item.upper() not in FAULT_DOMAINS for item in FAULT_DOMAINS):
        raise ValueError("ontology contains an invalid fault domain")
    if len(ROOT_CAUSE_FAMILIES) != len(set(ROOT_CAUSE_FAMILIES)):
        raise ValueError("ontology contains duplicate root-cause families")
    for spec in FAILURE_MODE_SPECS:
        if not spec.failure_mode_id or spec.fault_domain not in FAULT_DOMAINS:
            raise ValueError(f"invalid ontology entry: {spec!r}")


validate_ontology()
