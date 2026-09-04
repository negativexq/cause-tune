"""Declarative causal archetypes for the small 03B scenario engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import DeterministicAttribute, EVIDENCE_ROLE_SET
from .ontology import FAILURE_MODE_BY_ID, OBSERVATION_KIND_SET
from .runtimes import RUNTIME_FAMILY_BY_ID
from .topologies import TOPOLOGY_FAMILY_BY_ID


VARIABLE_TYPES = frozenset({"integer", "number", "ratio", "string", "boolean"})
SUPPORTED_DIFFICULTIES = frozenset({"STANDARD", "HARD", "INCOMPLETE", "COUNTERFACTUAL"})


@dataclass(frozen=True)
class CausalVariableSpec:
    name: str
    value_type: str
    incident_value: Any
    jitter: int | float = 0
    invariant_id: str | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or self.value_type not in VARIABLE_TYPES:
            raise ValueError(f"invalid causal variable specification: {self!r}")
        if self.jitter < 0:
            raise ValueError("causal variable jitter must be non-negative")
        if self.value_type in {"integer", "number", "ratio"} and not isinstance(self.incident_value, (int, float)):
            raise ValueError(f"numeric variable has a non-numeric incident value: {self.name}")
        if self.value_type == "integer" and not isinstance(self.incident_value, int):
            raise ValueError(f"integer variable has a non-integer incident value: {self.name}")
        if self.value_type == "ratio" and not 0 <= self.incident_value <= 1:
            raise ValueError(f"ratio variable is outside [0, 1]: {self.name}")


@dataclass(frozen=True)
class StateInvariant:
    invariant_id: str
    left_variable: str
    operator: str
    right_variable: str | None = None
    right_value: Any = None

    def __post_init__(self) -> None:
        if not self.invariant_id.strip() or not self.left_variable.strip():
            raise ValueError("state invariant requires an ID and left variable")
        if self.operator not in {"==", "!=", ">", ">=", "<", "<="}:
            raise ValueError(f"unsupported state invariant operator: {self.operator!r}")
        if (self.right_variable is None) == (self.right_value is None):
            raise ValueError("state invariant needs exactly one right-hand operand")


@dataclass(frozen=True)
class EvidenceSpec:
    evidence_id: str
    kind: str
    source_role: str
    causal_role: str
    required_for_answer: bool = True
    may_be_removed: bool = False
    attribute_variables: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or self.kind not in OBSERVATION_KIND_SET:
            raise ValueError(f"invalid evidence specification: {self!r}")
        if not self.source_role.strip() or self.causal_role not in EVIDENCE_ROLE_SET - {"DISTRACTOR"}:
            raise ValueError(f"invalid evidence role/source: {self!r}")
        if self.causal_role == "MISSING_REQUIRED" and not self.may_be_removed:
            raise ValueError("MISSING_REQUIRED evidence must be removable")


@dataclass(frozen=True)
class DistractorSpec:
    evidence_id: str
    kind: str
    source_role: str
    rule_id: str
    attribute_variables: tuple[str, ...] = ()
    attributes: tuple[DeterministicAttribute, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or self.kind not in OBSERVATION_KIND_SET:
            raise ValueError(f"invalid distractor specification: {self!r}")
        if not self.source_role.strip() or not self.rule_id.strip():
            raise ValueError("distractor source role and rule are required")
        attribute_names = [item.name for item in self.attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("distractor attribute names must be unique")


@dataclass(frozen=True)
class ChangePatternSpec:
    pattern_id: str
    change_type: str
    source_role: str
    causal: bool


@dataclass(frozen=True)
class DependencyConstraintSpec:
    constraint_id: str
    source_role: str
    state: str


@dataclass(frozen=True)
class Archetype:
    archetype_id: str
    archetype_version: str
    root_cause_id: str
    compatible_fault_domain: str
    allowed_affected_roles: tuple[str, ...]
    allowed_runtime_families: tuple[str, ...]
    compatible_topology_families: tuple[str, ...]
    causal_variables: tuple[CausalVariableSpec, ...]
    invariants: tuple[StateInvariant, ...]
    causal_evidence: tuple[EvidenceSpec, ...]
    supporting_evidence: tuple[EvidenceSpec, ...]
    permitted_distractors: tuple[DistractorSpec, ...]
    recent_change_patterns: tuple[ChangePatternSpec, ...]
    dependency_constraints: tuple[DependencyConstraintSpec, ...]
    remediation_runbook_id: str
    answerability_capabilities: tuple[str, ...]
    supported_difficulty_classes: tuple[str, ...]
    split_group_family: str

    def __post_init__(self) -> None:
        if not self.archetype_id.strip() or not self.archetype_version.strip():
            raise ValueError("archetype ID and version are required")
        if self.root_cause_id not in FAILURE_MODE_BY_ID:
            raise ValueError(f"unknown archetype root cause: {self.root_cause_id}")
        if self.compatible_fault_domain != FAILURE_MODE_BY_ID[self.root_cause_id].fault_domain:
            raise ValueError(f"archetype domain mismatch: {self.archetype_id}")
        if not self.allowed_affected_roles or len(self.allowed_affected_roles) != len(set(self.allowed_affected_roles)):
            raise ValueError(f"invalid affected roles: {self.archetype_id}")
        if not set(self.allowed_runtime_families).issubset(RUNTIME_FAMILY_BY_ID):
            raise ValueError(f"archetype has unknown runtime family: {self.archetype_id}")
        if not set(self.compatible_topology_families).issubset(TOPOLOGY_FAMILY_BY_ID):
            raise ValueError(f"archetype has unknown topology family: {self.archetype_id}")
        variable_names = [item.name for item in self.causal_variables]
        if len(variable_names) != len(set(variable_names)):
            raise ValueError(f"archetype has duplicate causal variables: {self.archetype_id}")
        invariant_ids = [item.invariant_id for item in self.invariants]
        if len(invariant_ids) != len(set(invariant_ids)):
            raise ValueError(f"archetype has duplicate invariants: {self.archetype_id}")
        if any(item.left_variable not in variable_names for item in self.invariants):
            raise ValueError(f"archetype invariant references unknown variable: {self.archetype_id}")
        if any(item.right_variable is not None and item.right_variable not in variable_names for item in self.invariants):
            raise ValueError(f"archetype invariant references unknown RHS variable: {self.archetype_id}")
        evidence = (*self.causal_evidence, *self.supporting_evidence)
        evidence_ids = [item.evidence_id for item in evidence]
        distractor_ids = [item.evidence_id for item in self.permitted_distractors]
        if len(evidence_ids) != len(set(evidence_ids)) or len(distractor_ids) != len(set(distractor_ids)):
            raise ValueError(f"archetype evidence IDs are not unique: {self.archetype_id}")
        if set(evidence_ids) & set(distractor_ids):
            raise ValueError(f"archetype marks an evidence ID as causal and distractor: {self.archetype_id}")
        if not self.causal_evidence:
            raise ValueError(f"archetype requires causal evidence: {self.archetype_id}")
        if not self.remediation_runbook_id.strip() or not self.split_group_family.strip():
            raise ValueError(f"archetype requires runbook and split group family: {self.archetype_id}")
        if not set(self.answerability_capabilities).issubset({"ANSWERABLE", "INSUFFICIENT_EVIDENCE"}):
            raise ValueError(f"unsupported answerability capability: {self.archetype_id}")
        if not set(self.supported_difficulty_classes).issubset(SUPPORTED_DIFFICULTIES):
            raise ValueError(f"unsupported difficulty class: {self.archetype_id}")


def _v(name: str, value_type: str, value: Any, jitter: int | float = 0, invariant_id: str | None = None) -> CausalVariableSpec:
    return CausalVariableSpec(name, value_type, value, jitter, invariant_id)


def _e(evidence_id: str, kind: str, source_role: str, role: str = "CAUSAL", *, required: bool = True, removable: bool = False, attrs: tuple[str, ...] = ()) -> EvidenceSpec:
    return EvidenceSpec(evidence_id, kind, source_role, role, required, removable, attrs)


def _d(evidence_id: str, kind: str, source_role: str, rule_id: str, attrs: tuple[str, ...] = (), attributes: tuple[DeterministicAttribute, ...] = ()) -> DistractorSpec:
    return DistractorSpec(evidence_id, kind, source_role, rule_id, attrs, attributes)


def _archetype(
    archetype_id: str,
    root_cause_id: str,
    domain: str,
    roles: tuple[str, ...],
    runtimes: tuple[str, ...],
    topologies: tuple[str, ...],
    variables: tuple[CausalVariableSpec, ...],
    invariants: tuple[StateInvariant, ...],
    causal_evidence: tuple[EvidenceSpec, ...],
    supporting: tuple[EvidenceSpec, ...] = (),
    distractors: tuple[DistractorSpec, ...] = (),
    runbook_suffix: str = "mitigate",
) -> Archetype:
    return Archetype(
        archetype_id=archetype_id,
        archetype_version="03b-v1",
        root_cause_id=root_cause_id,
        compatible_fault_domain=domain,
        allowed_affected_roles=roles,
        allowed_runtime_families=runtimes,
        compatible_topology_families=topologies,
        causal_variables=variables,
        invariants=invariants,
        causal_evidence=causal_evidence,
        supporting_evidence=supporting,
        permitted_distractors=distractors,
        recent_change_patterns=(
            ChangePatternSpec("causal_change", "deploy", "affected", True),
            ChangePatternSpec("unrelated_change", "deploy", "neighbor", False),
        ),
        dependency_constraints=(DependencyConstraintSpec("healthy_neighbor", "neighbor", "healthy"),),
        remediation_runbook_id=f"runbook.{domain.lower()}.{runbook_suffix}",
        answerability_capabilities=("ANSWERABLE", "INSUFFICIENT_EVIDENCE"),
        supported_difficulty_classes=("STANDARD", "HARD", "INCOMPLETE", "COUNTERFACTUAL"),
        split_group_family=archetype_id,
    )


_COMMON_DISTRACTORS = (
    _d("metric_101", "metric", "neighbor", "mild_neighbor_signal", attributes=(DeterministicAttribute("distractor_intensity", "mild"),)),
    _d("change_101", "change", "neighbor", "recent_change_noise", attributes=(DeterministicAttribute("change_scope", "neighbor"),)),
)


ARCHETYPES: tuple[Archetype, ...] = (
    _archetype(
        "container_memory_exhaustion", "container_memory_exhaustion", "RESOURCE",
        ("api", "checkout_api", "worker", "orders_api"),
        ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "ASYNC_WORKER", "MULTI_SERVICE"),
        (_v("memory_limit", "integer", 4096), _v("working_set", "integer", 7800, 300, "working_set_above_limit"), _v("restart_count", "integer", 4, 1), _v("termination_reason", "string", "OOMKilled"), _v("latency_effect", "string", "elevated")),
        (StateInvariant("working_set_above_limit", "working_set", ">", "memory_limit"),),
        (_e("metric_001", "metric", "affected", attrs=("memory_limit", "working_set")), _e("event_001", "runtime_event", "affected", attrs=("restart_count", "termination_reason"))),
        (_e("alert_001", "alert", "affected", "SUPPORTING", required=False, attrs=("latency_effect",)),),
        _COMMON_DISTRACTORS, "restart_or_replace",
    ),
    _archetype(
        "cpu_throttling", "cpu_throttling", "RESOURCE",
        ("api", "checkout_api", "worker", "orders_api"),
        ("python_fastapi", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "ASYNC_WORKER", "MULTI_SERVICE"),
        (_v("cpu_limit", "number", 2.0), _v("demand", "number", 3.5, 0.3, "demand_above_limit"), _v("throttle_ratio", "ratio", 0.62, 0.05), _v("latency_effect", "string", "elevated")),
        (StateInvariant("demand_above_limit", "demand", ">", "cpu_limit"),),
        (_e("metric_001", "metric", "affected", attrs=("cpu_limit", "demand", "throttle_ratio")), _e("alert_001", "alert", "affected", attrs=("latency_effect",))),
        (_e("event_001", "runtime_event", "affected", "SUPPORTING", required=False, attrs=("throttle_ratio",)),),
        _COMMON_DISTRACTORS,
    ),
    _archetype(
        "db_connection_pool_exhaustion", "db_connection_pool_exhaustion", "DATABASE",
        ("postgres",), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "ASYNC_WORKER"),
        (_v("pool_size", "integer", 40), _v("active_connections", "integer", 40), _v("waiting_requests", "integer", 180, 20), _v("acquisition_timeout_rate", "ratio", 0.71, 0.04), _v("database_health", "string", "healthy")),
        (StateInvariant("active_not_over_pool", "active_connections", "<=", "pool_size"), StateInvariant("waiters_positive", "waiting_requests", ">", right_value=0)),
        (_e("metric_001", "metric", "affected", attrs=("pool_size", "active_connections", "waiting_requests")), _e("log_001", "application_log", "affected", attrs=("acquisition_timeout_rate",))),
        (_e("dependency_001", "dependency_health", "affected", "SUPPORTING", required=False, attrs=("database_health",)),),
        _COMMON_DISTRACTORS,
        "increase_connection_capacity",
    ),
    _archetype(
        "db_query_regression", "db_query_regression", "DATABASE",
        ("postgres",), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "ASYNC_WORKER"),
        (_v("query_latency_ms", "number", 4600, 300), _v("baseline_query_latency_ms", "number", 82), _v("database_cpu", "ratio", 0.77, 0.04), _v("connection_wait", "string", "normal")),
        (StateInvariant("query_slower_than_baseline", "query_latency_ms", ">", "baseline_query_latency_ms"),),
        (_e("metric_001", "metric", "affected", attrs=("query_latency_ms", "baseline_query_latency_ms", "database_cpu")), _e("log_001", "application_log", "affected", attrs=("connection_wait",))),
        (), _COMMON_DISTRACTORS, "mitigate_query",
    ),
    _archetype(
        "downstream_dependency_timeout", "downstream_dependency_timeout", "DEPENDENCY",
        ("payment_api",), ("python_fastapi", "java_spring", "go_http", "node_nestjs"),
        ("CHECKOUT", "MULTI_SERVICE"),
        (_v("dependency_latency_ms", "number", 7100, 400), _v("client_deadline_ms", "number", 2000), _v("local_queue_state", "string", "healthy"), _v("dependency_health", "string", "degraded")),
        (StateInvariant("latency_over_deadline", "dependency_latency_ms", ">", "client_deadline_ms"),),
        (_e("metric_001", "metric", "affected", attrs=("dependency_latency_ms", "client_deadline_ms")), _e("alert_001", "alert", "affected", attrs=("dependency_health",))),
        (_e("dependency_001", "dependency_health", "affected", "SUPPORTING", required=False, attrs=("local_queue_state",)),),
        _COMMON_DISTRACTORS, "restore_dependency",
    ),
    _archetype(
        "dns_resolution_failure", "dns_resolution_failure", "NETWORK",
        ("payment_api",), ("python_fastapi", "java_spring", "go_http", "node_nestjs"),
        ("CHECKOUT", "MULTI_SERVICE"),
        (_v("lookup_latency_ms", "number", 2400, 150), _v("normal_lookup_latency_ms", "number", 3), _v("negative_answer_rate", "ratio", 0.84, 0.04), _v("existing_connections", "string", "healthy")),
        (StateInvariant("lookup_slower_than_baseline", "lookup_latency_ms", ">", "normal_lookup_latency_ms"),),
        (_e("metric_001", "metric", "affected", attrs=("lookup_latency_ms", "normal_lookup_latency_ms")), _e("log_001", "application_log", "affected", attrs=("negative_answer_rate",))),
        (_e("dependency_001", "dependency_health", "affected", "SUPPORTING", required=False, attrs=("existing_connections",)),),
        _COMMON_DISTRACTORS, "restore_dns",
    ),
    _archetype(
        "thread_pool_exhaustion", "thread_pool_exhaustion", "APPLICATION",
        ("api", "checkout_api", "worker", "orders_api"), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "ASYNC_WORKER", "MULTI_SERVICE"),
        (_v("thread_limit", "integer", 64), _v("busy_threads", "integer", 64), _v("blocked_requests", "integer", 120, 20), _v("cpu_state", "string", "moderate")),
        (StateInvariant("busy_not_over_limit", "busy_threads", "<=", "thread_limit"), StateInvariant("blocked_positive", "blocked_requests", ">", right_value=0)),
        (_e("metric_001", "metric", "affected", attrs=("thread_limit", "busy_threads", "blocked_requests")), _e("log_001", "application_log", "affected", attrs=("cpu_state",))),
        (), _COMMON_DISTRACTORS, "restart_or_replace",
    ),
    _archetype(
        "configuration_regression", "configuration_regression", "APPLICATION",
        ("api", "checkout_api", "orders_api"), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"),
        ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "MULTI_SERVICE"),
        (_v("error_rate", "ratio", 0.18, 0.02), _v("baseline_error_rate", "ratio", 0.004), _v("feature_flag_state", "string", "enabled"), _v("dependency_health", "string", "healthy")),
        (StateInvariant("error_rate_above_baseline", "error_rate", ">", "baseline_error_rate"),),
        (_e("metric_001", "metric", "affected", attrs=("error_rate", "baseline_error_rate")), _e("change_001", "change", "affected", attrs=("feature_flag_state",))),
        (_e("dependency_001", "dependency_health", "affected", "SUPPORTING", required=False, attrs=("dependency_health",)),),
        _COMMON_DISTRACTORS, "rollback_configuration",
    ),
    _archetype(
        "kafka_consumer_lag", "kafka_consumer_lag", "MESSAGING",
        ("consumer",), ("python_fastapi", "java_spring", "go_http"), ("EVENT_PIPELINE",),
        (_v("incoming_rate", "number", 1200, 80), _v("processing_rate", "number", 300, 30), _v("lag", "integer", 184000, 12000), _v("consumer_health", "string", "degraded")),
        (StateInvariant("incoming_above_processing", "incoming_rate", ">", "processing_rate"), StateInvariant("lag_positive", "lag", ">", right_value=0)),
        (_e("metric_001", "metric", "affected", attrs=("incoming_rate", "processing_rate", "lag")), _e("alert_001", "alert", "affected", attrs=("consumer_health",))),
        (), _COMMON_DISTRACTORS, "rebalance_consumer",
    ),
    _archetype(
        "cache_stampede", "cache_stampede", "CACHE",
        ("redis",), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"), ("WEB_CACHE_DB", "CHECKOUT"),
        (_v("cache_miss_ratio", "ratio", 0.92, 0.02), _v("database_read_rate", "number", 8500, 500), _v("request_rate", "number", 8500, 500), _v("hot_key_expiry", "string", "shared")),
        (StateInvariant("cache_misses_high", "cache_miss_ratio", ">", right_value=0.8),),
        (_e("metric_001", "metric", "affected", attrs=("cache_miss_ratio", "database_read_rate")), _e("event_001", "runtime_event", "affected", attrs=("hot_key_expiry",))),
        (_e("metric_002", "metric", "affected", "SUPPORTING", required=False, attrs=("request_rate",)),),
        _COMMON_DISTRACTORS, "throttle_cache_pressure",
    ),
    _archetype(
        "readiness_probe_failure", "readiness_probe_failure", "KUBERNETES_RUNTIME",
        ("api", "checkout_api", "orders_api"), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"), ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "MULTI_SERVICE"),
        (_v("ready_replicas", "integer", 0), _v("desired_replicas", "integer", 4), _v("probe_failure_rate", "ratio", 0.95, 0.02), _v("container_state", "string", "running")),
        (StateInvariant("ready_below_desired", "ready_replicas", "<", "desired_replicas"),),
        (_e("metric_001", "metric", "affected", attrs=("ready_replicas", "desired_replicas")), _e("event_001", "runtime_event", "affected", attrs=("probe_failure_rate", "container_state"))),
        (), _COMMON_DISTRACTORS, "restore_readiness",
    ),
    _archetype(
        "insufficient_replicas", "insufficient_replicas", "SCALING",
        ("api", "checkout_api", "orders_api"), ("python_fastapi", "python_django", "java_spring", "go_http", "node_nestjs"), ("WEB_DB", "WEB_CACHE_DB", "CHECKOUT", "MULTI_SERVICE"),
        (_v("desired_replicas", "integer", 8), _v("available_replicas", "integer", 2), _v("traffic_rate", "number", 9000, 500), _v("capacity_state", "string", "exhausted")),
        (StateInvariant("available_below_desired", "available_replicas", "<", "desired_replicas"),),
        (_e("metric_001", "metric", "affected", attrs=("desired_replicas", "available_replicas", "traffic_rate")), _e("alert_001", "alert", "affected", attrs=("capacity_state",))),
        (), _COMMON_DISTRACTORS, "scale_capacity",
    ),
)

ARCHETYPE_BY_ID = {item.archetype_id: item for item in ARCHETYPES}
IMPLEMENTED_ARCHETYPE_IDS = tuple(item.root_cause_id for item in ARCHETYPES)


def validate_archetype_catalog() -> None:
    if len(ARCHETYPES) != len(ARCHETYPE_BY_ID) or len(IMPLEMENTED_ARCHETYPE_IDS) != len(set(IMPLEMENTED_ARCHETYPE_IDS)):
        raise ValueError("archetype catalog contains duplicate IDs")
    for archetype in ARCHETYPES:
        for spec in (*archetype.causal_evidence, *archetype.supporting_evidence):
            if not set(spec.attribute_variables).issubset({variable.name for variable in archetype.causal_variables}):
                raise ValueError(f"evidence references unknown variable: {archetype.archetype_id}")
        for spec in archetype.permitted_distractors:
            if not set(spec.attribute_variables).issubset({variable.name for variable in archetype.causal_variables}):
                raise ValueError(f"distractor references unknown variable: {archetype.archetype_id}")


validate_archetype_catalog()
