"""Typed, provider-independent Experiment 03A domain models.

CanonicalScenario is authoritative for labels.  RenderedTelemetry intentionally
has no root-cause, answerability, or remediation fields; those values cannot be
returned by a renderer through this contract.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

from .ontology import (
    FAILURE_MODE_BY_ID,
    FAILURE_MODE_SET,
    FAULT_DOMAINS,
    OBSERVATION_KIND_SET,
    ONTOLOGY_VERSION,
    ROOT_CAUSE_FAMILY_SET,
)


SCENARIO_SCHEMA_VERSION = "canonical-scenario-v1"
TELEMETRY_SCHEMA_VERSION = "rendered-telemetry-v1"
OUTPUT_CONTRACT_VERSION = "diagnostic-output-v1"

CASE_TYPES: tuple[str, ...] = (
    "STANDARD",
    "HARD",
    "COUNTERFACTUAL",
    "INCOMPLETE",
    "HEALTHY_CONTROL",
)
CASE_TYPE_SET = frozenset(CASE_TYPES)
ANSWERABILITY_VALUES: tuple[str, ...] = (
    "ANSWERABLE",
    "INSUFFICIENT_EVIDENCE",
    "HEALTHY_CONTROL",
)
ANSWERABILITY_SET = frozenset(ANSWERABILITY_VALUES)
EVIDENCE_ROLES: tuple[str, ...] = ("CAUSAL", "SUPPORTING", "DISTRACTOR", "MISSING_REQUIRED")
EVIDENCE_ROLE_SET = frozenset(EVIDENCE_ROLES)
SPLIT_NAMES: tuple[str, ...] = (
    "TRAIN",
    "VALIDATION",
    "ID_TEST",
    "HARD_TEST",
    "TEMPLATE_OOD",
    "TOPOLOGY_OOD",
    "GENERATOR_OOD",
    "COUNTERFACTUAL_TEST",
    "ABSTENTION_TEST",
    "SEALED_HOLDOUT",
)
SPLIT_SET = frozenset(SPLIT_NAMES)

_ID_PATTERNS = {
    "evidence": re.compile(r"^(?:log|metric|event|change|dependency|alert|topology)_\d{3,}$"),
    "component": re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
    "scenario": re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$"),
}


def _non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_id(value: Any, field: str, kind: str) -> str:
    value = _non_empty(value, field)
    if not _ID_PATTERNS[kind].fullmatch(value):
        raise ValueError(f"{field} has invalid ID format: {value!r}")
    return value


def _validate_seed(value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("seed must be a non-negative integer")
    return value


def _json_value(value: Any, field: str) -> None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_value(item, f"{field}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{field} contains a non-string key")
            _json_value(item, f"{field}.{key}")
        return
    raise ValueError(f"{field} contains a non-JSON value")


@dataclass(frozen=True)
class ServiceComponent:
    component_id: str
    role: str
    service_name: str

    def __post_init__(self) -> None:
        _validate_id(self.component_id, "component_id", "component")
        _non_empty(self.role, "role")
        _non_empty(self.service_name, "service_name")

    def to_dict(self) -> dict[str, str]:
        return {
            "component_id": self.component_id,
            "role": self.role,
            "service_name": self.service_name,
        }


@dataclass(frozen=True)
class TopologyEdge:
    source_component: str
    target_component: str
    relation: str = "depends_on"

    def __post_init__(self) -> None:
        _validate_id(self.source_component, "source_component", "component")
        _validate_id(self.target_component, "target_component", "component")
        _non_empty(self.relation, "relation")

    def to_dict(self) -> dict[str, str]:
        return {
            "source_component": self.source_component,
            "target_component": self.target_component,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class ServiceTopology:
    topology_id: str
    topology_family: str
    components: tuple[ServiceComponent, ...]
    edges: tuple[TopologyEdge, ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.topology_id, "topology_id", "scenario")
        _non_empty(self.topology_family, "topology_family")
        if not self.components:
            raise ValueError("topology must contain at least one component")
        component_ids = [item.component_id for item in self.components]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("topology contains duplicate component IDs")
        component_set = set(component_ids)
        for edge in self.edges:
            if edge.source_component not in component_set or edge.target_component not in component_set:
                raise ValueError("topology edge references an unknown component")

    def to_dict(self) -> dict[str, Any]:
        return {
            "topology_id": self.topology_id,
            "topology_family": self.topology_family,
            "components": [item.to_dict() for item in self.components],
            "edges": [item.to_dict() for item in self.edges],
        }


@dataclass(frozen=True)
class RuntimeEnvironment:
    runtime_id: str
    ecosystem: str
    version: str | None = None

    def __post_init__(self) -> None:
        _non_empty(self.runtime_id, "runtime_id")
        _non_empty(self.ecosystem, "ecosystem")
        if self.version is not None:
            _non_empty(self.version, "version")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "runtime_id": self.runtime_id,
            "ecosystem": self.ecosystem,
            "version": self.version,
        }


@dataclass(frozen=True)
class CausalStateVariable:
    name: str
    value: Any

    def __post_init__(self) -> None:
        _non_empty(self.name, "causal state variable name")
        _json_value(self.value, f"causal state variable {self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class DeterministicAttribute:
    name: str
    value: Any

    def __post_init__(self) -> None:
        _non_empty(self.name, "attribute name")
        _json_value(self.value, f"attribute {self.name}")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "value": self.value}


@dataclass(frozen=True)
class CausalEvidenceDefinition:
    evidence_id: str
    kind: str
    description: str
    required_for_answer: bool = True
    may_be_removed: bool = False
    source_component: str | None = None
    causal_role: str = "CAUSAL"
    attributes: tuple[DeterministicAttribute, ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.evidence_id, "evidence_id", "evidence")
        if self.kind not in OBSERVATION_KIND_SET:
            raise ValueError(f"unknown evidence kind: {self.kind!r}")
        _non_empty(self.description, "evidence description")
        if not isinstance(self.required_for_answer, bool) or not isinstance(self.may_be_removed, bool):
            raise ValueError("evidence flags must be boolean")
        if self.source_component is not None:
            _validate_id(self.source_component, "evidence source_component", "component")
        if self.causal_role not in EVIDENCE_ROLE_SET:
            raise ValueError(f"unknown evidence role: {self.causal_role!r}")
        if self.causal_role == "DISTRACTOR":
            raise ValueError("distractor evidence must use DistractorDefinition")
        if self.causal_role == "MISSING_REQUIRED" and not self.may_be_removed:
            raise ValueError("MISSING_REQUIRED evidence must be removable")
        attribute_names = [item.name for item in self.attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("evidence attribute names must be unique")
    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "description": self.description,
            "required_for_answer": self.required_for_answer,
            "may_be_removed": self.may_be_removed,
        }
        # Preserve the original 03A wire shape for legacy/default instances;
        # 03B emits its additive authority metadata when it is populated.
        if self.source_component is not None:
            result["source_component"] = self.source_component
        if self.causal_role != "CAUSAL":
            result["causal_role"] = self.causal_role
        if self.attributes:
            result["attributes"] = [item.to_dict() for item in self.attributes]
        return result


@dataclass(frozen=True)
class DistractorDefinition:
    evidence_id: str
    kind: str
    description: str
    approved_noise: bool = True
    source_component: str | None = None
    distractor_rule_id: str = "explicit_distractor"
    attributes: tuple[DeterministicAttribute, ...] = ()

    def __post_init__(self) -> None:
        _validate_id(self.evidence_id, "distractor evidence_id", "evidence")
        if self.kind not in OBSERVATION_KIND_SET:
            raise ValueError(f"unknown distractor kind: {self.kind!r}")
        _non_empty(self.description, "distractor description")
        if not isinstance(self.approved_noise, bool):
            raise ValueError("approved_noise must be boolean")
        if self.source_component is not None:
            _validate_id(self.source_component, "distractor source_component", "component")
        _non_empty(self.distractor_rule_id, "distractor_rule_id")
        attribute_names = [item.name for item in self.attributes]
        if len(attribute_names) != len(set(attribute_names)):
            raise ValueError("distractor attribute names must be unique")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "description": self.description,
            "approved_noise": self.approved_noise,
        }
        if self.source_component is not None:
            result["source_component"] = self.source_component
        if self.distractor_rule_id != "explicit_distractor":
            result["distractor_rule_id"] = self.distractor_rule_id
        if self.attributes:
            result["attributes"] = [item.to_dict() for item in self.attributes]
        return result


@dataclass(frozen=True)
class RecentChange:
    change_id: str
    component_id: str
    change_type: str
    timestamp: str
    causal: bool = False
    pattern_id: str | None = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"change_\d{3,}", self.change_id):
            raise ValueError(f"invalid change_id: {self.change_id!r}")
        _validate_id(self.component_id, "change component_id", "component")
        _non_empty(self.change_type, "change_type")
        _non_empty(self.timestamp, "change timestamp")
        if not isinstance(self.causal, bool):
            raise ValueError("causal must be boolean")
        if self.pattern_id is not None:
            _non_empty(self.pattern_id, "change pattern_id")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "change_id": self.change_id,
            "component_id": self.component_id,
            "change_type": self.change_type,
            "timestamp": self.timestamp,
            "causal": self.causal,
        }
        if self.pattern_id is not None:
            result["pattern_id"] = self.pattern_id
        return result


@dataclass(frozen=True)
class DependencyState:
    dependency_id: str
    component_id: str
    state: str
    observation: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"dependency_\d{3,}", self.dependency_id):
            raise ValueError(f"invalid dependency_id: {self.dependency_id!r}")
        _validate_id(self.component_id, "dependency component_id", "component")
        _non_empty(self.state, "dependency state")
        _non_empty(self.observation, "dependency observation")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dependency_id": self.dependency_id,
            "component_id": self.component_id,
            "state": self.state,
            "observation": self.observation,
        }


@dataclass(frozen=True)
class Provenance:
    experiment_version: str
    ontology_version: str
    scenario_schema_version: str
    telemetry_schema_version: str
    output_contract_version: str
    archetype_id: str
    generator_family: str
    template_family: str
    seed: int
    renderer_provider: str | None = None
    renderer_model: str | None = None
    renderer_prompt_version: str | None = None
    canonical_scenario_hash: str | None = None
    rendered_telemetry_hash: str | None = None
    split_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        for field in (
            "experiment_version",
            "ontology_version",
            "scenario_schema_version",
            "telemetry_schema_version",
            "output_contract_version",
            "archetype_id",
            "generator_family",
            "template_family",
        ):
            _non_empty(getattr(self, field), field)
        _validate_seed(self.seed)
        for field in ("renderer_provider", "renderer_model", "renderer_prompt_version"):
            value = getattr(self, field)
            if value is not None:
                _non_empty(value, field)
        for field in ("canonical_scenario_hash", "rendered_telemetry_hash", "split_manifest_hash"):
            value = getattr(self, field)
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": self.experiment_version,
            "ontology_version": self.ontology_version,
            "scenario_schema_version": self.scenario_schema_version,
            "telemetry_schema_version": self.telemetry_schema_version,
            "output_contract_version": self.output_contract_version,
            "archetype_id": self.archetype_id,
            "generator_family": self.generator_family,
            "template_family": self.template_family,
            "seed": self.seed,
            "renderer_provider": self.renderer_provider,
            "renderer_model": self.renderer_model,
            "renderer_prompt_version": self.renderer_prompt_version,
            "canonical_scenario_hash": self.canonical_scenario_hash,
            "rendered_telemetry_hash": self.rendered_telemetry_hash,
            "split_manifest_hash": self.split_manifest_hash,
        }


@dataclass(frozen=True)
class CanonicalScenario:
    scenario_id: str
    schema_version: str
    root_cause_id: str | None
    fault_domain: str | None
    root_cause_family: str | None
    affected_component: str | None
    topology: ServiceTopology
    runtime: RuntimeEnvironment
    causal_state: tuple[CausalStateVariable, ...]
    causal_evidence: tuple[CausalEvidenceDefinition, ...]
    distractors: tuple[DistractorDefinition, ...]
    recent_changes: tuple[RecentChange, ...]
    dependency_state: tuple[DependencyState, ...]
    difficulty: str
    case_type: str
    answerability: str
    expected_runbook_id: str | None
    provenance: Provenance
    split_group_id: str
    topology_group_id: str
    counterfactual_pair_id: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.scenario_id, "scenario_id", "scenario")
        if self.schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"unsupported scenario schema version: {self.schema_version!r}")
        if self.root_cause_id is not None:
            if self.root_cause_id not in FAILURE_MODE_SET:
                raise ValueError(f"unknown root cause ID: {self.root_cause_id!r}")
            expected_domain = FAILURE_MODE_BY_ID[self.root_cause_id].fault_domain
            if self.fault_domain != expected_domain:
                raise ValueError("root cause and fault domain are inconsistent")
            if self.root_cause_family not in ROOT_CAUSE_FAMILY_SET:
                raise ValueError("root cause family is required for an answerable cause")
        elif self.fault_domain is not None or self.root_cause_family is not None:
            raise ValueError("healthy controls cannot declare a root-cause domain or family")
        if self.affected_component is not None:
            component_ids = {component.component_id for component in self.topology.components}
            if self.affected_component not in component_ids:
                raise ValueError("affected_component is not in the topology")
        if self.difficulty not in CASE_TYPE_SET:
            raise ValueError(f"unknown difficulty/case type: {self.difficulty!r}")
        if self.case_type != self.difficulty:
            raise ValueError("difficulty and case_type must match")
        if self.answerability not in ANSWERABILITY_SET:
            raise ValueError(f"unknown answerability: {self.answerability!r}")
        if self.case_type == "INCOMPLETE" and self.answerability != "INSUFFICIENT_EVIDENCE":
            raise ValueError("INCOMPLETE cases must be insufficient-evidence cases")
        if self.case_type == "HEALTHY_CONTROL" and self.answerability != "HEALTHY_CONTROL":
            raise ValueError("HEALTHY_CONTROL cases must have healthy-control answerability")
        if self.case_type in {"STANDARD", "HARD", "COUNTERFACTUAL"} and self.answerability != "ANSWERABLE":
            raise ValueError("diagnostic cases must be answerable")
        if self.answerability == "ANSWERABLE":
            if self.root_cause_id is None or self.affected_component is None or self.expected_runbook_id is None:
                raise ValueError("answerable cases require cause, component, and runbook")
        elif self.answerability == "HEALTHY_CONTROL":
            if any(value is not None for value in (self.root_cause_id, self.affected_component, self.expected_runbook_id)):
                raise ValueError("healthy controls cannot require a diagnosis or runbook")
        else:
            if self.root_cause_id is None or self.expected_runbook_id is None:
                raise ValueError("incomplete cases retain authoritative cause and runbook state")
        _non_empty(self.split_group_id, "split_group_id")
        _non_empty(self.topology_group_id, "topology_group_id")
        if self.counterfactual_pair_id is not None:
            _non_empty(self.counterfactual_pair_id, "counterfactual_pair_id")
        component_ids = {component.component_id for component in self.topology.components}
        for change in self.recent_changes:
            if change.component_id not in component_ids:
                raise ValueError("recent change references an unknown component")
        for dependency in self.dependency_state:
            if dependency.component_id not in component_ids:
                raise ValueError("dependency state references an unknown component")
        evidence_ids = [item.evidence_id for item in self.causal_evidence]
        distractor_ids = [item.evidence_id for item in self.distractors]
        if len(evidence_ids) != len(set(evidence_ids)) or len(distractor_ids) != len(set(distractor_ids)):
            raise ValueError("evidence IDs must be unique within each evidence collection")
        if set(evidence_ids) & set(distractor_ids):
            raise ValueError("causal evidence and distractors cannot share evidence IDs")
        if self.answerability == "ANSWERABLE" and not any(item.required_for_answer for item in self.causal_evidence):
            raise ValueError("answerable cases require at least one required causal evidence definition")
        if self.case_type == "INCOMPLETE" and not any(
            item.required_for_answer and item.may_be_removed for item in self.causal_evidence
        ):
            raise ValueError("incomplete cases require explicitly removable required evidence")
        if self.provenance.ontology_version != ONTOLOGY_VERSION:
            raise ValueError("scenario provenance ontology version does not match the frozen ontology")
        if self.provenance.scenario_schema_version != SCENARIO_SCHEMA_VERSION:
            raise ValueError("scenario provenance schema version does not match the scenario")
        if self.provenance.telemetry_schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError("scenario provenance telemetry version does not match the frozen schema")
        if self.provenance.output_contract_version != OUTPUT_CONTRACT_VERSION:
            raise ValueError("scenario provenance output version does not match the frozen contract")
        state_names = [item.name for item in self.causal_state]
        if len(state_names) != len(set(state_names)):
            raise ValueError("causal state variable names must be unique")
        change_ids = [item.change_id for item in self.recent_changes]
        if len(change_ids) != len(set(change_ids)):
            raise ValueError("recent change IDs must be unique")
        dependency_ids = [item.dependency_id for item in self.dependency_state]
        if len(dependency_ids) != len(set(dependency_ids)):
            raise ValueError("dependency IDs must be unique")

    @property
    def declared_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in (*self.causal_evidence, *self.distractors))

    @property
    def required_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.causal_evidence if item.required_for_answer)

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, Any]:
        provenance = self.provenance.to_dict()
        if not include_hashes:
            for key in ("canonical_scenario_hash", "rendered_telemetry_hash", "split_manifest_hash"):
                provenance.pop(key)
        return {
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "root_cause_id": self.root_cause_id,
            "fault_domain": self.fault_domain,
            "root_cause_family": self.root_cause_family,
            "affected_component": self.affected_component,
            "topology": self.topology.to_dict(),
            "runtime": self.runtime.to_dict(),
            "causal_state": [item.to_dict() for item in self.causal_state],
            "causal_evidence": [item.to_dict() for item in self.causal_evidence],
            "distractors": [item.to_dict() for item in self.distractors],
            "recent_changes": [item.to_dict() for item in self.recent_changes],
            "dependency_state": [item.to_dict() for item in self.dependency_state],
            "difficulty": self.difficulty,
            "case_type": self.case_type,
            "answerability": self.answerability,
            "expected_runbook_id": self.expected_runbook_id,
            "provenance": provenance,
            "split_group_id": self.split_group_id,
            "topology_group_id": self.topology_group_id,
            "counterfactual_pair_id": self.counterfactual_pair_id,
        }


@dataclass(frozen=True)
class RendererMetadata:
    provider: str
    model_identifier: str
    prompt_template_version: str
    seed: int

    def __post_init__(self) -> None:
        _non_empty(self.provider, "renderer provider")
        _non_empty(self.model_identifier, "renderer model_identifier")
        _non_empty(self.prompt_template_version, "renderer prompt_template_version")
        _validate_seed(self.seed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model_identifier": self.model_identifier,
            "prompt_template_version": self.prompt_template_version,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TelemetryObservation:
    evidence_id: str
    kind: str
    timestamp: str
    component_id: str | None
    text: str
    value: Any = None

    def __post_init__(self) -> None:
        _validate_id(self.evidence_id, "observation evidence_id", "evidence")
        if self.kind not in OBSERVATION_KIND_SET:
            raise ValueError(f"unknown observation kind: {self.kind!r}")
        _non_empty(self.timestamp, "observation timestamp")
        if self.component_id is not None:
            _validate_id(self.component_id, "observation component_id", "component")
        _non_empty(self.text, "observation text")
        _json_value(self.value, "observation value")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "component_id": self.component_id,
            "text": self.text,
            "value": self.value,
        }


@dataclass(frozen=True)
class RenderedTelemetry:
    scenario_id: str
    schema_version: str
    observations: tuple[TelemetryObservation, ...]
    renderer: RendererMetadata
    canonical_scenario_hash: str
    rendered_telemetry_hash: str | None = None
    split_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        _validate_id(self.scenario_id, "rendered scenario_id", "scenario")
        if self.schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError(f"unsupported telemetry schema version: {self.schema_version!r}")
        if not self.observations:
            raise ValueError("rendered telemetry must contain observations")
        if not re.fullmatch(r"[0-9a-f]{64}", self.canonical_scenario_hash):
            raise ValueError("canonical_scenario_hash must be a lowercase SHA-256 hex digest")
        for field in ("rendered_telemetry_hash", "split_manifest_hash"):
            value = getattr(self, field)
            if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{field} must be a lowercase SHA-256 hex digest")
        ids = [item.evidence_id for item in self.observations]
        if len(ids) != len(set(ids)):
            raise ValueError("rendered telemetry contains duplicate evidence IDs")

    def to_dict(self, *, include_hashes: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "schema_version": self.schema_version,
            "observations": [item.to_dict() for item in self.observations],
            "renderer": self.renderer.to_dict(),
            "canonical_scenario_hash": self.canonical_scenario_hash,
        }
        if include_hashes:
            result["rendered_telemetry_hash"] = self.rendered_telemetry_hash
            result["split_manifest_hash"] = self.split_manifest_hash
        return result


@dataclass(frozen=True)
class DiagnosticOutput:
    root_cause: str | None
    affected_component: str | None
    evidence_ids: tuple[str, ...]
    remediation_runbook_id: str | None
    needs_more_data: bool
    required_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.needs_more_data, bool):
            raise ValueError("needs_more_data must be boolean")
        for field in ("root_cause", "affected_component", "remediation_runbook_id"):
            value = getattr(self, field)
            if value is not None:
                _non_empty(value, field)
        if any(not isinstance(item, str) or not item.strip() for item in (*self.evidence_ids, *self.required_evidence)):
            raise ValueError("output evidence IDs must be non-empty strings")
        if len(self.evidence_ids) != len(set(self.evidence_ids)):
            raise ValueError("output evidence_ids must be unique")
        if len(self.required_evidence) != len(set(self.required_evidence)):
            raise ValueError("output required_evidence must be unique")
        if self.needs_more_data:
            if any(value is not None for value in (self.root_cause, self.affected_component, self.remediation_runbook_id)):
                raise ValueError("abstaining output cannot assert a diagnosis or remediation")
            if not self.required_evidence:
                raise ValueError("abstaining output must request required evidence")
            if self.evidence_ids:
                raise ValueError("abstaining output cannot cite diagnostic evidence")
        elif self.required_evidence:
            raise ValueError("non-abstaining output cannot request required evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_cause": self.root_cause,
            "affected_component": self.affected_component,
            "evidence_ids": list(self.evidence_ids),
            "remediation_runbook_id": self.remediation_runbook_id,
            "needs_more_data": self.needs_more_data,
            "required_evidence": list(self.required_evidence),
        }


def validate_scenario(scenario: CanonicalScenario) -> CanonicalScenario:
    """Return a scenario after enforcing its typed, versioned contract."""

    if not isinstance(scenario, CanonicalScenario):
        raise ValueError("scenario must be CanonicalScenario")
    CanonicalScenario(**{field: getattr(scenario, field) for field in scenario.__dataclass_fields__})
    return scenario
