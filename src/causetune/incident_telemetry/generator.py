"""Deterministic canonical scenario engine for Experiment 03B."""

from __future__ import annotations

import hashlib
import operator
import random
from dataclasses import dataclass, replace
from typing import Any, Mapping

from .archetypes import (
    ARCHETYPE_BY_ID,
    Archetype,
    CausalVariableSpec,
    validate_archetype_catalog,
)
from .fingerprints import canonical_json, scenario_fingerprint
from .models import (
    CanonicalScenario,
    CausalEvidenceDefinition,
    CausalStateVariable,
    DependencyState,
    DeterministicAttribute,
    DistractorDefinition,
    Provenance,
    RecentChange,
)
from .ontology import FAILURE_MODE_BY_ID, ONTOLOGY_VERSION
from .runtimes import build_runtime
from .topologies import TOPOLOGY_FAMILY_BY_ID, build_topology


GENERATOR_VERSION = "incident-telemetry-scenario-engine-03b-v1"
_OPERATORS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


@dataclass(frozen=True)
class GenerationContext:
    """Identity/group data assigned before any future telemetry rendering."""

    split_group_id: str | None = None
    topology_group_id: str | None = None
    generator_family: str = GENERATOR_VERSION
    template_family: str = "not-rendered-03b-v1"
    counterfactual_pair_id: str | None = None


@dataclass(frozen=True)
class ScenarioRequest:
    archetype_id: str
    topology_family: str
    runtime_family: str
    difficulty: str
    seed: int
    context: GenerationContext = GenerationContext()
    affected_component_role: str | None = None

    def __post_init__(self) -> None:
        if self.archetype_id not in ARCHETYPE_BY_ID:
            raise ValueError(f"unknown archetype ID: {self.archetype_id!r}")
        if self.topology_family not in TOPOLOGY_FAMILY_BY_ID:
            raise ValueError(f"unknown topology family: {self.topology_family!r}")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("scenario seed must be a non-negative integer")


@dataclass(frozen=True)
class CounterfactualPair:
    pair_id: str
    source_archetype_id: str
    counterfactual_archetype_id: str
    source_scenario_id: str
    counterfactual_scenario_id: str
    variables_held_constant: tuple[str, ...]
    variables_changed: tuple[str, ...]
    pair_seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "source_archetype_id": self.source_archetype_id,
            "counterfactual_archetype_id": self.counterfactual_archetype_id,
            "source_scenario_id": self.source_scenario_id,
            "counterfactual_scenario_id": self.counterfactual_scenario_id,
            "variables_held_constant": list(self.variables_held_constant),
            "variables_changed": list(self.variables_changed),
            "pair_seed": self.pair_seed,
        }


def _vary_value(spec: CausalVariableSpec, rng: random.Random) -> Any:
    value = spec.incident_value
    if not spec.jitter:
        return value
    if spec.value_type == "integer":
        return value + rng.randint(-int(spec.jitter), int(spec.jitter))
    if spec.value_type in {"number", "ratio"}:
        varied = value + rng.uniform(-float(spec.jitter), float(spec.jitter))
        return max(0, min(1, varied)) if spec.value_type == "ratio" else round(varied, 4)
    return value


def _resolve_role(source_role: str, roles: tuple[str, ...], affected_role: str, rng: random.Random) -> str:
    if source_role == "affected":
        return affected_role
    if source_role == "neighbor":
        candidates = [role for role in roles if role != affected_role]
        if not candidates:
            raise ValueError("archetype requires a neighbor but topology has none")
        return sorted(candidates)[rng.randrange(len(candidates))]
    if source_role not in roles:
        raise ValueError(f"role {source_role!r} is not present in selected topology")
    return source_role


def _attributes(spec_variables: tuple[str, ...], state: Mapping[str, Any]) -> tuple[DeterministicAttribute, ...]:
    return tuple(DeterministicAttribute(name, state[name]) for name in spec_variables)


def validate_state_invariants(archetype: Archetype, state: Mapping[str, Any]) -> None:
    """Apply declarative archetype invariants to generated causal state."""

    for invariant in archetype.invariants:
        left = state[invariant.left_variable]
        right = state[invariant.right_variable] if invariant.right_variable is not None else invariant.right_value
        if not _OPERATORS[invariant.operator](left, right):
            raise ValueError(f"{archetype.archetype_id}: invariant failed: {invariant.invariant_id}")


def _validate_request_compatibility(request: ScenarioRequest, archetype: Archetype, roles: tuple[str, ...]) -> str:
    if request.runtime_family not in archetype.allowed_runtime_families:
        raise ValueError(f"runtime {request.runtime_family!r} is incompatible with {archetype.archetype_id}")
    if request.topology_family not in archetype.compatible_topology_families:
        raise ValueError(f"topology {request.topology_family!r} is incompatible with {archetype.archetype_id}")
    if request.difficulty not in archetype.supported_difficulty_classes:
        raise ValueError(f"difficulty {request.difficulty!r} is unsupported by {archetype.archetype_id}")
    candidates = sorted(set(archetype.allowed_affected_roles) & set(roles))
    if not candidates:
        raise ValueError(f"no compatible affected role for {archetype.archetype_id} and {request.topology_family}")
    if request.affected_component_role is not None and request.affected_component_role not in candidates:
        raise ValueError(f"affected role {request.affected_component_role!r} is incompatible with request")
    return request.affected_component_role or candidates[0]


class ScenarioGenerator:
    """Generate one authoritative scenario per explicit request."""

    def __init__(self, archetypes: Mapping[str, Archetype] | None = None) -> None:
        validate_archetype_catalog()
        self.archetypes = dict(archetypes or ARCHETYPE_BY_ID)

    def generate(self, request: ScenarioRequest) -> CanonicalScenario:
        archetype = self.archetypes.get(request.archetype_id)
        if archetype is None:
            raise ValueError(f"unknown archetype ID: {request.archetype_id!r}")
        topology_spec = TOPOLOGY_FAMILY_BY_ID[request.topology_family]
        affected_role = _validate_request_compatibility(request, archetype, topology_spec.roles)
        rng = random.Random(request.seed)
        topology = build_topology(request.topology_family)
        role_to_component = {item.role: item.component_id for item in topology.components}
        affected_component = role_to_component[affected_role]

        state = {spec.name: _vary_value(spec, rng) for spec in archetype.causal_variables}
        validate_state_invariants(archetype, state)
        generation_identity = {
            "generator_version": GENERATOR_VERSION,
            "archetype_id": archetype.archetype_id,
            "archetype_version": archetype.archetype_version,
            "topology_family": request.topology_family,
            "runtime_family": request.runtime_family,
            "difficulty": request.difficulty,
            "seed": request.seed,
            "affected_component_role": affected_role,
            "context": {
                "split_group_id": request.context.split_group_id,
                "topology_group_id": request.context.topology_group_id,
                "generator_family": request.context.generator_family,
                "template_family": request.context.template_family,
                "counterfactual_pair_id": request.context.counterfactual_pair_id,
            },
        }
        scenario_id = "scenario_03b_" + hashlib.sha256(canonical_json(generation_identity)).hexdigest()[:20]
        roles = topology_spec.roles

        def component_for(source_role: str) -> str:
            return role_to_component[_resolve_role(source_role, roles, affected_role, rng)]

        causal_evidence: list[CausalEvidenceDefinition] = []
        for index, spec in enumerate(archetype.causal_evidence):
            removable = request.difficulty == "INCOMPLETE" and index == 0
            role = "MISSING_REQUIRED" if removable else spec.causal_role
            causal_evidence.append(
                CausalEvidenceDefinition(
                    spec.evidence_id,
                    spec.kind,
                    f"semantic_{archetype.archetype_id}_{spec.evidence_id}",
                    spec.required_for_answer,
                    removable,
                    component_for(spec.source_role),
                    role,
                    _attributes(spec.attribute_variables, state),
                )
            )
        supporting = tuple(
            CausalEvidenceDefinition(
                spec.evidence_id,
                spec.kind,
                f"semantic_{archetype.archetype_id}_{spec.evidence_id}",
                spec.required_for_answer,
                spec.may_be_removed,
                component_for(spec.source_role),
                "SUPPORTING",
                _attributes(spec.attribute_variables, state),
            )
            for spec in archetype.supporting_evidence
        )
        distractors = tuple(
            DistractorDefinition(
                spec.evidence_id,
                spec.kind,
                f"semantic_{archetype.archetype_id}_{spec.evidence_id}",
                True,
                component_for(spec.source_role),
                spec.rule_id,
                tuple(spec.attributes) + _attributes(spec.attribute_variables, state),
            )
            for spec in archetype.permitted_distractors
            if request.difficulty in {"HARD", "COUNTERFACTUAL"}
        )
        recent_changes = tuple(
            RecentChange(
                f"change_{index:03d}",
                component_for(spec.source_role),
                spec.change_type,
                f"2026-01-01T00:{index:02d}:00Z",
                spec.causal,
                spec.pattern_id,
            )
            for index, spec in enumerate(archetype.recent_change_patterns, start=1)
        )
        dependencies = tuple(
            DependencyState(
                f"dependency_{index:03d}",
                component_for(spec.source_role),
                spec.state,
                spec.constraint_id,
            )
            for index, spec in enumerate(archetype.dependency_constraints, start=1)
        )
        provenance = Provenance(
            experiment_version="03B",
            ontology_version=ONTOLOGY_VERSION,
            scenario_schema_version="canonical-scenario-v1",
            telemetry_schema_version="rendered-telemetry-v1",
            output_contract_version="diagnostic-output-v1",
            archetype_id=archetype.archetype_id,
            generator_family=request.context.generator_family,
            template_family=request.context.template_family,
            seed=request.seed,
        )
        scenario = CanonicalScenario(
            scenario_id=scenario_id,
            schema_version="canonical-scenario-v1",
            root_cause_id=archetype.root_cause_id,
            fault_domain=archetype.compatible_fault_domain,
            root_cause_family=archetype.compatible_fault_domain.lower(),
            affected_component=affected_component,
            topology=topology,
            runtime=build_runtime(request.runtime_family),
            causal_state=tuple(CausalStateVariable(name, state[name]) for name in sorted(state)),
            causal_evidence=tuple(causal_evidence) + supporting,
            distractors=distractors,
            recent_changes=recent_changes,
            dependency_state=dependencies,
            difficulty=request.difficulty,
            case_type=request.difficulty,
            answerability="INSUFFICIENT_EVIDENCE" if request.difficulty == "INCOMPLETE" else "ANSWERABLE",
            expected_runbook_id=archetype.remediation_runbook_id,
            provenance=provenance,
            split_group_id=request.context.split_group_id or f"03b:{archetype.split_group_family}:{scenario_id}",
            topology_group_id=request.context.topology_group_id or f"03b:{request.topology_family}",
            counterfactual_pair_id=request.context.counterfactual_pair_id,
        )
        final_provenance = Provenance(
            **{**provenance.to_dict(), "canonical_scenario_hash": scenario_fingerprint(scenario)}
        )
        return replace(scenario, provenance=final_provenance)

    def generate_counterfactual_pair(
        self,
        source_archetype_id: str,
        counterfactual_archetype_id: str,
        *,
        topology_family: str,
        runtime_family: str,
        pair_seed: int,
        pair_id: str | None = None,
        split_group_id: str | None = None,
        context: GenerationContext | None = None,
    ) -> tuple[CanonicalScenario, CanonicalScenario, CounterfactualPair]:
        if source_archetype_id == counterfactual_archetype_id:
            raise ValueError("counterfactual pair requires distinct archetypes")
        pair_id = pair_id or "pair_03b_" + hashlib.sha256(f"{source_archetype_id}|{counterfactual_archetype_id}|{pair_seed}".encode()).hexdigest()[:16]
        context = context or GenerationContext(
            split_group_id=split_group_id or f"03b:counterfactual:{pair_id}",
            topology_group_id=f"03b:{topology_family}",
            counterfactual_pair_id=pair_id,
        )
        if context.counterfactual_pair_id != pair_id:
            raise ValueError("counterfactual context pair ID does not match pair ID")
        first_request = ScenarioRequest(source_archetype_id, topology_family, runtime_family, "COUNTERFACTUAL", pair_seed, context)
        first = self.generate(first_request)
        first_role = next(item.role for item in first.topology.components if item.component_id == first.affected_component)
        second_context = context
        second = self.generate(
            ScenarioRequest(counterfactual_archetype_id, topology_family, runtime_family, "COUNTERFACTUAL", pair_seed, second_context, first_role)
        )
        first_state = dict((item.name, item.value) for item in first.causal_state)
        second_state = dict((item.name, item.value) for item in second.causal_state)
        held = tuple(sorted(name for name in set(first_state) & set(second_state) if first_state[name] == second_state[name]))
        all_names = set(first_state) | set(second_state)
        changed = tuple(
            sorted(
                {
                    name
                    for name in all_names
                    if first_state.get(name, object()) != second_state.get(name, object())
                }
                | {"root_cause_id"}
            )
        )
        pair = CounterfactualPair(
            pair_id,
            source_archetype_id,
            counterfactual_archetype_id,
            first.scenario_id,
            second.scenario_id,
            held,
            changed,
            pair_seed,
        )
        validate_counterfactual_pair(first, second, pair)
        return first, second, pair


def validate_counterfactual_pair(
    source: CanonicalScenario,
    counterfactual: CanonicalScenario,
    pair: CounterfactualPair,
) -> None:
    if source.counterfactual_pair_id != pair.pair_id or counterfactual.counterfactual_pair_id != pair.pair_id:
        raise ValueError("counterfactual pair ID is not present on both scenarios")
    if source.scenario_id != pair.source_scenario_id or counterfactual.scenario_id != pair.counterfactual_scenario_id:
        raise ValueError("counterfactual pair scenario IDs do not match metadata")
    if source.topology.to_dict() != counterfactual.topology.to_dict() or source.runtime != counterfactual.runtime:
        raise ValueError("counterfactual pair changed topology or runtime")
    source_role = next(item.role for item in source.topology.components if item.component_id == source.affected_component)
    counter_role = next(item.role for item in counterfactual.topology.components if item.component_id == counterfactual.affected_component)
    if source_role != counter_role:
        raise ValueError("counterfactual pair changed affected service role")
    if source.root_cause_id == counterfactual.root_cause_id:
        raise ValueError("counterfactual pair did not change root cause")
    first_state = dict((item.name, item.value) for item in source.causal_state)
    second_state = dict((item.name, item.value) for item in counterfactual.causal_state)
    for name in pair.variables_held_constant:
        if name not in first_state or name not in second_state or first_state[name] != second_state[name]:
            raise ValueError(f"declared held-constant variable changed: {name}")
    for name in pair.variables_changed:
        if name == "root_cause_id":
            continue
        if name in first_state and name in second_state and first_state[name] == second_state[name]:
            raise ValueError(f"declared changed variable did not change: {name}")


def validate_generated_scenario(scenario: CanonicalScenario, archetype: Archetype | None = None) -> None:
    """Validate 03B-specific compatibility and declarative expansion rules."""

    archetype = archetype or ARCHETYPE_BY_ID.get(scenario.provenance.archetype_id)
    if archetype is None:
        raise ValueError(f"unknown scenario archetype: {scenario.provenance.archetype_id!r}")
    if scenario.root_cause_id != archetype.root_cause_id:
        raise ValueError("scenario root cause disagrees with its archetype")
    if scenario.fault_domain != archetype.compatible_fault_domain:
        raise ValueError("scenario fault domain disagrees with its archetype")
    if scenario.runtime.runtime_id not in archetype.allowed_runtime_families:
        raise ValueError("scenario runtime is incompatible with its archetype")
    if scenario.topology.topology_family not in archetype.compatible_topology_families:
        raise ValueError("scenario topology is incompatible with its archetype")
    component_by_id = {item.component_id: item for item in scenario.topology.components}
    affected = component_by_id.get(scenario.affected_component or "")
    if affected is None or affected.role not in archetype.allowed_affected_roles:
        raise ValueError("scenario affected component role is incompatible with its archetype")

    state = {item.name: item.value for item in scenario.causal_state}
    expected_variables = {item.name for item in archetype.causal_variables}
    if set(state) != expected_variables:
        raise ValueError("scenario state variables disagree with its archetype")
    validate_state_invariants(archetype, state)

    expected_evidence = {item.evidence_id for item in (*archetype.causal_evidence, *archetype.supporting_evidence)}
    actual_evidence = {item.evidence_id for item in scenario.causal_evidence}
    if actual_evidence != expected_evidence:
        raise ValueError("scenario causal evidence IDs disagree with its archetype")
    expected_distractors = (
        {item.evidence_id for item in archetype.permitted_distractors}
        if scenario.difficulty in {"HARD", "COUNTERFACTUAL"}
        else set()
    )
    if {item.evidence_id for item in scenario.distractors} != expected_distractors:
        raise ValueError("scenario distractor set disagrees with difficulty/archetype")
    if expected_evidence & {item.evidence_id for item in scenario.distractors}:
        raise ValueError("scenario distractor overlaps archetype evidence")
    for definition in scenario.causal_evidence:
        if definition.source_component not in component_by_id:
            raise ValueError("scenario evidence source is not in topology")
        if definition.causal_role == "DISTRACTOR":
            raise ValueError("scenario causal evidence is marked as a distractor")
    for distractor in scenario.distractors:
        if not distractor.approved_noise or distractor.source_component not in component_by_id:
            raise ValueError("scenario contains an unapproved or invalid distractor")
        if distractor.distractor_rule_id not in {item.rule_id for item in archetype.permitted_distractors}:
            raise ValueError("scenario distractor rule is not permitted by archetype")
    if scenario.difficulty == "INCOMPLETE":
        if not any(item.causal_role == "MISSING_REQUIRED" and item.may_be_removed for item in scenario.causal_evidence):
            raise ValueError("INCOMPLETE scenario does not mark missing required evidence")
