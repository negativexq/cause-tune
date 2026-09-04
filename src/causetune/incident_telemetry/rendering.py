"""Controlled Experiment 03C telemetry rendering contracts.

This module is deliberately subordinate to ``CanonicalScenario``.  A render
plan exposes only semantic observations and their constraints; it never passes
authoritative diagnosis labels to a provider.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Mapping

from .fingerprints import rendered_telemetry_fingerprint, scenario_fingerprint
from .models import (
    TELEMETRY_SCHEMA_VERSION,
    CanonicalScenario,
    RenderedTelemetry,
    RendererMetadata,
    TelemetryObservation,
)
from .ontology import FAILURE_MODE_IDS, OBSERVATION_KIND_SET
from .renderer import validate_rendered_telemetry


# The wire names are intentionally a small, explicit V1 vocabulary.  The
# frozen 03A schema continues to store their lower-case semantic kind values.
SURFACE_TYPES: tuple[str, ...] = (
    "APPLICATION_LOG",
    "RUNTIME_EVENT",
    "METRIC",
    "DEPLOYMENT_CHANGE",
    "DEPENDENCY_HEALTH",
    "ALERT",
)
SURFACE_TO_KIND = {
    "APPLICATION_LOG": "application_log",
    "RUNTIME_EVENT": "runtime_event",
    "METRIC": "metric",
    "DEPLOYMENT_CHANGE": "change",
    "DEPENDENCY_HEALTH": "dependency_health",
    "ALERT": "alert",
}
KIND_TO_SURFACE = {value: key for key, value in SURFACE_TO_KIND.items()}
_FORBIDDEN_LABEL_FIELDS = (
    "root_cause",
    "root_cause_id",
    "fault_domain",
    "remediation_runbook_id",
    "answerability",
    "needs_more_data",
    "required_evidence",
    "expected_output",
)
_SPLIT_NAMES = (
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
_BASE_TIMESTAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


class RenderContractError(ValueError):
    """Raised when a provider response crosses the rendering boundary."""

    def __init__(self, message: str, *, category: str = "semantic_consistency") -> None:
        self.category = category
        super().__init__(message)


@dataclass(frozen=True)
class RendererFamilySpec:
    family_id: str
    prompt_template_version: str
    style: str
    runtime_instruction: str

    def __post_init__(self) -> None:
        for field in ("family_id", "prompt_template_version", "style", "runtime_instruction"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field).strip():
                raise ValueError(f"renderer family {field} must be non-empty")


RENDERER_FAMILIES: tuple[RendererFamilySpec, ...] = (
    RendererFamilySpec(
        "concise_ops_v1",
        "03c-concise-ops-v1",
        "compact operational records with short, precise messages",
        "Prefer terse operator language and one observable fact per record.",
    ),
    RendererFamilySpec(
        "verbose_enterprise_v1",
        "03c-verbose-enterprise-v1",
        "verbose enterprise telemetry with explicit field context",
        "Use complete sentences, contextual field names, and incident-timeline phrasing.",
    ),
    RendererFamilySpec(
        "ecosystem_native_v1",
        "03c-ecosystem-native-v1",
        "runtime-native telemetry terminology",
        "Use terminology appropriate to the supplied runtime without naming a diagnosis.",
    ),
)
RENDERER_FAMILY_BY_ID = {item.family_id: item for item in RENDERER_FAMILIES}

_RUNTIME_HINTS = {
    "python_fastapi": "Python/FastAPI request and worker terminology",
    "python_django": "Python/Django request and worker terminology",
    "java_spring": "JVM/Spring/JDBC terminology",
    "go_http": "idiomatic Go HTTP/client terminology",
    "node_nestjs": "Node.js/NestJS request and event-loop terminology",
}


@dataclass(frozen=True)
class EvidenceRenderPlan:
    evidence_id: str
    semantic_kind: str
    source_component: str
    causal_role: str
    required_attributes: tuple[tuple[str, Any], ...]
    may_be_omitted: bool
    permitted_surface_types: tuple[str, ...]
    variation_family: str

    def __post_init__(self) -> None:
        if not self.evidence_id.strip() or self.semantic_kind not in OBSERVATION_KIND_SET:
            raise ValueError("invalid evidence render-plan identity")
        if not self.source_component.strip() or not self.causal_role.strip():
            raise ValueError("render-plan evidence requires source and role")
        if not self.permitted_surface_types or not set(self.permitted_surface_types).issubset(SURFACE_TYPES):
            raise ValueError("render-plan evidence has an unsupported surface")
        names = [name for name, _value in self.required_attributes]
        if len(names) != len(set(names)):
            raise ValueError("render-plan attributes must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "semantic_kind": self.semantic_kind,
            "source_component": self.source_component,
            "causal_role": self.causal_role,
            "required_attributes": [{"name": name, "value": value} for name, value in self.required_attributes],
            "may_be_omitted": self.may_be_omitted,
            "permitted_surface_types": list(self.permitted_surface_types),
            "variation_family": self.variation_family,
        }


@dataclass(frozen=True)
class RenderPlan:
    scenario_id: str
    canonical_scenario_fingerprint: str
    renderer_family: str
    prompt_template_version: str
    runtime_id: str
    render_seed: int
    entries: tuple[EvidenceRenderPlan, ...]
    split_manifest_hash: str | None = None

    def __post_init__(self) -> None:
        if self.renderer_family not in RENDERER_FAMILY_BY_ID:
            raise ValueError(f"unknown renderer family: {self.renderer_family!r}")
        if self.prompt_template_version != RENDERER_FAMILY_BY_ID[self.renderer_family].prompt_template_version:
            raise ValueError("renderer family and prompt version disagree")
        if not re.fullmatch(r"[0-9a-f]{64}", self.canonical_scenario_fingerprint):
            raise ValueError("render plan needs a canonical scenario fingerprint")
        if not isinstance(self.render_seed, int) or isinstance(self.render_seed, bool) or self.render_seed < 0:
            raise ValueError("render seed must be a non-negative integer")
        ids = [item.evidence_id for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("render plan contains duplicate evidence IDs")

    @property
    def expected_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.entries if not item.may_be_omitted)

    @property
    def all_evidence_ids(self) -> frozenset[str]:
        return frozenset(item.evidence_id for item in self.entries)

    def to_provider_dict(self) -> dict[str, Any]:
        """Return the intentionally label-free provider input."""

        family = RENDERER_FAMILY_BY_ID[self.renderer_family]
        return {
            "render_contract": "03c-telemetry-surface-v1",
            "renderer_family": self.renderer_family,
            "prompt_template_version": self.prompt_template_version,
            "runtime_id": self.runtime_id,
            "render_seed": self.render_seed,
            "style": family.style,
            "runtime_instruction": family.runtime_instruction,
            "evidence": [item.to_dict() for item in self.entries],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "canonical_scenario_fingerprint": self.canonical_scenario_fingerprint,
            "renderer_family": self.renderer_family,
            "prompt_template_version": self.prompt_template_version,
            "runtime_id": self.runtime_id,
            "render_seed": self.render_seed,
            "entries": [item.to_dict() for item in self.entries],
            "split_manifest_hash": self.split_manifest_hash,
        }


def _attributes(definition: Any) -> tuple[tuple[str, Any], ...]:
    return tuple((item.name, item.value) for item in getattr(definition, "attributes", ()))


def build_render_plan(
    scenario: CanonicalScenario,
    *,
    renderer_family: str,
    render_seed: int,
    split_manifest_hash: str | None = None,
) -> RenderPlan:
    """Build a provider request plan from canonical semantic definitions only."""

    if renderer_family not in RENDERER_FAMILY_BY_ID:
        raise ValueError(f"unknown renderer family: {renderer_family!r}")
    definitions = (*scenario.causal_evidence, *scenario.distractors)
    entries: list[EvidenceRenderPlan] = []
    for definition in definitions:
        source = getattr(definition, "source_component", None)
        if source is None:
            raise RenderContractError(
                f"evidence {definition.evidence_id} has no authoritative source component",
                category="semantic_consistency",
            )
        semantic_kind = definition.kind
        surface = KIND_TO_SURFACE.get(semantic_kind)
        if surface is None:
            raise RenderContractError(
                f"evidence {definition.evidence_id} has no 03C surface mapping",
                category="unsupported_evidence",
            )
        role = getattr(definition, "causal_role", "DISTRACTOR")
        entries.append(
            EvidenceRenderPlan(
                definition.evidence_id,
                semantic_kind,
                source,
                role,
                _attributes(definition),
                role == "MISSING_REQUIRED" and bool(getattr(definition, "may_be_removed", False)),
                (surface,),
                renderer_family,
            )
        )
    return RenderPlan(
        scenario.scenario_id,
        scenario_fingerprint(scenario),
        renderer_family,
        RENDERER_FAMILY_BY_ID[renderer_family].prompt_template_version,
        scenario.runtime.runtime_id,
        render_seed,
        tuple(entries),
        split_manifest_hash,
    )


def _normalise_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def forbidden_label_tokens(scenario: CanonicalScenario) -> tuple[str, ...]:
    values = list(FAILURE_MODE_IDS)
    values.extend(_FORBIDDEN_LABEL_FIELDS)
    values.extend(_SPLIT_NAMES)
    # Generic fault-domain words such as "database" or "dependency" are
    # legitimate telemetry vocabulary.  The leakage boundary targets the
    # machine-readable diagnosis labels and internal control metadata instead.
    values.extend((scenario.root_cause_id or "",))
    values.extend(
        (
            scenario.expected_runbook_id or "",
            scenario.difficulty,
            scenario.case_type,
            scenario.answerability,
            scenario.provenance.archetype_id,
            scenario.provenance.generator_family,
            scenario.provenance.template_family,
        )
    )
    return tuple(sorted({_normalise_label(value) for value in values if value}))


def check_label_leakage(scenario: CanonicalScenario, value: Any) -> None:
    """Reject machine labels in raw provider output or rendered surface text."""

    haystack = _normalise_label(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))
    leaked = [token for token in forbidden_label_tokens(scenario) if token and token in haystack]
    if leaked:
        raise RenderContractError(f"rendered telemetry leaks forbidden labels: {sorted(leaked)}", category="label_leakage")


def _parse_provider_payload(payload: Any, plan: RenderPlan) -> list[TelemetryObservation]:
    if not isinstance(payload, Mapping) or set(payload) != {"observations"}:
        raise RenderContractError("provider output must contain only observations", category="malformed_output")
    observations = payload["observations"]
    if not isinstance(observations, list):
        raise RenderContractError("provider observations must be a list", category="malformed_output")
    seen: set[str] = set()
    parsed: list[TelemetryObservation] = []
    by_id = {item.evidence_id: item for item in plan.entries}
    for item in observations:
        if not isinstance(item, Mapping):
            raise RenderContractError("provider observation must be an object", category="malformed_output")
        allowed = {"evidence_id", "kind", "component_id", "timestamp_offset_seconds", "text", "value"}
        if set(item) - allowed or not {"evidence_id", "kind", "component_id", "timestamp_offset_seconds", "text"}.issubset(item):
            raise RenderContractError("provider observation has malformed fields", category="malformed_output")
        evidence_id = item["evidence_id"]
        if not isinstance(evidence_id, str) or evidence_id in seen:
            raise RenderContractError("provider output contains duplicate evidence IDs", category="malformed_output")
        seen.add(evidence_id)
        entry = by_id.get(evidence_id)
        if entry is None:
            if evidence_id in plan.all_evidence_ids:
                raise RenderContractError("intentionally missing evidence reappeared", category="missing_evidence")
            raise RenderContractError("provider invented an unsupported evidence ID", category="unsupported_evidence")
        surface = item["kind"]
        if surface not in entry.permitted_surface_types:
            raise RenderContractError("provider changed the permitted telemetry surface", category="semantic_consistency")
        if item["component_id"] != entry.source_component:
            raise RenderContractError("provider changed the authoritative source component", category="semantic_consistency")
        offset = item["timestamp_offset_seconds"]
        if not isinstance(offset, int) or isinstance(offset, bool) or not 0 <= offset <= 86400:
            raise RenderContractError("timestamp offset is invalid", category="malformed_output")
        text = item["text"]
        if not isinstance(text, str) or not text.strip():
            raise RenderContractError("observation text must be non-empty", category="malformed_output")
        timestamp = (_BASE_TIMESTAMP + timedelta(seconds=offset)).isoformat().replace("+00:00", "Z")
        parsed.append(
            TelemetryObservation(
                evidence_id,
                SURFACE_TO_KIND[surface],
                timestamp,
                entry.source_component,
                text,
                item.get("value"),
            )
        )
    actual = {item.evidence_id for item in parsed}
    missing = plan.expected_evidence_ids - actual
    if missing:
        raise RenderContractError(f"required semantic evidence was not rendered: {sorted(missing)}", category="missing_evidence")
    if actual & {item.evidence_id for item in plan.entries if item.may_be_omitted}:
        raise RenderContractError("intentionally missing evidence reappeared", category="missing_evidence")
    return parsed


def render_provider_payload(
    scenario: CanonicalScenario,
    plan: RenderPlan,
    payload: Any,
    *,
    provider: str,
    model_identifier: str,
    split_manifest_hash: str | None = None,
) -> RenderedTelemetry:
    """Convert and validate one provider JSON payload without semantic repair."""

    if plan.scenario_id != scenario.scenario_id or plan.canonical_scenario_fingerprint != scenario_fingerprint(scenario):
        raise RenderContractError("render plan does not match canonical scenario", category="semantic_consistency")
    check_label_leakage(scenario, payload)
    observations = tuple(_parse_provider_payload(payload, plan))
    rendered = RenderedTelemetry(
        scenario.scenario_id,
        TELEMETRY_SCHEMA_VERSION,
        observations,
        RendererMetadata(provider, model_identifier, plan.prompt_template_version, plan.render_seed),
        scenario_fingerprint(scenario),
        split_manifest_hash=split_manifest_hash,
    )
    validate_rendered_telemetry(scenario, rendered, expected_split_manifest_hash=split_manifest_hash)
    check_label_leakage(scenario, rendered.to_dict(include_hashes=False))
    return RenderedTelemetry(
        rendered.scenario_id,
        rendered.schema_version,
        rendered.observations,
        rendered.renderer,
        rendered.canonical_scenario_hash,
        rendered_telemetry_fingerprint(rendered),
        rendered.split_manifest_hash,
    )


def build_render_prompt(plan: RenderPlan) -> str:
    family = RENDERER_FAMILY_BY_ID[plan.renderer_family]
    provider_view = plan.to_provider_dict()
    return (
        "You are a telemetry renderer, not an incident diagnostician.\n"
        "Return JSON with exactly one key, observations. Render only the declared semantic observations.\n"
        "Never give a diagnosis, root-cause label, remediation, answerability, or expected output.\n"
        "Never add evidence, change evidence IDs/components/kinds, make missing evidence reappear, "
        "or turn distractors into causal evidence.\n"
        f"Rendering contract family: {family.family_id}; style: {family.style}.\n"
        f"Runtime guidance: {_RUNTIME_HINTS.get(plan.runtime_id, 'use only supplied runtime-compatible terminology')}.\n"
        "Provider input (semantic evidence only):\n"
        + json.dumps(provider_view, ensure_ascii=False, sort_keys=True, indent=2)
    )
