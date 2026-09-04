"""Provider-neutral renderer boundary for future telemetry generation.

03A defines the interface only.  No provider SDK, network call, or rendering
implementation belongs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .fingerprints import rendered_telemetry_fingerprint, scenario_fingerprint
from .models import CanonicalScenario, RenderedTelemetry


class RendererAuthorityError(ValueError):
    """Raised when a renderer attempts to cross the canonical-data boundary."""


@runtime_checkable
class TelemetryRenderer(Protocol):
    """Future renderer interface; implementations must not produce labels."""

    def render(self, scenario: CanonicalScenario) -> RenderedTelemetry:
        """Render observations for a canonical scenario without changing its truth."""


_FORBIDDEN_RENDERER_FIELDS = frozenset(
    {
        "root_cause_id",
        "root_cause",
        "fault_domain",
        "affected_component",
        "failure_mode",
        "expected_runbook_id",
        "answerability",
        "needs_more_data",
        "benchmark_labels",
    }
)


def validate_rendered_telemetry(
    scenario: CanonicalScenario,
    rendered: RenderedTelemetry,
    *,
    expected_split_manifest_hash: str | None = None,
) -> None:
    """Validate that rendered observations remain subordinate to the scenario."""

    if not isinstance(rendered, RenderedTelemetry):
        raise RendererAuthorityError("renderer must return RenderedTelemetry")
    if rendered.scenario_id != scenario.scenario_id:
        raise RendererAuthorityError("renderer changed scenario identity")
    if rendered.canonical_scenario_hash != scenario_fingerprint(scenario):
        raise RendererAuthorityError("renderer supplied a mismatched canonical scenario hash")
    if expected_split_manifest_hash is not None and rendered.split_manifest_hash != expected_split_manifest_hash:
        raise RendererAuthorityError("renderer supplied a mismatched split-manifest hash")

    component_ids = {item.component_id for item in scenario.topology.components}
    unknown_components = {
        item.component_id
        for item in rendered.observations
        if item.component_id is not None and item.component_id not in component_ids
    }
    if unknown_components:
        raise RendererAuthorityError(
            f"renderer referenced unsupported components: {sorted(unknown_components)}"
        )
    definitions = {
        item.evidence_id: item
        for item in (*scenario.causal_evidence, *scenario.distractors)
    }
    rendered_ids = {item.evidence_id for item in rendered.observations}
    unknown = rendered_ids - set(definitions)
    if unknown:
        raise RendererAuthorityError(f"renderer invented unsupported evidence IDs: {sorted(unknown)}")
    mismatched_kinds = [
        item.evidence_id
        for item in rendered.observations
        if definitions[item.evidence_id].kind != item.kind
    ]
    if mismatched_kinds:
        raise RendererAuthorityError(f"renderer changed evidence kind: {sorted(mismatched_kinds)}")
    missing_required = scenario.required_evidence_ids - rendered_ids
    if missing_required:
        allowed_omissions = {
            item.evidence_id
            for item in scenario.causal_evidence
            if item.may_be_removed
        }
        if scenario.case_type != "INCOMPLETE" or not missing_required.issubset(allowed_omissions):
            raise RendererAuthorityError(
                f"renderer removed required causal evidence without an explicit omission rule: {sorted(missing_required)}"
            )
    if rendered.rendered_telemetry_hash is not None:
        expected_hash = rendered_telemetry_fingerprint(rendered)
        if rendered.rendered_telemetry_hash != expected_hash:
            raise RendererAuthorityError("rendered telemetry fingerprint does not match its content")


def render_scenario(renderer: TelemetryRenderer, scenario: CanonicalScenario) -> RenderedTelemetry:
    """Invoke a provider-neutral renderer and enforce the authority boundary."""

    result = renderer.render(scenario)
    if isinstance(result, Mapping):
        forbidden = _FORBIDDEN_RENDERER_FIELDS & set(result)
        if forbidden:
            raise RendererAuthorityError(
                f"renderer output contains authoritative fields: {sorted(forbidden)}"
            )
    validate_rendered_telemetry(scenario, result)
    return result


render_telemetry = render_scenario
