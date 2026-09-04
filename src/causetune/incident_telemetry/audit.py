"""Fail-closed integrity audit for Experiment 03 canonical data and renderings."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from .evaluation import parse_diagnostic_output, validate_diagnostic_output
from .fingerprints import rendered_content_fingerprint, scenario_content_fingerprint
from .generator import CounterfactualPair, validate_counterfactual_pair, validate_generated_scenario
from .models import CanonicalScenario, DiagnosticOutput, RenderedTelemetry, validate_scenario
from .ontology import FAILURE_MODE_BY_ID, validate_ontology
from .renderer import validate_rendered_telemetry
from .splits import SplitManifest, validate_split_manifest


class DatasetIntegrityError(ValueError):
    """Raised when the 03A audit finds an integrity violation."""

    def __init__(self, report: Mapping[str, Any]):
        self.report = dict(report)
        super().__init__("Experiment 03A dataset audit failed: " + "; ".join(self.report["errors"]))


def _duplicate_values(values: Sequence[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def audit_dataset(
    scenarios: Sequence[CanonicalScenario],
    *,
    split_manifest: SplitManifest | None = None,
    rendered_telemetry: Sequence[RenderedTelemetry] = (),
    outputs: Mapping[str, DiagnosticOutput | str] | None = None,
    archetypes: Mapping[str, Any] | None = None,
    counterfactual_pairs: Sequence[CounterfactualPair] = (),
    fail_closed: bool = True,
) -> dict[str, Any]:
    """Audit canonical scenarios, optional renderings, split assignments, and outputs.

    The default is fail-closed: any violation raises DatasetIntegrityError.  A
    non-raising report is available for inspection with ``fail_closed=False``.
    """

    errors: list[str] = []
    try:
        validate_ontology()
    except ValueError as exc:
        errors.append(str(exc))

    scenario_ids = [getattr(item, "scenario_id", "<invalid>") for item in scenarios]
    duplicate_ids = _duplicate_values(scenario_ids)
    if duplicate_ids:
        errors.append(f"duplicate scenario IDs: {duplicate_ids}")
    content_fingerprints: list[str] = []
    for scenario in scenarios:
        if not isinstance(scenario, CanonicalScenario):
            errors.append("scenario collection contains a non-CanonicalScenario value")
            continue
        try:
            validate_scenario(scenario)
            if archetypes is not None:
                archetype = archetypes.get(scenario.provenance.archetype_id)
                if archetype is None:
                    raise ValueError("scenario archetype is not in the supplied catalog")
                validate_generated_scenario(scenario, archetype)
            content_fingerprints.append(scenario_content_fingerprint(scenario))
        except (TypeError, ValueError) as exc:
            errors.append(f"{scenario.scenario_id}: schema invalidity: {exc}")
    duplicate_content = _duplicate_values(content_fingerprints)
    if duplicate_content:
        errors.append(f"duplicate canonical scenario content: {duplicate_content}")

    if split_manifest is not None:
        try:
            split_report = validate_split_manifest(split_manifest, scenarios)
        except ValueError as exc:
            errors.append(str(exc))
            split_report = {"status": "fail"}
    else:
        split_report = {"status": "not_provided"}

    for pair in counterfactual_pairs:
        try:
            source = next(item for item in scenarios if item.scenario_id == pair.source_scenario_id)
            counterfactual = next(item for item in scenarios if item.scenario_id == pair.counterfactual_scenario_id)
            validate_counterfactual_pair(source, counterfactual, pair)
        except (StopIteration, ValueError) as exc:
            errors.append(f"counterfactual pair {pair.pair_id}: {exc}")

    valid_scenarios = [item for item in scenarios if isinstance(item, CanonicalScenario)]
    scenario_by_id = {scenario.scenario_id: scenario for scenario in valid_scenarios}
    rendered_ids = [getattr(item, "scenario_id", "<invalid>") for item in rendered_telemetry]
    if _duplicate_values(rendered_ids):
        errors.append(f"duplicate rendered scenario IDs: {_duplicate_values(rendered_ids)}")
    rendered_hashes: list[str] = []
    rendered_by_id: dict[str, RenderedTelemetry] = {}
    for rendered in rendered_telemetry:
        if not isinstance(rendered, RenderedTelemetry):
            errors.append("rendered telemetry collection contains a non-RenderedTelemetry value")
            continue
        scenario = scenario_by_id.get(rendered.scenario_id)
        if scenario is None:
            errors.append(f"rendered telemetry references unknown scenario: {rendered.scenario_id}")
            continue
        try:
            validate_rendered_telemetry(scenario, rendered)
            rendered_hashes.append(rendered_content_fingerprint(rendered))
            rendered_by_id[rendered.scenario_id] = rendered
        except ValueError as exc:
            errors.append(f"{rendered.scenario_id}: rendered telemetry invalid: {exc}")
    duplicate_rendered = _duplicate_values(rendered_hashes)
    if duplicate_rendered:
        errors.append(f"exact rendered duplicates: {duplicate_rendered}")

    output_report: dict[str, Any] = {"status": "not_provided"}
    if outputs is not None:
        if set(outputs) != set(scenario_by_id):
            errors.append("outputs do not cover exactly the canonical scenarios")
        output_errors: list[str] = []
        for scenario_id, value in outputs.items():
            scenario = scenario_by_id.get(scenario_id)
            if scenario is None:
                continue
            if isinstance(value, str):
                parsed, category, _json_valid = parse_diagnostic_output(value)
                if parsed is None:
                    output_errors.append(f"{scenario_id}: output {category}")
                    continue
            elif isinstance(value, DiagnosticOutput):
                parsed = value
            else:
                output_errors.append(f"{scenario_id}: output has invalid type")
                continue
            rendered = rendered_by_id.get(scenario_id)
            if rendered is None:
                output_errors.append(f"{scenario_id}: output cannot be reference-validated without telemetry")
                continue
            try:
                validate_diagnostic_output(parsed, scenario, rendered)
                if scenario.answerability == "INSUFFICIENT_EVIDENCE" and not parsed.needs_more_data:
                    output_errors.append(f"{scenario_id}: insufficient-evidence case was diagnosed")
                if scenario.answerability == "ANSWERABLE" and parsed.needs_more_data:
                    output_errors.append(f"{scenario_id}: answerable case was abstained")
                if scenario.answerability == "HEALTHY_CONTROL" and (
                    parsed.needs_more_data or parsed.root_cause is not None or parsed.affected_component is not None
                ):
                    output_errors.append(f"{scenario_id}: healthy-control output contradicts answerability")
            except ValueError as exc:
                output_errors.append(f"{scenario_id}: output contradiction: {exc}")
        errors.extend(output_errors)
        output_report = {"status": "fail" if output_errors else "pass", "error_count": len(output_errors)}

    family_counts = Counter(scenario.root_cause_id or "HEALTHY_CONTROL" for scenario in valid_scenarios)
    domain_counts = Counter(scenario.fault_domain or "NONE" for scenario in valid_scenarios)
    difficulty_counts = Counter(scenario.difficulty for scenario in valid_scenarios)
    report = {
        "status": "fail" if errors else "pass",
        "scenario_count": len(scenarios),
        "rendered_telemetry_count": len(rendered_telemetry),
        "counterfactual_pair_count": len(counterfactual_pairs),
        "errors": errors,
        "split": split_report,
        "outputs": output_report,
        "distributions": {
            "root_cause_id": dict(sorted(family_counts.items())),
            "fault_domain": dict(sorted(domain_counts.items())),
            "difficulty": dict(sorted(difficulty_counts.items())),
            "failure_mode_catalog_size": len(FAILURE_MODE_BY_ID),
        },
        "duplicate_canonical_content_count": len(duplicate_content),
        "duplicate_rendered_count": len(duplicate_rendered),
    }
    if errors and fail_closed:
        raise DatasetIntegrityError(report)
    return report
