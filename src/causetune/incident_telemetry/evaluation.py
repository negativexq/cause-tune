"""Deterministic evaluation contract for future Experiment 03 runs."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from .models import CanonicalScenario, DiagnosticOutput, RenderedTelemetry
from .ontology import FAILURE_MODE_SET


DIAGNOSTIC_OUTPUT_KEYS = frozenset(
    {
        "root_cause",
        "affected_component",
        "evidence_ids",
        "remediation_runbook_id",
        "needs_more_data",
        "required_evidence",
    }
)


def parse_diagnostic_output(
    text: str,
) -> tuple[DiagnosticOutput | None, str, bool]:
    """Parse output without repair; return object, category, and JSON validity."""

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid JSON", False
    if not isinstance(value, dict):
        return None, "invalid schema", True
    if set(value) != DIAGNOSTIC_OUTPUT_KEYS:
        return None, "unexpected keys", True
    try:
        fields = dict(value)
        if not isinstance(fields["evidence_ids"], list) or not all(
            isinstance(item, str) for item in fields["evidence_ids"]
        ):
            return None, "invalid evidence_ids", True
        if not isinstance(fields["required_evidence"], list) or not all(
            isinstance(item, str) for item in fields["required_evidence"]
        ):
            return None, "invalid required_evidence", True
        for field in ("root_cause", "affected_component", "remediation_runbook_id"):
            if fields[field] is not None and not isinstance(fields[field], str):
                return None, f"invalid {field}", True
        output = DiagnosticOutput(
            root_cause=fields["root_cause"],
            affected_component=fields["affected_component"],
            evidence_ids=tuple(fields["evidence_ids"]),
            remediation_runbook_id=fields["remediation_runbook_id"],
            needs_more_data=fields["needs_more_data"],
            required_evidence=tuple(fields["required_evidence"]),
        )
    except (KeyError, TypeError, ValueError):
        return None, "invalid schema", True
    return output, "valid JSON", True


def validate_diagnostic_output(
    output: DiagnosticOutput,
    scenario: CanonicalScenario,
    rendered: RenderedTelemetry,
) -> None:
    """Validate semantic references without deciding whether the model is correct."""

    if not isinstance(output, DiagnosticOutput):
        raise ValueError("output must be DiagnosticOutput")
    if rendered.scenario_id != scenario.scenario_id:
        raise ValueError("rendered telemetry belongs to a different scenario")
    available = {item.evidence_id for item in rendered.observations}
    declared = scenario.declared_evidence_ids
    if not set(output.evidence_ids).issubset(available):
        raise ValueError("diagnostic output references evidence absent from rendered telemetry")
    if not set(output.required_evidence).issubset(declared):
        raise ValueError("required_evidence references an undeclared evidence ID")
    if output.root_cause is not None and output.root_cause not in FAILURE_MODE_SET:
        raise ValueError("diagnostic output contains an unknown root cause ID")
    components = {item.component_id for item in scenario.topology.components}
    if output.affected_component is not None and output.affected_component not in components:
        raise ValueError("diagnostic output contains an unknown component")


def _rate(correct: int, total: int) -> dict[str, float | int]:
    return {"count": correct, "total": total, "rate": correct / total if total else 0.0}


def _metric_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerable"]]
    incomplete = [row for row in rows if row["insufficient_evidence"]]
    root_correct = sum(row["root_cause_correct"] for row in answerable)
    component_correct = sum(row["component_correct"] for row in answerable)
    diagnosis_correct = sum(row["diagnosis_joint_correct"] for row in answerable)
    runbook_correct = sum(row["runbook_correct"] for row in answerable)
    tp = sum(row["evidence_tp"] for row in answerable)
    fp = sum(row["evidence_fp"] for row in answerable)
    fn = sum(row["evidence_fn"] for row in answerable)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    predicted_abstentions = sum(row["predicted_abstention"] for row in rows)
    true_abstentions = sum(row["insufficient_evidence"] for row in rows)
    abstention_tp = sum(row["predicted_abstention"] and row["insufficient_evidence"] for row in rows)
    abstention_precision = abstention_tp / predicted_abstentions if predicted_abstentions else 0.0
    abstention_recall = abstention_tp / true_abstentions if true_abstentions else 0.0
    abstention_f1 = (
        2 * abstention_precision * abstention_recall / (abstention_precision + abstention_recall)
        if abstention_precision + abstention_recall
        else 0.0
    )
    false_diagnoses = sum(row["false_diagnosis_on_insufficient_evidence"] for row in incomplete)
    return {
        "count": len(rows),
        "answerable_count": len(answerable),
        "insufficient_evidence_count": len(incomplete),
        "root_cause_exact_accuracy": _rate(root_correct, len(answerable)),
        "affected_component_accuracy": _rate(component_correct, len(answerable)),
        "diagnosis_joint_exact": _rate(diagnosis_correct, len(answerable)),
        "evidence_precision": precision,
        "evidence_recall": recall,
        "evidence_f1": f1,
        "remediation_runbook_accuracy": _rate(runbook_correct, len(answerable)),
        "valid_json_rate": _rate(sum(row["json_valid"] for row in rows), len(rows)),
        "schema_valid_rate": _rate(sum(row["schema_valid"] for row in rows), len(rows)),
        "abstention_precision": abstention_precision,
        "abstention_recall": abstention_recall,
        "abstention_f1": abstention_f1,
        "false_diagnosis_rate_on_insufficient_evidence": _rate(false_diagnoses, len(incomplete)),
    }


def evaluate_diagnostic_outputs(
    scenarios: Sequence[CanonicalScenario],
    outputs: Mapping[str, DiagnosticOutput | str | Mapping[str, Any]],
    rendered_by_scenario: Mapping[str, RenderedTelemetry] | None = None,
    evaluation_split_by_scenario: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Evaluate structured outputs and return aggregate plus explicit slices.

    Primary answer-quality metrics use answerable scenarios as their denominator.
    Abstention metrics use all cases, while false-diagnosis rate uses only
    INCOMPLETE/INSUFFICIENT_EVIDENCE cases.  No semantic judge is involved.
    """

    if set(outputs) != {scenario.scenario_id for scenario in scenarios}:
        raise ValueError("outputs must cover exactly the supplied scenario IDs")
    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        value = outputs[scenario.scenario_id]
        parsed: DiagnosticOutput | None
        category: str
        json_valid: bool
        if isinstance(value, str):
            parsed, category, json_valid = parse_diagnostic_output(value)
        elif isinstance(value, DiagnosticOutput):
            parsed, category, json_valid = value, "valid JSON", True
        elif isinstance(value, Mapping):
            try:
                parsed, category, json_valid = parse_diagnostic_output(json.dumps(dict(value)))
            except (TypeError, ValueError):
                parsed, category, json_valid = None, "invalid schema", True
        else:
            parsed, category, json_valid = None, "invalid schema", True
        rendered = rendered_by_scenario.get(scenario.scenario_id) if rendered_by_scenario else None
        if parsed is not None and rendered is not None:
            try:
                validate_diagnostic_output(parsed, scenario, rendered)
                schema_valid = True
            except ValueError:
                schema_valid = False
        else:
            schema_valid = parsed is not None
        expected_evidence = scenario.required_evidence_ids
        predicted_evidence = set(parsed.evidence_ids) if parsed else set()
        row = {
            "scenario_id": scenario.scenario_id,
            "parsed": parsed,
            "parse_category": category,
            "json_valid": json_valid,
            "schema_valid": schema_valid,
            "answerable": scenario.answerability == "ANSWERABLE",
            "insufficient_evidence": scenario.answerability == "INSUFFICIENT_EVIDENCE",
            "predicted_abstention": bool(parsed and parsed.needs_more_data),
            "root_cause_correct": bool(parsed and parsed.root_cause == scenario.root_cause_id),
            "component_correct": bool(parsed and parsed.affected_component == scenario.affected_component),
            "runbook_correct": bool(parsed and parsed.remediation_runbook_id == scenario.expected_runbook_id),
            "diagnosis_joint_correct": bool(
                parsed
                and parsed.root_cause == scenario.root_cause_id
                and parsed.affected_component == scenario.affected_component
            ),
            "evidence_tp": len(expected_evidence & predicted_evidence),
            "evidence_fp": len(predicted_evidence - expected_evidence),
            "evidence_fn": len(expected_evidence - predicted_evidence),
            "false_diagnosis_on_insufficient_evidence": bool(parsed and not parsed.needs_more_data),
        }
        rows.append(row)

    by_slice: dict[str, dict[str, Any]] = {}
    slice_fields = (
        ("root_cause_family", lambda scenario: scenario.root_cause_family or "NONE"),
        ("fault_domain", lambda scenario: scenario.fault_domain or "NONE"),
        ("difficulty", lambda scenario: scenario.difficulty),
        ("topology_family", lambda scenario: scenario.topology.topology_family),
        ("runtime/ecosystem", lambda scenario: scenario.runtime.ecosystem),
        ("generator_family", lambda scenario: scenario.provenance.generator_family),
        ("template_family", lambda scenario: scenario.provenance.template_family),
        (
            "evaluation_split",
            lambda scenario: (evaluation_split_by_scenario or {}).get(scenario.scenario_id, "UNASSIGNED"),
        ),
    )
    scenario_by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    for field, selector in slice_fields:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            key = selector(scenario_by_id[row["scenario_id"]])
            groups.setdefault(key, []).append(row)
        by_slice[field] = {key: _metric_summary(group) for key, group in sorted(groups.items())}

    return {
        **_metric_summary(rows),
        "slices": by_slice,
        "parse_categories": {
            category: sum(row["parse_category"] == category for row in rows)
            for category in sorted({row["parse_category"] for row in rows})
        },
    }
