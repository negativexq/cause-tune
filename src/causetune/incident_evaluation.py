"""Strict deterministic evaluation for Experiment 02A incident diagnoses."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from .incident_benchmark import packet_evidence_ids
from .incident_taxonomy import ACTIONS, FAILURE_FAMILIES


DIAGNOSIS_KEYS = {"culprit_service", "failure_mode", "recommended_action", "evidence_ids"}


def parse_incident_diagnosis(
    text: str,
    present_components: Sequence[str],
    available_evidence_ids: Sequence[str],
) -> tuple[dict[str, Any] | None, str, bool]:
    """Parse without repairing output; return parsed object, category, JSON validity."""

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, "invalid JSON", False
    if not isinstance(parsed, dict):
        return None, "invalid schema", True
    if set(parsed) != DIAGNOSIS_KEYS:
        return None, "unexpected additional keys", True
    if not isinstance(parsed["culprit_service"], str) or parsed["culprit_service"] not in set(present_components):
        return None, "unknown culprit", True
    if not isinstance(parsed["failure_mode"], str) or parsed["failure_mode"] not in set(FAILURE_FAMILIES):
        return None, "unknown failure mode", True
    if not isinstance(parsed["recommended_action"], str) or parsed["recommended_action"] not in set(ACTIONS):
        return None, "unknown action", True
    evidence = parsed["evidence_ids"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        return None, "invalid evidence_ids", True
    if len(evidence) != len(set(evidence)) or not set(evidence).issubset(set(available_evidence_ids)):
        return None, "invalid evidence reference", True
    return dict(parsed), "valid JSON", True


def _f1_metrics(expected: Sequence[str], predicted: Sequence[str | None], labels: Sequence[str]) -> tuple[float, dict[str, dict[str, float | int]]]:
    per_class: dict[str, dict[str, float | int]] = {}
    for label in labels:
        tp = sum(actual == label and guess == label for actual, guess in zip(expected, predicted))
        fp = sum(actual != label and guess == label for actual, guess in zip(expected, predicted))
        fn = sum(actual == label and guess != label for actual, guess in zip(expected, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"support": sum(actual == label for actual in expected), "precision": precision, "recall": recall, "f1": f1}
    return sum(float(row["f1"]) for row in per_class.values()) / len(labels), per_class


def _confusion(expected: Sequence[str], predicted: Sequence[str | None], labels: Sequence[str]) -> dict[str, dict[str, int]]:
    matrix = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(expected, predicted):
        if actual in matrix and guess in matrix[actual]:
            matrix[actual][guess] += 1
    return matrix


def evaluate_incidents(
    inputs: Sequence[Mapping[str, Any]],
    ground_truth: Mapping[str, Mapping[str, Any]],
    raw_outputs: Mapping[str, str],
) -> dict[str, Any]:
    """Evaluate one slice or the complete benchmark with exact field accounting."""

    if len(inputs) != len(raw_outputs):
        raise ValueError("raw output count does not match incident count")
    rows: list[dict[str, Any]] = []
    for incident in inputs:
        incident_id = incident["incident_id"]
        truth = ground_truth[incident_id]
        parsed, category, json_valid = parse_incident_diagnosis(
            raw_outputs[incident_id],
            incident["metadata"]["present_components"],
            incident["metadata"]["evidence_ids"],
        )
        expected_evidence = set(truth["evidence_ids"])
        predicted_evidence = set(parsed["evidence_ids"]) if parsed else set()
        evidence_tp = len(expected_evidence & predicted_evidence)
        evidence_fp = len(predicted_evidence - expected_evidence)
        evidence_fn = len(expected_evidence - predicted_evidence)
        expected_culprit = truth["culprit_service"]
        expected_mode = truth["failure_mode"]
        expected_action = truth["recommended_action"]
        row = {
            "incident_id": incident_id,
            "slice": incident["slice"],
            "expected": {
                "culprit_service": expected_culprit,
                "failure_mode": expected_mode,
                "recommended_action": expected_action,
                "evidence_ids": truth["evidence_ids"],
            },
            "predicted": parsed,
            "raw_output": raw_outputs[incident_id],
            "json_valid": json_valid,
            "strict_schema_valid": parsed is not None,
            "parse_category": category,
            "culprit_correct": bool(parsed and parsed["culprit_service"] == expected_culprit),
            "failure_mode_correct": bool(parsed and parsed["failure_mode"] == expected_mode),
            "action_correct": bool(parsed and parsed["recommended_action"] == expected_action),
            "diagnosis_exact": bool(parsed and parsed["culprit_service"] == expected_culprit and parsed["failure_mode"] == expected_mode),
            "resolution_exact": bool(parsed and parsed["culprit_service"] == expected_culprit and parsed["failure_mode"] == expected_mode and parsed["recommended_action"] == expected_action),
            "evidence_tp": evidence_tp,
            "evidence_fp": evidence_fp,
            "evidence_fn": evidence_fn,
            "difficulty": truth["metadata"]["difficulty"],
            "failure_family": truth["metadata"]["failure_family"],
            "topology_family": truth["metadata"]["topology_family"],
            "red_herring": truth["metadata"]["red_herring"],
        }
        rows.append(row)
    expected_modes = [row["expected"]["failure_mode"] for row in rows]
    predicted_modes = [row["predicted"]["failure_mode"] if row["predicted"] else None for row in rows]
    macro_f1, per_family_f1 = _f1_metrics(expected_modes, predicted_modes, FAILURE_FAMILIES)
    total = len(rows)
    evidence_tp = sum(row["evidence_tp"] for row in rows)
    evidence_fp = sum(row["evidence_fp"] for row in rows)
    evidence_fn = sum(row["evidence_fn"] for row in rows)
    evidence_precision = evidence_tp / (evidence_tp + evidence_fp) if evidence_tp + evidence_fp else 0.0
    evidence_recall = evidence_tp / (evidence_tp + evidence_fn) if evidence_tp + evidence_fn else 0.0
    evidence_f1 = 2 * evidence_precision * evidence_recall / (evidence_precision + evidence_recall) if evidence_precision + evidence_recall else 0.0

    def rate(field: str) -> float:
        return sum(bool(row[field]) for row in rows) / total if total else 0.0

    per_family: dict[str, dict[str, Any]] = {}
    for family in FAILURE_FAMILIES:
        selected = [row for row in rows if row["failure_family"] == family]
        per_family[family] = {
            "count": len(selected),
            "diagnosis_exact_match": sum(row["diagnosis_exact"] for row in selected) / len(selected) if selected else 0.0,
            "resolution_exact_match": sum(row["resolution_exact"] for row in selected) / len(selected) if selected else 0.0,
            "culprit_accuracy": sum(row["culprit_correct"] for row in selected) / len(selected) if selected else 0.0,
            "failure_mode_accuracy": sum(row["failure_mode_correct"] for row in selected) / len(selected) if selected else 0.0,
            "action_accuracy": sum(row["action_correct"] for row in selected) / len(selected) if selected else 0.0,
            "json_compliance": sum(row["strict_schema_valid"] for row in selected) / len(selected) if selected else 0.0,
            "failure_mode_f1": per_family_f1[family],
        }
    return {
        "count": total,
        "diagnosis_exact_match": {"count": sum(row["diagnosis_exact"] for row in rows), "rate": rate("diagnosis_exact")},
        "resolution_exact_match": {"count": sum(row["resolution_exact"] for row in rows), "rate": rate("resolution_exact")},
        "culprit_accuracy": {"count": sum(row["culprit_correct"] for row in rows), "rate": rate("culprit_correct")},
        "failure_mode_accuracy": {"count": sum(row["failure_mode_correct"] for row in rows), "rate": rate("failure_mode_correct")},
        "failure_mode_macro_f1": macro_f1,
        "recommended_action_accuracy": {"count": sum(row["action_correct"] for row in rows), "rate": rate("action_correct")},
        "evidence": {
            "precision": evidence_precision,
            "recall": evidence_recall,
            "f1": evidence_f1,
            "true_positive": evidence_tp,
            "false_positive": evidence_fp,
            "false_negative": evidence_fn,
        },
        "json_compliance": {"count": sum(row["strict_schema_valid"] for row in rows), "rate": rate("strict_schema_valid")},
        "json_valid_rate": sum(row["json_valid"] for row in rows) / total if total else 0.0,
        "parse_categories": dict(Counter(row["parse_category"] for row in rows)),
        "failure_mode_confusion": _confusion(expected_modes, predicted_modes, FAILURE_FAMILIES),
        "per_failure_family": per_family,
        "predictions": rows,
    }


def _latest_changed_component(packet: str, components: Sequence[str]) -> str | None:
    in_changes = False
    latest: str | None = None
    for line in packet.splitlines():
        if line == "RECENT CHANGES":
            in_changes = True
            continue
        if in_changes and not line:
            break
        if in_changes and len(line) >= 5 and line[:2].isdigit() and line[2] == ":":
            latest = next((component for component in components if component in line), latest)
    return latest


def failure_patterns(
    inputs: Sequence[Mapping[str, Any]],
    evaluation: Mapping[str, Any],
) -> dict[str, int]:
    """Classify only mechanically observable failure transitions."""

    counts: Counter[str] = Counter()
    for incident, row in zip(inputs, evaluation["predictions"]):
        prediction = row["predicted"]
        if row["diagnosis_exact"] and not row["resolution_exact"]:
            counts["correct_diagnosis_wrong_action"] += 1
        if row["failure_mode_correct"] and not row["culprit_correct"]:
            counts["correct_family_wrong_culprit"] += 1
        if row["culprit_correct"] and not row["failure_mode_correct"]:
            counts["correct_culprit_wrong_family"] += 1
        if not row["diagnosis_exact"] and not row["json_valid"]:
            counts["json_or_schema_failure"] += 1
        if prediction:
            latest = _latest_changed_component(incident["incident_packet"], incident["metadata"]["present_components"])
            if latest and prediction["culprit_service"] == latest and not row["culprit_correct"]:
                counts["recent_deploy_or_change_bias"] += 1
            gateway_like = next((item for item in incident["metadata"]["present_components"] if item.endswith(("gateway", "edge", "api"))), None)
            if gateway_like and prediction["culprit_service"] == gateway_like and not row["culprit_correct"]:
                counts["symptom_service_bias"] += 1
        if row["red_herring"] and not row["diagnosis_exact"]:
            counts["distractor_selection"] += 1
    counts["total_failures"] = sum(not row["diagnosis_exact"] for row in evaluation["predictions"])
    return dict(counts)
