"""Deterministic comparison and selection for the frozen Experiment 04A study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import sha256_file
from .verify import verify


RULE_VERSION = "e04a-selection-v1"
VARIANT_ORDER = ("025pct", "050pct", "075pct-retry-01", "100pct")
FRACTIONS = {"025pct": 0.25, "050pct": 0.50, "075pct-retry-01": 0.75, "100pct": 1.0}
QUALITY_METRICS = ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(value: Any) -> float:
    if isinstance(value, Mapping):
        value = value["rate"]
    return float(value)


def _count(value: Any) -> int:
    if isinstance(value, Mapping):
        return int(value["count"])
    raise ValueError("metric does not contain a count")


def _family_metrics(metrics: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for family, values in sorted(metrics["per_failure_family"].items()):
        result[family] = {
            "diagnosis_exact_match": float(values["diagnosis_exact_match"]),
            "resolution_exact_match": float(values["resolution_exact_match"]),
            "failure_mode_f1": float(values["failure_mode_f1"]["f1"]),
            "action_accuracy": float(values["action_accuracy"]),
            "json_compliance": float(values["json_compliance"]),
        }
    return result


def comparison_row(run_root: str | Path, variant: str) -> dict[str, Any]:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"unknown E04-A variant: {variant}")
    root = Path(run_root)
    manifest = _read_json(root / "manifest.json")
    evaluation = _read_json(root / "evaluation.json")
    summary = _read_json(root / "training_summary.json")
    subset_label = "075pct" if variant == "075pct-retry-01" else variant
    subset_manifest = _read_json(Path("data/experiment_04a/subsets") / subset_label / "subset_manifest.json")
    metrics = evaluation["metrics"]
    verification = verify(root, offline=True)
    if verification["summary"]["status"] != "PASS":
        raise ValueError(f"E04-A evidence verification failed for {variant}")
    if manifest["selection"]["source"] != "validation":
        raise ValueError(f"E04-A checkpoint selection is not validation-only for {variant}")
    return {
        "variant": variant,
        "fraction": FRACTIONS[variant],
        "example_count": int(subset_manifest["example_count"]),
        "subset_hash": subset_manifest["subset_hash"],
        "dataset_fingerprint": manifest["data"]["train"],
        "resolved_config_hash": manifest["config"]["sha256"],
        "model": manifest["model"],
        "selected_checkpoint": manifest["selection"]["checkpoint"],
        "actual_steps": manifest["training"]["actual_steps"],
        "configured_steps": manifest["training"]["configured_steps"],
        "stop_reason": manifest["training"]["stop_reason"],
        "metrics": {
            "diagnosis_exact_match": {"count": _count(metrics["diagnosis_exact_match"]), "rate": _rate(metrics["diagnosis_exact_match"])},
            "resolution_exact_match": {"count": _count(metrics["resolution_exact_match"]), "rate": _rate(metrics["resolution_exact_match"])},
            "failure_mode_macro_f1": float(metrics["failure_mode_macro_f1"]),
            "culprit_accuracy": {"count": _count(metrics["culprit_accuracy"]), "rate": _rate(metrics["culprit_accuracy"])},
            "action_accuracy": {"count": _count(metrics["recommended_action_accuracy"]), "rate": _rate(metrics["recommended_action_accuracy"])},
            "evidence_f1": float(metrics["evidence"]["f1"]),
            "valid_json": float(metrics["json_valid_rate"]),
            "strict_json": {"count": _count(metrics["json_compliance"]), "rate": _rate(metrics["json_compliance"])},
        },
        "failure_family_metrics": _family_metrics(metrics),
        "cost": {
            "training_examples_processed": int(summary["examples_processed"]),
            "input_tokens_processed": int(summary["input_tokens_processed"]),
            "supervised_tokens_processed": int(summary["supervised_tokens_processed"]),
            "wall_clock_training_seconds": float(summary["wall_clock_training_seconds"]),
            "validation_wall_clock_seconds": float(summary["validation_wall_clock_seconds"]),
            "peak_allocated_vram_gib": float(summary["peak_vram"]["allocated_gib"]),
            "peak_reserved_vram_gib": float(summary["peak_vram"]["reserved_gib"]),
            "trainable_parameters": int(_read_json(root / "parameter_provenance.json")["trainable_parameters"]),
        },
        "evidence": {
            "verification": verification["summary"]["status"],
            "artifact_hashes": _read_json(root / "artifact_hashes.json")["artifacts"],
            "artifact_hashes_sha256": sha256_file(root / "artifact_hashes.json"),
            "raw_predictions_preserved": (root / "predictions.jsonl").is_file(),
            "adapter_weights_local_only": True,
        },
    }


def select_variants(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen validation-only rule and return selection provenance."""

    if [row["variant"] for row in rows] != list(VARIANT_ORDER):
        raise ValueError("E04-A rows must contain the four variants in frozen order")
    for row in rows:
        if row["evidence"]["verification"] != "PASS":
            raise ValueError(f"cannot select an unverified variant: {row['variant']}")
    reference = max(
        rows,
        key=lambda row: tuple(float(row["metrics"][metric]["rate"] if metric != "failure_mode_macro_f1" else row["metrics"][metric]) for metric in QUALITY_METRICS),
    )
    reference_metrics = {
        metric: float(reference["metrics"][metric]["rate"] if metric != "failure_mode_macro_f1" else reference["metrics"][metric])
        for metric in QUALITY_METRICS
    }
    eligible: list[str] = []
    rejection_reasons: dict[str, list[str]] = {}
    for row in rows:
        reasons: list[str] = []
        for metric in QUALITY_METRICS:
            value = float(row["metrics"][metric]["rate"] if metric != "failure_mode_macro_f1" else row["metrics"][metric])
            if value + 1e-12 < reference_metrics[metric] - 0.01:
                reasons.append(f"{metric} more than 1.0 percentage point below reference")
        reference_families = reference["failure_family_metrics"]
        for family, values in row["failure_family_metrics"].items():
            if values["failure_mode_f1"] + 1e-12 < reference_families[family]["failure_mode_f1"] - 0.05:
                reasons.append(f"{family} failure-mode F1 more than 5.0 percentage points below reference")
        if row["metrics"]["valid_json"] + 1e-12 < reference["metrics"]["valid_json"] - 0.01 or row["metrics"]["strict_json"]["rate"] + 1e-12 < reference["metrics"]["strict_json"]["rate"] - 0.01:
            reasons.append("critical schema-validity regression")
        if reasons:
            rejection_reasons[row["variant"]] = reasons
        else:
            eligible.append(row["variant"])
    if not eligible:
        raise ValueError("no E04-A variant satisfies the frozen rule")
    selected = min((row for row in rows if row["variant"] in eligible), key=lambda row: row["fraction"])
    return {
        "rule_version": RULE_VERSION,
        "reference_variant": reference["variant"],
        "reference_metrics": reference_metrics,
        "thresholds": {
            "primary_metrics_tolerance_percentage_points": 1.0,
            "failure_family_tolerance_percentage_points": 5.0,
            "schema_validity_tolerance_percentage_points": 1.0,
        },
        "eligible_variants": eligible,
        "rejected_variants": rejection_reasons,
        "selected_variant": selected["variant"],
        "selected_fraction": selected["fraction"],
        "selected_subset_hash": selected["subset_hash"],
        "selection_basis": "smallest fraction satisfying the predeclared validation-only quality rule",
        "benchmark_used_for_selection": False,
        "e03_used_for_selection": False,
    }
