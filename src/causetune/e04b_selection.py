"""Deterministic comparison and selection for the frozen Experiment 04B study."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import sha256_file
from .verify import verify


RULE_VERSION = "e04b-selection-v1"
VARIANT_ORDER = ("r8", "r16", "r32")
RANKS = {"r8": 8, "r16": 16, "r32": 32}
QUALITY_METRICS = ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _rate(value: Any) -> float:
    return float(value["rate"] if isinstance(value, Mapping) else value)


def _families(metrics: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    return {
        family: {
            "diagnosis_exact_match": float(values["diagnosis_exact_match"]),
            "resolution_exact_match": float(values["resolution_exact_match"]),
            "failure_mode_f1": float(values["failure_mode_f1"]["f1"]),
            "action_accuracy": float(values["action_accuracy"]),
            "json_compliance": float(values["json_compliance"]),
        }
        for family, values in sorted(metrics["per_failure_family"].items())
    }


def comparison_row(run_root: str | Path, variant: str) -> dict[str, Any]:
    if variant not in VARIANT_ORDER:
        raise ValueError(f"unknown E04-B variant: {variant}")
    root = Path(run_root)
    manifest = _read_json(root / "manifest.json")
    evaluation = _read_json(root / "evaluation.json")
    summary = _read_json(root / "training_summary.json")
    config = _read_json(root / "resolved_config.json")
    metrics = evaluation["metrics"]
    verification = verify(root, offline=True)
    if verification["summary"]["status"] != "PASS":
        raise ValueError(f"E04-B evidence verification failed for {variant}")
    if manifest["selection"]["source"] != "validation":
        raise ValueError(f"E04-B checkpoint selection is not validation-only for {variant}")
    return {
        "variant": variant,
        "rank": RANKS[variant],
        "alpha": int(config["training"]["lora"]["alpha"]),
        "data_fraction": 0.25,
        "example_count": 600,
        "subset_hash": "eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56",
        "dataset_fingerprint": manifest["data"]["train"],
        "validation_fingerprint": manifest["data"]["validation"],
        "resolved_config_hash": manifest["config"]["sha256"],
        "model": manifest["model"],
        "selected_checkpoint": manifest["selection"]["checkpoint"],
        "actual_steps": manifest["training"]["actual_steps"],
        "configured_steps": manifest["training"]["configured_steps"],
        "stop_reason": manifest["training"]["stop_reason"],
        "metrics": {
            "diagnosis_exact_match": {"count": int(metrics["diagnosis_exact_match"]["count"]), "rate": _rate(metrics["diagnosis_exact_match"])},
            "resolution_exact_match": {"count": int(metrics["resolution_exact_match"]["count"]), "rate": _rate(metrics["resolution_exact_match"])},
            "failure_mode_macro_f1": float(metrics["failure_mode_macro_f1"]),
            "culprit_accuracy": {"count": int(metrics["culprit_accuracy"]["count"]), "rate": _rate(metrics["culprit_accuracy"])},
            "action_accuracy": {"count": int(metrics["recommended_action_accuracy"]["count"]), "rate": _rate(metrics["recommended_action_accuracy"])},
            "evidence_f1": float(metrics["evidence"]["f1"]),
            "valid_json": float(metrics["json_valid_rate"]),
            "strict_json": {"count": int(metrics["json_compliance"]["count"]), "rate": _rate(metrics["json_compliance"])},
        },
        "failure_family_metrics": _families(metrics),
        "cost": {
            "trainable_parameters": int(_read_json(root / "parameter_provenance.json")["trainable_parameters"]),
            "training_examples_processed": int(summary["examples_processed"]),
            "input_tokens_processed": int(summary["input_tokens_processed"]),
            "supervised_tokens_processed": int(summary["supervised_tokens_processed"]),
            "wall_clock_training_seconds": float(summary["wall_clock_training_seconds"]),
            "validation_wall_clock_seconds": float(summary["validation_wall_clock_seconds"]),
            "peak_allocated_vram_gib": float(summary["peak_vram"]["allocated_gib"]),
            "peak_reserved_vram_gib": float(summary["peak_vram"]["reserved_gib"]),
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
    if [row["variant"] for row in rows] != list(VARIANT_ORDER):
        raise ValueError("E04-B rows must contain r8, r16 and r32 in frozen order")
    if any(row["evidence"]["verification"] != "PASS" for row in rows):
        raise ValueError("cannot select an unverified E04-B variant")
    reference = max(rows, key=lambda row: tuple(_rate(row["metrics"][metric]) for metric in QUALITY_METRICS))
    reference_metrics = {metric: _rate(reference["metrics"][metric]) for metric in QUALITY_METRICS}
    eligible: list[str] = []
    rejected: dict[str, list[str]] = {}
    for row in rows:
        reasons: list[str] = []
        for metric in QUALITY_METRICS:
            if _rate(row["metrics"][metric]) + 1e-12 < reference_metrics[metric] - 0.01:
                reasons.append(f"{metric} more than 1.0 percentage point below reference")
        for family, values in row["failure_family_metrics"].items():
            if values["failure_mode_f1"] + 1e-12 < reference["failure_family_metrics"][family]["failure_mode_f1"] - 0.05:
                reasons.append(f"{family} failure-mode F1 more than 5.0 percentage points below reference")
        if row["metrics"]["valid_json"] + 1e-12 < reference["metrics"]["valid_json"] - 0.01 or row["metrics"]["strict_json"]["rate"] + 1e-12 < reference["metrics"]["strict_json"]["rate"] - 0.01:
            reasons.append("critical schema-validity regression")
        if reasons:
            rejected[row["variant"]] = reasons
        else:
            eligible.append(row["variant"])
    if not eligible:
        raise ValueError("no E04-B rank satisfies the frozen rule")
    selected = min((row for row in rows if row["variant"] in eligible), key=lambda row: row["rank"])
    return {
        "rule_version": RULE_VERSION,
        "reference_variant": reference["variant"],
        "reference_metrics": reference_metrics,
        "thresholds": {"primary_metrics_tolerance_percentage_points": 1.0, "failure_family_tolerance_percentage_points": 5.0, "schema_validity_tolerance_percentage_points": 1.0},
        "eligible_variants": eligible,
        "rejected_variants": rejected,
        "selected_variant": selected["variant"],
        "selected_rank": selected["rank"],
        "selected_trainable_parameters": selected["cost"]["trainable_parameters"],
        "selection_basis": "smallest rank satisfying the predeclared validation-only quality rule",
        "benchmark_used_for_selection": False,
        "e03_used_for_selection": False,
    }
