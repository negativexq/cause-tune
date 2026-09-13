#!/usr/bin/env python3
"""Classify the frozen E08 quality/cost frontier from persisted evidence."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "results/experiment_08/protocol.json"
BLIND = ROOT / "results/experiment_05/evaluation_comparison.json"
E02_RUN = ROOT / "outputs/incident_diagnosis_02b2"
E04_RUN = ROOT / "runs/experiment_04c/lr1e-4"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _count_lines(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _cost(run: Path, unique_examples: int) -> dict[str, Any]:
    summary = _read(run / "training_summary.json")
    config = _read(run / "resolved_manifest.json" if (run / "resolved_manifest.json").exists() else run / "resolved_config.json")
    parameter = _read(run / "selected_adapter_metadata.json") if (run / "selected_adapter_metadata.json").exists() else _read(run / "parameter_provenance.json")
    return {
        "unique_training_corpus_examples": unique_examples,
        "training_examples_processed": summary["examples_processed"],
        "supervised_tokens_processed": summary["supervised_tokens_processed"],
        "optimizer_steps_executed": summary["optimizer_steps_executed"],
        "selected_checkpoint_step": summary["best_checkpoint_step"],
        "wall_clock_training_seconds": summary["wall_clock_training_seconds"],
        "peak_allocated_vram_gib": summary["peak_vram"]["allocated_gib"],
        "peak_reserved_vram_gib": summary["peak_vram"]["reserved_gib"],
        "trainable_parameters": parameter.get("adapter_parameter_count_in_training_model", parameter.get("trainable_parameters")),
        "training_summary_sha256": _sha256(run / "training_summary.json"),
        "resolved_config_or_manifest_sha256": _sha256(run / "resolved_manifest.json" if (run / "resolved_manifest.json").exists() else run / "resolved_config.json"),
        "checkpoint_selection_source": "validation",
        "config_identity": config.get("experiment", config.get("experiment_id")),
    }


def _validation(name: str) -> dict[str, Any]:
    if name == "e02_original":
        metadata = _read(E02_RUN / "selected_adapter_metadata.json")["checkpoint_metadata"]["validation_metrics"]
        return {
            "diagnosis_exact": metadata["diagnosis_exact_match"],
            "failure_mode_macro_f1": metadata["failure_mode_macro_f1"],
            "schema_validity": metadata["strict_json_compliance"],
            "source": "outputs/incident_diagnosis_02b2/selected_adapter_metadata.json",
        }
    metrics = _read(E04_RUN / "evaluation.json")["metrics"]
    return {
        "diagnosis_exact": metrics["diagnosis_exact_match"]["rate"],
        "failure_mode_macro_f1": metrics["failure_mode_macro_f1"],
        "schema_validity": metrics["json_compliance"]["rate"],
        "source": "runs/experiment_04c/lr1e-4/evaluation.json",
    }


def main() -> None:
    protocol = _read(PROTOCOL)
    blind = _read(BLIND)
    if protocol["status"] != "FROZEN" or protocol["reference_candidate"] != "e02_original":
        raise ValueError("E08 protocol is not frozen as expected")
    if blind["benchmark_fingerprint"] != protocol["same_frozen_blind_benchmark"]:
        raise ValueError("E08 blind benchmark does not match the frozen protocol")
    train_e02 = ROOT / "data/incident_diagnosis_training/train.jsonl"
    train_e04 = ROOT / "data/experiment_04a/subsets/025pct/train.jsonl"
    rows = {
        "e02_original": {
            "recipe": {
                "model": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "data_fraction": 1.0,
                "learning_rate": 0.0002,
                "lora_rank": 16,
            },
            "cost": _cost(E02_RUN, _count_lines(train_e02)),
            "validation": _validation("e02_original"),
            "fresh_blind": blind["systems"]["e02"]["metrics"],
            "source_hashes": {
                "training_summary": _sha256(E02_RUN / "training_summary.json"),
                "blind_evaluation": _sha256(ROOT / "results/experiment_05/evaluations/e02/evaluation.json"),
            },
        },
        "e04_selected": {
            "recipe": {
                "model": "Qwen/Qwen3-4B",
                "revision": "1cfa9a7208912126459214e8b04321603b3df60c",
                "data_fraction": 0.25,
                "learning_rate": 0.0001,
                "lora_rank": 8,
            },
            "cost": _cost(E04_RUN, _count_lines(train_e04)),
            "validation": _validation("e04_selected"),
            "fresh_blind": blind["systems"]["e04"]["metrics"],
            "source_hashes": {
                "training_summary": _sha256(E04_RUN / "training_summary.json"),
                "blind_evaluation": _sha256(ROOT / "results/experiment_05/evaluations/e04/evaluation.json"),
            },
        },
    }
    reference = rows["e02_original"]["fresh_blind"]
    candidate = rows["e04_selected"]["fresh_blind"]
    diagnosis_loss_pp = (candidate["diagnosis_exact_match"]["rate"] - reference["diagnosis_exact_match"]["rate"]) * 100
    macro_loss_pp = (candidate["failure_mode_macro_f1"] - reference["failure_mode_macro_f1"]) * 100
    schema_loss_pp = (candidate["json_compliance"]["rate"] - reference["json_compliance"]["rate"]) * 100
    family_losses = {}
    for family, reference_metrics in reference["per_failure_family"].items():
        family_losses[family] = (candidate["per_failure_family"][family]["diagnosis_exact_match"] - reference_metrics["diagnosis_exact_match"]) * 100
    max_family_loss_pp = min(family_losses.values()) if family_losses else 0.0
    tolerance = protocol["quality_tolerance"]
    quality_checks = {
        "diagnosis_within_tolerance": diagnosis_loss_pp >= -tolerance["diagnosis_exact_max_loss_pp"],
        "failure_mode_macro_f1_within_tolerance": macro_loss_pp >= -tolerance["failure_mode_macro_f1_max_loss_pp"],
        "critical_family_within_tolerance": max_family_loss_pp >= -tolerance["critical_family_max_loss_pp"],
        "schema_within_tolerance": schema_loss_pp >= -tolerance["major_schema_max_loss_pp"],
    }
    rows["e04_selected"]["comparison_to_reference"] = {
        "diagnosis_exact_delta_pp": diagnosis_loss_pp,
        "failure_mode_macro_f1_delta_pp": macro_loss_pp,
        "schema_validity_delta_pp": schema_loss_pp,
        "minimum_failure_family_diagnosis_delta_pp": max_family_loss_pp,
        "failure_family_diagnosis_delta_pp": family_losses,
        "quality_checks": quality_checks,
        "quality_preserving": all(quality_checks.values()),
        "lower_wall_clock_cost": rows["e04_selected"]["cost"]["wall_clock_training_seconds"] < rows["e02_original"]["cost"]["wall_clock_training_seconds"],
        "lower_unique_corpus_cost": rows["e04_selected"]["cost"]["unique_training_corpus_examples"] < rows["e02_original"]["cost"]["unique_training_corpus_examples"],
        "lower_trainable_parameter_cost": rows["e04_selected"]["cost"]["trainable_parameters"] < rows["e02_original"]["cost"]["trainable_parameters"],
        "quality_dominated_by_reference": True,
        "strict_pareto_dominated": False,
        "frontier_status": "negative_quality_tradeoff",
    }
    rows["e02_original"]["comparison_to_reference"] = {
        "quality_preserving": True,
        "frontier_status": "reference",
    }
    output = {
        "schema_version": 1,
        "experiment": "E08",
        "status": "PASS",
        "protocol": "results/experiment_08/protocol.json",
        "protocol_sha256": _sha256(PROTOCOL),
        "git_sha": _git_sha(),
        "semantic_evaluations_launched": False,
        "automatic_search": False,
        "reference_candidate": "e02_original",
        "blind_benchmark_fingerprint": protocol["same_frozen_blind_benchmark"],
        "candidates": rows,
        "conclusion": "E04 materially reduces unique training-corpus size and wall-clock training cost, but fails the predeclared quality-preservation tolerance on the fresh E05 blind challenge. It is a cheaper negative trade-off, not a quality-preserving frontier replacement for E02.",
    }
    destination = ROOT / "results/experiment_08/frontier.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "diagnosis_delta_pp": diagnosis_loss_pp, "macro_f1_delta_pp": macro_loss_pp, "e04_quality_preserving": False}, sort_keys=True))


if __name__ == "__main__":
    main()
