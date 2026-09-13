#!/usr/bin/env python3
"""Build the v1.0 readiness audit from persisted CauseTune evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load(relative: str) -> dict[str, Any]:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def rate(metrics: dict[str, Any], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, dict):
        return value.get("rate")
    return value


def metric_summary(metrics: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: metrics.get(key) for key in keys if key in metrics}


def evidence_row(
    *,
    experiment: str,
    question: str,
    contract_version: str,
    git_sha: str,
    model: str | None,
    revision: str | None,
    training_hash: str | None,
    validation_hash: str | None,
    benchmark_hash: str | None,
    checkpoint: int | None,
    headline_metrics: dict[str, Any],
    artifact_location: list[str],
    offline_verification: str,
    limitations: list[str],
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "scientific_question": question,
        "contract_version": contract_version,
        "git_sha": git_sha,
        "model": model,
        "model_revision": revision,
        "training_dataset_hash": training_hash,
        "validation_hash": validation_hash,
        "benchmark_hash": benchmark_hash,
        "selected_checkpoint": checkpoint,
        "headline_metrics": headline_metrics,
        "artifact_location": artifact_location,
        "offline_verification": offline_verification,
        "limitations": limitations,
    }


def main() -> None:
    e04_recipe = load("results/experiment_04/selected_recipe.json")
    model = e04_recipe["model"]["model_id"]
    revision = e04_recipe["model"]["revision"]
    train_hash = e04_recipe["data"]["train_fingerprint"]
    validation_hash = e04_recipe["data"]["validation_fingerprint"]
    subset_hash = e04_recipe["data"]["subset_hash"]

    e04a_selection = load("results/experiment_04a/selection.json")
    e04a_comparison = load("results/experiment_04a/validation_comparison.json")
    e04b_selection = load("results/experiment_04b/selection.json")
    e04b_comparison = load("results/experiment_04b/validation_comparison.json")
    e04c_selection = load("results/experiment_04c/selection.json")
    e04c_comparison = load("results/experiment_04c/validation_comparison.json")

    for selection, name in (
        (e04a_selection, "E04A"),
        (e04b_selection, "E04B"),
        (e04c_selection, "E04C"),
    ):
        require(selection["status"] == "PASS", f"{name} selection is not PASS")
        require(
            selection["benchmark_used_for_selection"] is False
            and selection["e03_used_for_selection"] is False,
            f"{name} selection boundary is not sealed",
        )

    for comparison, name in (
        (e04a_comparison, "E04A"),
        (e04b_comparison, "E04B"),
        (e04c_comparison, "E04C"),
    ):
        require(comparison["status"] == "PASS", f"{name} comparison is not PASS")
        for variant in comparison["variants"]:
            require(
                variant["evidence"]["verification"] == "PASS",
                f"{name}/{variant['variant']} evidence is not verified",
            )
            require(
                variant["evidence"]["raw_predictions_preserved"] is True,
                f"{name}/{variant['variant']} lacks raw predictions",
            )

    e04a_variant = next(
        row for row in e04a_comparison["variants"] if row["variant"] == e04a_selection["selected_variant"]
    )
    e04b_variant = next(
        row for row in e04b_comparison["variants"] if row["variant"] == e04b_selection["selected_variant"]
    )
    e04c_variant = next(
        row for row in e04c_comparison["variants"] if row["variant"] == e04c_selection["selected_variant"]
    )

    rows = [
        evidence_row(
            experiment="E04-A",
            question="How much training data is required to preserve specialization capability?",
            contract_version="e04a-selection-v1",
            git_sha="4ab4d3da466acb89ab6e6fb5b8eaf6b960ed3be2",
            model=model,
            revision=revision,
            training_hash=e04a_variant["dataset_fingerprint"],
            validation_hash=validation_hash,
            benchmark_hash=None,
            checkpoint=e04a_variant["selected_checkpoint"],
            headline_metrics=metric_summary(
                e04a_variant["metrics"],
                ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1", "strict_json"),
            )
            | {"selected_fraction": e04a_selection["selected_fraction"], "examples": e04a_variant["example_count"]},
            artifact_location=[
                "results/experiment_04a/selection.json",
                "results/experiment_04a/validation_comparison.json",
                "runs/experiment_04a/",
            ],
            offline_verification="PASS for all four valid semantic runs; 75% technical failure retained separately",
            limitations=["Validation task saturation does not establish fresh generalization sufficiency."],
        ),
        evidence_row(
            experiment="E04-B",
            question="What LoRA capacity is required after the selected data fraction?",
            contract_version="e04b-selection-v1",
            git_sha="4ab4d3da466acb89ab6e6fb5b8eaf6b960ed3be2",
            model=model,
            revision=revision,
            training_hash=e04b_variant["dataset_fingerprint"],
            validation_hash=e04b_variant["validation_fingerprint"],
            benchmark_hash=None,
            checkpoint=e04b_variant["selected_checkpoint"],
            headline_metrics=metric_summary(
                e04b_variant["metrics"],
                ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1", "strict_json"),
            )
            | {"selected_rank": e04b_selection["selected_rank"], "selected_alpha": 32},
            artifact_location=[
                "results/experiment_04b/selection.json",
                "results/experiment_04b/validation_comparison.json",
                "runs/experiment_04b/",
            ],
            offline_verification="PASS for r8, r16, and r32",
            limitations=["Capacity conclusion is bounded by the frozen Qwen3-4B validation task."],
        ),
        evidence_row(
            experiment="E04-C",
            question="What learning rate preserves specialization quality under the selected recipe?",
            contract_version="e04c-selection-v1",
            git_sha="4ab4d3da466acb89ab6e6fb5b8eaf6b960ed3be2",
            model=model,
            revision=revision,
            training_hash=e04c_variant["dataset_fingerprint"],
            validation_hash=e04c_variant["validation_fingerprint"],
            benchmark_hash=None,
            checkpoint=e04c_variant["selected_checkpoint"],
            headline_metrics=metric_summary(
                e04c_variant["metrics"],
                ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1", "strict_json"),
            )
            | {"selected_learning_rate": e04c_selection["selected_learning_rate"]},
            artifact_location=[
                "results/experiment_04c/selection.json",
                "results/experiment_04c/validation_comparison.json",
                "runs/experiment_04c/",
            ],
            offline_verification="PASS for 1e-4, 2e-4, and 4e-4",
            limitations=["Learning-rate sensitivity is measured only within the frozen E04 protocol."],
        ),
    ]

    e05_summary = load("results/experiment_05/g05b_summary.json")
    e05_comparison = load("results/experiment_05/evaluation_comparison.json")
    require(e05_summary["status"] == "PASS", "E05 G05B is not PASS")
    require(e05_summary["offline_reproduction"] is True, "E05 offline reproduction failed")
    require(e05_summary["one_shot_per_system"] is True, "E05 was not one-shot")
    e05_systems = e05_comparison["systems"]
    rows.append(
        evidence_row(
            experiment="E05",
            question="Does the E04-selected recipe generalize to a fresh blind challenge?",
            contract_version="e05-one-shot-blind-v1",
            git_sha="6705442d73a8d3ae6019f16b1501fb67ed7c83e3",
            model=model,
            revision=revision,
            training_hash=None,
            validation_hash=None,
            benchmark_hash=e05_summary["benchmark_fingerprint"],
            checkpoint=None,
            headline_metrics={
                system: {
                    "diagnosis_exact": rate(data["metrics"], "diagnosis_exact_match"),
                    "resolution_exact": rate(data["metrics"], "resolution_exact_match"),
                    "failure_mode_macro_f1": data["metrics"]["failure_mode_macro_f1"],
                    "strict_json": rate(data["metrics"], "json_compliance"),
                }
                for system, data in e05_systems.items()
            }
            | {"e04_vs_e02_diagnosis_delta_pp": -5.833333333333323},
            artifact_location=[
                "results/experiment_05/g05b_summary.json",
                "results/experiment_05/evaluation_comparison.json",
                "results/experiment_05/artifact_hashes.json",
            ],
            offline_verification="PASS for base, E02, and E04; raw predictions and transitions preserved",
            limitations=[
                "Synthetic, taxonomy-aligned challenge; not production or real-world OOD accuracy.",
                "The E04 generalization regression remains historical evidence and was not used retroactively.",
            ],
        )
    )

    e06_summary = load("results/experiment_06/final_evaluation-retry-01/g06_summary.json")
    e06_protocol = load("results/experiment_06/final_protocol.json")
    e06_preflight = load("results/experiment_06/training_preflight.json")
    require(e06_summary["status"] == "PASS", "E06 G06 is not PASS")
    require(e06_summary["offline_reproduction"] is True, "E06 offline reproduction failed")
    require(e06_summary["capability_screen_used_as_final_evidence"] is False, "E06 reused capability screen")
    e06_metrics = {}
    for system in ("base", "tuned"):
        evaluation = load(f"results/experiment_06/final_evaluation-retry-01/{system}/evaluation.json")
        metrics = evaluation["metrics"]
        e06_metrics[system] = metric_summary(
            metrics,
            ("diagnosis_exact_match", "resolution_exact_match", "failure_mode_macro_f1", "json_compliance", "json_valid_rate"),
        )
    rows.append(
        evidence_row(
            experiment="E06",
            question="Is the measured specialization behavior reproducible on a non-Qwen model family?",
            contract_version="e06-one-shot-final-blind-v1",
            git_sha="37bc3ffcec1201095734d9a48e99303a53a8a0d8",
            model=e06_protocol["systems"][0]["model_id"],
            revision=e06_protocol["systems"][0]["revision"],
            training_hash=train_hash,
            validation_hash=validation_hash,
            benchmark_hash=e06_protocol["benchmark_fingerprint"],
            checkpoint=125,
            headline_metrics=e06_metrics
            | {
                "capability_screen": "CAPABILITY_GAP_PRESENT",
                "architecture_specific_targets": e06_preflight["lora_attachment"]["target_module_counts"],
            },
            artifact_location=[
                "results/experiment_06/capability_gap-retry-01/",
                "results/experiment_06/final_evaluation-retry-01/",
                "results/experiment_06/compatibility_recovery.json",
                "results/experiment_06/capability_gap/technical_failure.json",
            ],
            offline_verification="PASS for capability screen, training evidence, and separate final base/tuned benchmark",
            limitations=[
                "Synthetic, taxonomy-aligned held-out challenge with one additional model family.",
                "Phi attempt 0 and a post-evaluation aggregation failure remain preserved as technical provenance.",
            ],
        )
    )

    e07_summary = load("results/experiment_07/evaluations/g07_summary.json")
    e07_protocol = load("results/experiment_07/protocol.json")
    require(e07_summary["status"] == "PASS", "E07 G07 is not PASS")
    require(e07_summary["offline_reproduction"] is True, "E07 offline reproduction failed")
    e07_metrics = {}
    for system in ("base", "e02", "e04"):
        metrics = load(f"results/experiment_07/evaluations/{system}/evaluation.json")["metrics"]
        e07_metrics[system] = {
            "abstention_precision": metrics["abstention_precision"],
            "abstention_recall": metrics["abstention_recall"],
            "false_confident_diagnosis_rate": metrics["false_confident_diagnosis_rate"],
            "false_abstention_rate": metrics["false_abstention_rate"],
            "sufficient_case_accuracy": metrics["normal_sufficient_case_accuracy"],
            "schema_valid_rate": metrics["schema_valid_rate"],
        }
    rows.append(
        evidence_row(
            experiment="E07",
            question="Does specialization increase confident wrong diagnoses at the failure boundary?",
            contract_version="e07-boundary-v1",
            git_sha="a380c1fe0c5f8a9ca29d904faf7645c094ae0de5",
            model=model,
            revision=revision,
            training_hash=None,
            validation_hash=None,
            benchmark_hash=e07_protocol["benchmark_fingerprint"],
            checkpoint=None,
            headline_metrics=e07_metrics,
            artifact_location=[
                "results/experiment_07/protocol.json",
                "results/experiment_07/evaluations/g07_summary.json",
                "results/experiment_07/evaluations/",
            ],
            offline_verification="PASS for all three systems; scorer correction recorded without semantic rerun",
            limitations=[
                "Synthetic boundary benchmark; taxonomy-aligned and not a production safety guarantee.",
                "E04 still produced 9/48 false confident diagnoses at the boundary.",
            ],
        )
    )

    e08 = load("results/experiment_08/frontier.json")
    require(e08["status"] == "PASS", "E08 G08 is not PASS")
    require(e08["automatic_search"] is False and e08["semantic_evaluations_launched"] is False, "E08 launched new search/evals")
    rows.append(
        evidence_row(
            experiment="E08",
            question="What quality/cost frontier is revealed by the already measured recipes?",
            contract_version="e08-efficiency-frontier-v1",
            git_sha="86512b817d5ab0d889b24779a2ae66a6b999fb20",
            model=model,
            revision=revision,
            training_hash=None,
            validation_hash=None,
            benchmark_hash=e08["blind_benchmark_fingerprint"],
            checkpoint=None,
            headline_metrics={
                "frontier_status": "negative_quality_tradeoff",
                "e04_quality_preserving": False,
                "e04_diagnosis_delta_pp": e08["candidates"]["e04_selected"]["comparison_to_reference"]["diagnosis_exact_delta_pp"],
                "e04_macro_f1_delta_pp": e08["candidates"]["e04_selected"]["comparison_to_reference"]["failure_mode_macro_f1_delta_pp"],
                "e04_unique_corpus_reduction": 0.75,
                "e04_wall_clock_reduction": 0.6647,
            },
            artifact_location=[
                "results/experiment_08/protocol.json",
                "results/experiment_08/frontier.json",
                "docs/experiment-08/e08-results.md",
            ],
            offline_verification="PASS: deterministic analysis of persisted E02/E04 cost and E05 blind evidence",
            limitations=["E08 reuses historical measurements and does not provide a new semantic sample."],
        )
    )

    audit = {
        "schema_version": 1,
        "status": "PASS",
        "release": "v1.0.0-readiness",
        "scope": "mandatory scientific methodology and evidence audit; EX-LV is optional and excluded",
        "generated_from": "persisted evidence only; no semantic experiment rerun",
        "mandatory_gates": {
            "E04": "PASS",
            "E05": "PASS",
            "E06": "PASS",
            "E07": "PASS",
            "E08": "PASS",
        },
        "rows": rows,
        "limitations": [
            "Synthetic benchmark construction and taxonomy alignment.",
            "Limited model-family coverage despite the controlled Phi replication.",
            "One-shot blind challenges and static/manual fixture limitations where disclosed.",
            "Hardware-specific cost measurements.",
            "Adapter weights are local-only; lightweight hashes, manifests, metrics, and locations are persisted.",
        ],
    }

    output = ROOT / "results/release/v1.0-readiness-audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = [
        "# v1.0 Readiness Audit",
        "",
        "Status: `PASS`.",
        "",
        "This audit is generated from persisted CauseTune evidence. It does not",
        "rerun semantic experiments. EX-LV is optional and is not required for v1.0.",
        "",
        "| Experiment | Contract | Model | Benchmark hash | Checkpoint | Verification |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['experiment']} | `{row['contract_version']}` | "
            f"{row['model'] or 'evaluation-only'} | "
            f"`{row['benchmark_hash'] or 'not applicable'}` | "
            f"{row['selected_checkpoint'] if row['selected_checkpoint'] is not None else '—'} | "
            f"{row['offline_verification']} |"
        )
    lines.extend(
        [
            "",
            "## Mandatory gate closure",
            "",
            "- E04 controlled optimization studies: `PASS`.",
            "- E05 fresh blind generalization: `PASS`; E04 diagnosis regression versus E02 remains `-5.83 pp`.",
            "- E06 cross-model replication: `PASS`; Phi capability gap and final held-out evaluation are separate.",
            "- E07 failure boundary: `PASS`; E04 false confident diagnosis remains non-zero.",
            "- E08 efficiency frontier: `PASS`; E04 is a cheaper negative trade-off, not quality-preserving.",
            "",
            "## Limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in audit["limitations"])
    lines.append("")
    (ROOT / "docs/release-v1.0-readiness.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"wrote {output.relative_to(ROOT)}")
    print("wrote docs/release-v1.0-readiness.md")


if __name__ == "__main__":
    main()
