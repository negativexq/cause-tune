#!/usr/bin/env python3
"""Finalize already-written M6 artifacts without loading a model or rerunning eval."""

from __future__ import annotations

import json
from pathlib import Path

from fineforge.m6 import m6_artifact_metadata


def main() -> None:
    output_dir = Path("outputs/realistic_qlora_m6")
    post = json.loads((output_dir / "post_training_metrics.json").read_text())
    training = json.loads((output_dir / "training_metrics.json").read_text())
    training["trainable_percentage"] = (
        training["trainable_parameter_count"]
        / training["logical_parameter_count"]
        * 100
    )
    (output_dir / "training_metrics.json").write_text(
        json.dumps(training, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    comparison = json.loads((output_dir / "base_vs_m5_vs_m6.json").read_text())
    progression = json.loads((output_dir / "validation_progression.json").read_text())
    summary = {
        "status": "success",
        "order_audit": json.loads((output_dir / "order_audit.json").read_text()),
        "training": training,
        "validation_progression": progression,
        "post_training_metrics": {
            split: {
                key: post["metrics"][split][key]
                for key in (
                    "intent_accuracy",
                    "macro_f1",
                    "valid_json_rate",
                    "exact_schema_compliance_rate",
                    "teacher_forced_loss",
                )
            }
            for split in post["metrics"]
        },
        "base_vs_m5_vs_m6": comparison,
        "reload_verification": json.loads((output_dir / "reload_verification.json").read_text()),
    }
    (output_dir / "artifact_metadata.json").write_text(
        json.dumps({
            **m6_artifact_metadata(output_dir),
            "m6_specific_files": [
                "experiment_diff.json",
                "shuffled_order.json",
                "order_audit.json",
                "optimizer_steps.jsonl",
                "validation_progression.json",
                "base_vs_m5_vs_m6.json",
            ],
        }, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "m6_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "finalized", "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
