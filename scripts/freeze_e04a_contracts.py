#!/usr/bin/env python3
"""Write the four resolved Experiment 04A contracts after subset freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.evidence import sha256_path
from causetune.experiment_contract import resolve_experiment_config


MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
MODEL_ID = "Qwen/Qwen3-4B"
SEED = 20260941


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", default="data/experiment_04a/subsets")
    parser.add_argument("--output", default="configs/experiment_04a")
    args = parser.parse_args()
    subsets = Path(args.subsets)
    output = Path(args.output)
    validation = subsets / "validation"
    benchmark = Path("data/incident_diagnosis")
    validation_hash = sha256_path(validation)
    benchmark_hash = sha256_path(benchmark)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        label = f"{int(fraction * 100):03d}pct"
        train = subsets / label
        manifest = json.loads((train / "subset_manifest.json").read_text(encoding="utf-8"))
        examples = int(manifest["example_count"])
        max_steps = (examples // 8) * 2
        raw = {
            "schema_version": 1,
            "experiment_id": f"e04a-{label}",
            "model": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "revision_policy": "pinned",
            },
            "data": {
                "train": {"path": str(train), "fingerprint": sha256_path(train)},
                "validation": {"path": str(validation), "fingerprint": validation_hash},
                "benchmark": {"path": str(benchmark), "fingerprint": benchmark_hash},
            },
            "training": {
                "seed": SEED,
                "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True},
                "lora": {"rank": 16, "alpha": 32, "dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]},
                "optimizer": {"micro_batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 2e-4, "max_epochs": 2, "max_sequence_length": 768, "gradient_checkpointing": True, "gradient_checkpointing_use_reentrant": False},
                "checkpoint_policy": {"selection_split": "validation", "validation_only": True, "primary_metric": "diagnosis_exact_match", "interval_steps": 25},
                "stopping_policy": {"mode": "early_stopping", "max_steps": max_steps, "patience": 3, "min_delta": 0.005, "eval_interval_steps": 25},
                "preprocessing": {"deterministic": True, "shuffle": True, "seed": SEED, "version": "incident-preprocess-v1"},
            },
            "evaluation": {"contract_version": "incident-evaluation-v1", "scorer_version": "incident-scorer-v1"},
            "output": {"output_dir": f"runs/experiment_04a/{label}"},
            "metadata": {"tags": "e04a-data-efficiency", "notes": f"fraction={fraction}; benchmark=frozen-reporting-only"},
        }
        contract = resolve_experiment_config(raw)
        contract.write_resolved(output / f"{label}.json")
        print(f"{label} {contract.sha256()}")


if __name__ == "__main__":
    main()
