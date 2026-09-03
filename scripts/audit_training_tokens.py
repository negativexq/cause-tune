#!/usr/bin/env python3
"""Audit and correct completed-run token telemetry without loading weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

from fineforge.config import load_config
from fineforge.data.preprocess import IGNORE_INDEX, preprocess_records
from fineforge.data.schema import read_jsonl
from fineforge.data.split import load_split_manifest
from fineforge.training import (
    count_preprocessed_input_tokens,
    count_preprocessed_supervised_tokens,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_smoke.json")
    parser.add_argument("--output-dir", help="override config.output_dir")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config.output_dir)
    metrics_path = output_dir / "training_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"completed training metrics not found: {metrics_path}")

    records = read_jsonl(config.dataset_path)
    splits = load_split_manifest(
        records,
        Path(config.dataset_path).with_suffix(".splits.json"),
    )
    tokenizer = AutoTokenizer.from_pretrained(output_dir)
    examples = preprocess_records(
        splits["train"],
        tokenizer,
        config.max_sequence_length,
    )
    per_example = [
        {
            "example_id": example.example_id,
            "supervised_assistant_tokens": sum(
                label != IGNORE_INDEX for label in example.labels
            ),
            "causal_shifted_supervised_assistant_tokens": sum(
                label != IGNORE_INDEX for label in example.labels[1:]
            ),
            "model_input_tokens": len(example.input_ids),
        }
        for example in examples
    ]
    supervised_counts = [
        item["supervised_assistant_tokens"] for item in per_example
    ]
    shifted_counts = [
        item["causal_shifted_supervised_assistant_tokens"]
        for item in per_example
    ]
    input_counts = [item["model_input_tokens"] for item in per_example]
    supervised_total = count_preprocessed_supervised_tokens(examples)
    shifted_total = sum(shifted_counts)
    input_total = count_preprocessed_input_tokens(examples)
    assert supervised_total == shifted_total

    audit = {
        "tokenizer_source": str(output_dir),
        "train_examples": len(examples),
        "supervised_assistant_tokens": {
            "min": min(supervised_counts),
            "max": max(supervised_counts),
            "mean": mean(supervised_counts),
            "median": median(supervised_counts),
            "total": sum(supervised_counts),
        },
        "causal_shifted_supervised_assistant_tokens_total": shifted_total,
        "model_input_tokens": {
            "min": min(input_counts),
            "max": max(input_counts),
            "mean": mean(input_counts),
            "total": input_total,
        },
        "per_example": per_example,
        "counter_conclusion": (
            "The original counter summed non-ignored assistant labels over all "
            "160 train examples. It did not count optimizer steps or mishandle "
            "gradient accumulation."
        ),
    }
    write_json(output_dir / "token_audit.json", audit)

    training_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    duration = float(training_metrics["wall_clock_training_seconds"])
    training_metrics.update(
        {
            "effective_training_tokens": supervised_total,
            "supervised_assistant_tokens_processed": supervised_total,
            "causal_shifted_supervised_assistant_tokens": shifted_total,
            "model_input_tokens_processed": input_total,
            "supervised_assistant_tokens_per_second": supervised_total / duration,
            "input_tokens_per_second": input_total / duration,
            "training_tokens_per_second": supervised_total / duration,
            "token_counter_definition": (
                "supervised assistant target tokens after causal next-token shift"
            ),
        }
    )
    write_json(metrics_path, training_metrics)

    overall_metrics_path = output_dir / "metrics.json"
    overall_metrics = json.loads(overall_metrics_path.read_text(encoding="utf-8"))
    overall_metrics["training"] = training_metrics
    write_json(overall_metrics_path, overall_metrics)

    print("Updated:", metrics_path)
    print("Updated:", overall_metrics_path)
    print("Wrote:", output_dir / "token_audit.json")
    print(json.dumps(audit["supervised_assistant_tokens"], sort_keys=True))
    print("causal_shifted_total:", shifted_total)
    print(json.dumps(audit["model_input_tokens"], sort_keys=True))


if __name__ == "__main__":
    main()
