#!/usr/bin/env python3
"""Persist the verified Experiment 04C comparison and frozen selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.e04c_selection import VARIANT_ORDER, comparison_row, select_variants


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/experiment_04c")
    parser.add_argument("--output", default="results/experiment_04c")
    args = parser.parse_args()
    root = Path(args.runs)
    rows = [comparison_row(root / variant, variant) for variant in VARIANT_ORDER]
    selection = select_variants(rows)
    comparison = {
        "schema_version": 1,
        "study_id": "e04c-learning-rate",
        "status": "PASS",
        "primary_intervention": "learning_rate",
        "selection_boundary": "validation-only",
        "benchmark_used_for_selection": False,
        "e03_used_for_selection": False,
        "variants": rows,
        "technical_failures": [],
    }
    write_json(Path(args.output) / "validation_comparison.json", comparison)
    write_json(Path(args.output) / "selection.json", {"schema_version": 1, "study_id": "e04c-learning-rate", "status": "PASS", **selection})
    print(json.dumps({"status": "PASS", "reference_variant": selection["reference_variant"], "selected_variant": selection["selected_variant"], "selected_learning_rate": selection["selected_learning_rate"], "eligible_variants": selection["eligible_variants"]}, sort_keys=True))


if __name__ == "__main__":
    main()
