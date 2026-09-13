#!/usr/bin/env python3
"""Persist the verified Experiment 04A comparison and frozen selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.e04a_selection import VARIANT_ORDER, comparison_row, select_variants


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", default="runs/experiment_04a")
    parser.add_argument("--output", default="results/experiment_04a")
    args = parser.parse_args()
    run_root = Path(args.runs)
    output = Path(args.output)
    rows = [comparison_row(run_root / variant, variant) for variant in VARIANT_ORDER]
    selection = select_variants(rows)
    technical_failure = json.loads((run_root / "075pct" / "technical_failure.json").read_text(encoding="utf-8"))
    if technical_failure.get("valid_semantic_run") is not False:
        raise ValueError("preserved 75% technical failure is not marked non-semantic")
    comparison = {
        "schema_version": 1,
        "study_id": "e04a-data-efficiency",
        "status": "PASS",
        "selection_boundary": "validation-only",
        "benchmark_used_for_selection": False,
        "e03_used_for_selection": False,
        "variants": rows,
        "excluded_technical_attempts": [{"variant": "075pct", "status": "TECHNICAL_FAILURE", "path": "runs/experiment_04a/075pct/technical_failure.json", "record": technical_failure}],
    }
    write_json(output / "validation_comparison.json", comparison)
    write_json(output / "selection.json", {"schema_version": 1, "study_id": "e04a-data-efficiency", "status": "PASS", **selection})
    print(json.dumps({"status": "PASS", "selected_fraction": selection["selected_fraction"], "reference_variant": selection["reference_variant"], "eligible_variants": selection["eligible_variants"]}, sort_keys=True))


if __name__ == "__main__":
    main()
