#!/usr/bin/env python3
"""Freeze E04-C contracts from the selected E04-A and E04-B choices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.experiment_contract import resolve_experiment_config


LEARNING_RATES = (("lr1e-4", 1e-4), ("lr2e-4", 2e-4), ("lr4e-4", 4e-4))
SELECTED_VARIANT = "025pct"
SELECTED_SUBSET_HASH = "eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56"
SELECTED_RANK = 8
FIXED_ALPHA = 32


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/experiment_04b/r8.json")
    parser.add_argument("--data-selection", default="results/experiment_04a/selection.json")
    parser.add_argument("--capacity-selection", default="results/experiment_04b/selection.json")
    parser.add_argument("--subset-manifest", default="data/experiment_04a/subsets/025pct/subset_manifest.json")
    parser.add_argument("--output", default="configs/experiment_04c")
    args = parser.parse_args()

    data_selection = json.loads(Path(args.data_selection).read_text(encoding="utf-8"))
    capacity_selection = json.loads(Path(args.capacity_selection).read_text(encoding="utf-8"))
    subset = json.loads(Path(args.subset_manifest).read_text(encoding="utf-8"))
    if data_selection["selected_variant"] != SELECTED_VARIANT or data_selection["selected_fraction"] != 0.25:
        raise ValueError("E04-A selected fraction is not the frozen 25% result")
    if data_selection["selected_subset_hash"] != SELECTED_SUBSET_HASH or subset["subset_hash"] != SELECTED_SUBSET_HASH:
        raise ValueError("E04-A selected subset hash does not match the frozen E04-C input")
    if capacity_selection["selected_rank"] != SELECTED_RANK or capacity_selection["selected_variant"] != "r8":
        raise ValueError("E04-B selected rank is not the frozen rank-8 result")

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    for variant, learning_rate in LEARNING_RATES:
        raw = json.loads(json.dumps(base))
        raw["experiment_id"] = f"e04c-{variant}"
        raw["training"]["lora"]["rank"] = SELECTED_RANK
        raw["training"]["lora"]["alpha"] = FIXED_ALPHA
        raw["training"]["optimizer"]["learning_rate"] = learning_rate
        raw["output"]["output_dir"] = f"runs/experiment_04c/{variant}"
        raw["metadata"]["tags"] = "e04c-learning-rate"
        raw["metadata"]["notes"] = f"learning_rate={learning_rate}; rank={SELECTED_RANK}; alpha={FIXED_ALPHA}; data_fraction=0.25; selected_subset_hash={SELECTED_SUBSET_HASH}"
        write_json(output / f"{variant}.json", resolve_experiment_config(raw).to_dict())
    print(json.dumps({"status": "PASS", "learning_rates": LEARNING_RATES, "rank": SELECTED_RANK, "alpha": FIXED_ALPHA, "selected_subset_hash": SELECTED_SUBSET_HASH}, sort_keys=True))


if __name__ == "__main__":
    main()
