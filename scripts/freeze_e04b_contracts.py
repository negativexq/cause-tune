#!/usr/bin/env python3
"""Freeze E04-B contracts from the E04-A selected data fraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causetune.experiment_contract import resolve_experiment_config


RANKS = (8, 16, 32)
SELECTED_VARIANT = "025pct"
SELECTED_SUBSET_HASH = "eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56"
FIXED_ALPHA = 32


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", default="configs/experiment_04a/025pct.json")
    parser.add_argument("--selection", default="results/experiment_04a/selection.json")
    parser.add_argument("--subset-manifest", default="data/experiment_04a/subsets/025pct/subset_manifest.json")
    parser.add_argument("--output", default="configs/experiment_04b")
    args = parser.parse_args()

    selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
    subset = json.loads(Path(args.subset_manifest).read_text(encoding="utf-8"))
    if selection["selected_variant"] != SELECTED_VARIANT or selection["selected_fraction"] != 0.25:
        raise ValueError("E04-A selected fraction is not the frozen 25% result")
    if selection["selected_subset_hash"] != SELECTED_SUBSET_HASH or subset["subset_hash"] != SELECTED_SUBSET_HASH:
        raise ValueError("E04-A selected subset hash does not match the frozen E04-B input")

    base = json.loads(Path(args.base_config).read_text(encoding="utf-8"))
    output = Path(args.output)
    for rank in RANKS:
        raw = json.loads(json.dumps(base))
        raw["experiment_id"] = f"e04b-r{rank}"
        raw["training"]["lora"]["rank"] = rank
        raw["training"]["lora"]["alpha"] = FIXED_ALPHA
        raw["output"]["output_dir"] = f"runs/experiment_04b/r{rank}"
        raw["metadata"]["tags"] = "e04b-lora-capacity"
        raw["metadata"]["notes"] = f"rank={rank}; alpha={FIXED_ALPHA}; data_fraction=0.25; selected_subset_hash={SELECTED_SUBSET_HASH}"
        write_json(output / f"r{rank}.json", resolve_experiment_config(raw).to_dict())
    print(json.dumps({"status": "PASS", "ranks": RANKS, "alpha": FIXED_ALPHA, "selected_subset_hash": SELECTED_SUBSET_HASH}, sort_keys=True))


if __name__ == "__main__":
    main()
