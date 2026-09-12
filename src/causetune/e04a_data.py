"""Deterministic data-efficiency subset construction for Experiment 04A."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .incident_training import training_fingerprint, validate_training_split


E04A_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _stable_rank(seed: int, family: str, incident_id: str) -> str:
    return hashlib.sha256(f"e04a-subset-v1|{seed}|{family}|{incident_id}".encode("utf-8")).hexdigest()


def _distribution(inputs: list[Mapping[str, Any]], truths: list[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    truth_by_id = {str(row["incident_id"]): row for row in truths}
    return {
        "failure_family": dict(sorted(Counter(str(row["failure_mode"]) for row in truths).items())),
        "difficulty": dict(sorted(Counter(str(row["metadata"]["difficulty"]) for row in truths).items())),
        "topology_family": dict(sorted(Counter(str(row["metadata"]["topology_family"]) for row in truths).items())),
        "red_herring": dict(sorted(Counter(str(row["metadata"]["red_herring"]).lower() for row in truths).items())),
    }


def load_training_pairs(source_dir: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = Path(source_dir)
    inputs = _read_jsonl(source / "train.jsonl")
    truths = _read_jsonl(source / "ground_truth_train.jsonl")
    validate_training_split(inputs, truths, expected_count=len(inputs), expected_per_family=len(inputs) // 12)
    return inputs, truths


def materialize_data_efficiency_subsets(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    seed: int,
    fractions: tuple[float, ...] = E04A_FRACTIONS,
) -> dict[str, Any]:
    """Materialize nested family-stratified subsets with path-independent IDs."""

    inputs, truths = load_training_pairs(source_dir)
    truth_by_id = {str(row["incident_id"]): row for row in truths}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in inputs:
        incident_id = str(row["incident_id"])
        family = str(truth_by_id[incident_id]["failure_mode"])
        grouped.setdefault(family, []).append(row)
    if not fractions or tuple(sorted(fractions)) != fractions:
        raise ValueError("fractions must be non-empty and sorted")
    if any(fraction <= 0 or fraction > 1 for fraction in fractions):
        raise ValueError("fractions must be in (0, 1]")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    source_hash = training_fingerprint(inputs, truths)
    validation_dir = destination / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(source_dir) / "validation.jsonl", validation_dir / "validation.jsonl")
    shutil.copyfile(Path(source_dir) / "ground_truth_validation.jsonl", validation_dir / "ground_truth_validation.jsonl")
    records: list[dict[str, Any]] = []
    for fraction in fractions:
        selected: list[dict[str, Any]] = []
        for family in sorted(grouped):
            ranked = sorted(grouped[family], key=lambda row: _stable_rank(seed, family, str(row["incident_id"])))
            count = len(ranked) if fraction == 1.0 else max(1, round(len(ranked) * fraction))
            selected.extend(ranked[:count])
        selected.sort(key=lambda row: str(row["incident_id"]))
        selected_truths = [truth_by_id[str(row["incident_id"])] for row in selected]
        validate_training_split(selected, selected_truths, expected_count=len(selected), expected_per_family=len(selected) // 12)
        name = f"{int(fraction * 100):03d}pct"
        subset_dir = destination / name
        _write_jsonl(subset_dir / "train.jsonl", selected)
        _write_jsonl(subset_dir / "ground_truth_train.jsonl", selected_truths)
        subset_hash = training_fingerprint(selected, selected_truths)
        manifest = {
            "schema_version": 1,
            "study_id": "e04a-data-efficiency",
            "fraction": fraction,
            "example_count": len(selected),
            "seed": seed,
            "source_dataset_hash": source_hash,
            "subset_hash": subset_hash,
            "identity": hashlib.sha256(f"e04a-subset-v1|{seed}|{source_hash}|{fraction}".encode("utf-8")).hexdigest(),
            "family_distribution": _distribution(selected, selected_truths)["failure_family"],
            "difficulty_distribution": _distribution(selected, selected_truths)["difficulty"],
            "topology_distribution": _distribution(selected, selected_truths)["topology_family"],
            "red_herring_distribution": _distribution(selected, selected_truths)["red_herring"],
            "ordering": "incident_id ascending after per-family SHA-256 rank selection",
        }
        _write_json(subset_dir / "subset_manifest.json", manifest)
        records.append(manifest)
    result = {
        "schema_version": 1,
        "study_id": "e04a-data-efficiency",
        "seed": seed,
        "source_dataset_hash": source_hash,
        "subsets": records,
    }
    _write_json(destination / "study_manifest.json", result)
    return result
