"""Strict validation and manifest construction for the M4 realistic dataset."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from .leakage import duplicate_report
from .perturbations import PHENOMENA
from .realistic_generate import DATASET_VERSION, GENERATOR_VERSION, SPLITS
from .schema import INTENT_SET, SchemaValidationError, validate_records
from .taxonomy import AMBIGUITY_RULES

DIFFICULTIES = frozenset({"easy", "medium", "hard"})
REQUIRED_METADATA = {
    "split", "difficulty", "phenomena", "confusable_with", "label_rule",
    "scenario_family", "template_family", "semantic_scenario", "generator_version",
}


def validate_realistic_record(record: Any, split: str | None = None) -> dict[str, Any]:
    """Validate the base chat schema plus all M4 metadata and policy constraints."""

    try:
        result = validate_records([record])[0]
    except SchemaValidationError:
        raise
    if set(result["expected_response"]) != {"intent"}:
        raise SchemaValidationError(f"{result['example_id']}: expected_response has extra keys")
    try:
        assistant = json.loads(result["messages"][1]["content"])
    except json.JSONDecodeError as exc:  # base schema normally catches this
        raise SchemaValidationError(f"{result['example_id']}: malformed assistant JSON: {exc.msg}") from exc
    if set(assistant) != {"intent"}:
        raise SchemaValidationError(f"{result['example_id']}: assistant output must contain only intent")
    metadata = {key: result.get(key) for key in REQUIRED_METADATA}
    missing = [key for key, value in metadata.items() if key not in result]
    if missing:
        raise SchemaValidationError(f"{result['example_id']}: missing metadata: {missing}")
    actual_split = result["split"]
    if actual_split not in SPLITS or (split is not None and actual_split != split):
        raise SchemaValidationError(f"{result['example_id']}: invalid split {actual_split!r}")
    if result["difficulty"] not in DIFFICULTIES:
        raise SchemaValidationError(f"{result['example_id']}: invalid difficulty")
    phenomena = result["phenomena"]
    if not isinstance(phenomena, list) or not all(isinstance(item, str) and item in PHENOMENA for item in phenomena):
        raise SchemaValidationError(f"{result['example_id']}: invalid phenomena metadata")
    confusable = result["confusable_with"]
    if not isinstance(confusable, list) or not all(item in INTENT_SET for item in confusable):
        raise SchemaValidationError(f"{result['example_id']}: invalid confusable_with metadata")
    if result["expected_response"]["intent"] in confusable:
        raise SchemaValidationError(f"{result['example_id']}: intent cannot be confusable with itself")
    if result["label_rule"] is not None and result["label_rule"] not in AMBIGUITY_RULES:
        raise SchemaValidationError(f"{result['example_id']}: unsupported ambiguity rule {result['label_rule']!r}")
    for field in ("scenario_family", "template_family", "semantic_scenario", "generator_version"):
        if not isinstance(result[field], str) or not result[field].strip():
            raise SchemaValidationError(f"{result['example_id']}: {field} must be non-empty")
    if result["label_rule"] is not None and "confusable_intent" not in phenomena and "multi_issue" not in phenomena:
        raise SchemaValidationError(f"{result['example_id']}: rule-tagged example lacks confusable or multi_issue phenomenon")
    return result


def validate_realistic_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_dataset_version: str = DATASET_VERSION,
) -> dict[str, Any]:
    """Validate all split invariants, including cross-split duplicate/family leakage."""

    if set(splits) != set(SPLITS):
        raise SchemaValidationError(f"expected exactly splits {list(SPLITS)}")
    validated: dict[str, list[dict[str, Any]]] = {}
    all_ids: list[str] = []
    for split in SPLITS:
        validated[split] = [validate_realistic_record(record, split) for record in splits[split]]
        all_ids.extend(record["example_id"] for record in validated[split])
    if len(all_ids) != len(set(all_ids)):
        raise SchemaValidationError("duplicate example IDs across realistic splits")
    if any(record["generator_version"] != GENERATOR_VERSION for record in (r for rows in validated.values() for r in rows)):
        raise SchemaValidationError("generator version mismatch")
    report = duplicate_report(validated)
    if report["exact_duplicate_count"] or report["normalized_duplicate_count"]:
        raise SchemaValidationError("duplicate normalized content detected")
    if report["cross_split_normalized_duplicates"]:
        raise SchemaValidationError("cross-split normalized duplicate leakage detected")
    if report["ood_template_family_leakage"]:
        raise SchemaValidationError("OOD template-family leakage detected")
    if report["cross_split_template_family_overlap"]:
        raise SchemaValidationError("template families overlap across splits")
    for split, rows in validated.items():
        for intent in INTENT_SET:
            if not any(row["expected_response"]["intent"] == intent for row in rows):
                raise SchemaValidationError(f"{split}: missing class {intent}")
    return {
        "status": "pass",
        "dataset_version": expected_dataset_version,
        "split_sizes": {split: len(rows) for split, rows in validated.items()},
        "duplicate_check": report,
        "class_distribution": {
            split: dict(Counter(row["expected_response"]["intent"] for row in rows))
            for split, rows in validated.items()
        },
    }


def read_realistic_splits(dataset_dir: str | Path) -> dict[str, list[dict[str, Any]]]:
    directory = Path(dataset_dir)
    splits: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        path = directory / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        rows: list[Any] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SchemaValidationError(f"{path}:{line_number}: malformed JSON: {exc.msg}") from exc
        splits[split] = rows
    validate_realistic_splits(splits)
    return splits


def build_manifest(splits: Mapping[str, Sequence[Mapping[str, Any]]], seed: int) -> dict[str, Any]:
    """Build a reproducibility and coverage manifest after strict validation."""

    validation = validate_realistic_splits(splits)
    all_rows = [row for rows in splits.values() for row in rows]

    def distribution(field: str) -> dict[str, dict[str, int]]:
        return {
            split: dict(Counter(
                value for row in rows for value in (
                    row[field] if isinstance(row[field], list) else [row[field]]
                )
            ))
            for split, rows in splits.items()
        }

    return {
        "dataset_version": DATASET_VERSION,
        "generation_seed": seed,
        "generator_version": GENERATOR_VERSION,
        "split_sizes": validation["split_sizes"],
        "class_distribution": validation["class_distribution"],
        "difficulty_distribution": distribution("difficulty"),
        "phenomenon_distribution": distribution("phenomena"),
        "scenario_family_counts": distribution("scenario_family"),
        "template_family_counts": distribution("template_family"),
        "confusable_pair_counts": {
            split: dict(Counter(
                f"{row['expected_response']['intent']} vs {other}"
                for row in rows for other in row["confusable_with"]
            ))
            for split, rows in splits.items()
        },
        "ambiguity_rule_counts": dict(Counter(row["label_rule"] for row in all_rows if row["label_rule"])),
        "example_ids": {split: [row["example_id"] for row in rows] for split, rows in splits.items()},
        "duplicate_check": validation["duplicate_check"],
    }


def write_validation_report(path: str | Path, report: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")

