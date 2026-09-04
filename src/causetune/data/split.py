"""Deterministic, class-preserving dataset splitting."""

from __future__ import annotations

import random
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .schema import INTENT_SET, validate_records


def split_records(
    records: Sequence[Mapping[str, Any]],
    seed: int,
    train_ratio: float = 0.80,
    validation_ratio: float = 0.10,
    test_ratio: float = 0.10,
) -> dict[str, list[dict[str, Any]]]:
    """Split records deterministically, preserving each intent's coverage.

    The smoke dataset has 20 records per intent, so the configured ratios map
    exactly to 16/2/2 records per intent and 160/20/20 overall.
    """

    validated = validate_records(records)
    if not validated:
        raise ValueError("cannot split an empty dataset")
    ratios = (train_ratio, validation_ratio, test_ratio)
    if any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("split ratios must be non-negative and sum to 1.0")

    by_intent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in validated:
        intent = record["expected_response"]["intent"]
        if intent not in INTENT_SET:
            raise ValueError(f"unknown intent in split input: {intent}")
        by_intent[intent].append(record)

    split_names = ("train", "validation", "test")
    result = {name: [] for name in split_names}
    rng = random.Random(seed)
    for intent in sorted(by_intent):
        group = sorted(by_intent[intent], key=lambda item: item["example_id"])
        group_rng = random.Random(rng.randrange(2**63))
        group_rng.shuffle(group)
        raw_counts = [len(group) * ratio for ratio in ratios]
        counts = [round(raw_count) for raw_count in raw_counts]
        if any(abs(raw - count) > 1e-9 for raw, count in zip(raw_counts, counts)):
            raise ValueError(
                "configured ratios must produce integer per-intent counts for "
                f"deterministic class-preserving splitting; intent={intent!r}, "
                f"size={len(group)}"
            )
        if sum(counts) != len(group):
            raise ValueError(f"split counts do not cover intent {intent!r}")
        start = 0
        for name, count in zip(split_names, counts):
            result[name].extend(group[start : start + count])
            start += count

    for name in split_names:
        random.Random(rng.randrange(2**63)).shuffle(result[name])

    assert_valid_split(result, expected_total=len(validated))
    return result


def split_ids(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, list[str]]:
    """Return only stable example IDs for experiment manifests."""

    return {
        name: [record["example_id"] for record in records]
        for name, records in splits.items()
    }


def load_split_manifest(
    records: Sequence[Mapping[str, Any]],
    manifest_path: str | Path,
) -> dict[str, list[dict[str, Any]]]:
    """Load the persisted split artifact and resolve its IDs to records."""

    manifest_file = Path(manifest_path)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid split manifest: {manifest_file}: {exc}") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("example_ids"), dict):
        raise ValueError(f"split manifest must contain an example_ids object: {manifest_file}")

    validated = validate_records(records)
    by_id = {record["example_id"]: record for record in validated}
    splits: dict[str, list[dict[str, Any]]] = {}
    for name in ("train", "validation", "test"):
        ids = manifest["example_ids"].get(name)
        if not isinstance(ids, list):
            raise ValueError(f"split manifest is missing {name!r} IDs")
        if len(ids) != len(set(ids)):
            raise ValueError(f"split manifest contains duplicate IDs in {name!r}")
        try:
            splits[name] = [by_id[example_id] for example_id in ids]
        except KeyError as exc:
            raise ValueError(f"split manifest references unknown example ID: {exc.args[0]}") from exc

    expected_counts = manifest.get("counts")
    if isinstance(expected_counts, dict):
        actual_counts = {name: len(items) for name, items in splits.items()}
        if expected_counts != actual_counts:
            raise ValueError(
                f"split manifest counts {expected_counts} do not match IDs {actual_counts}"
            )
    assert_valid_split(splits, expected_total=len(validated))
    return splits


def assert_valid_split(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    expected_total: int,
) -> None:
    """Assert exact coverage and non-overlap for a split mapping."""

    required = {"train", "validation", "test"}
    if set(splits) != required:
        raise AssertionError(f"expected splits {sorted(required)}, got {sorted(splits)}")
    ids_by_split = {
        name: [record["example_id"] for record in records]
        for name, records in splits.items()
    }
    all_ids = [example_id for ids in ids_by_split.values() for example_id in ids]
    if len(all_ids) != expected_total:
        raise AssertionError(
            f"split union has {len(all_ids)} records; expected {expected_total}"
        )
    if len(set(all_ids)) != expected_total:
        raise AssertionError("split union contains duplicate or overlapping example IDs")
