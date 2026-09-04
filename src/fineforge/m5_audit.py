"""CPU-only audit helpers for the completed M5 run and future sampler tests."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Mapping, Sequence


def deterministic_shuffled_indices(size: int, seed: int) -> list[int]:
    """Return a reproducible shuffled epoch order without touching model state."""

    if size < 0:
        raise ValueError("size must be non-negative")
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    return indices


def _counts(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _composition(values: Sequence[str]) -> dict[str, Any]:
    counts = Counter(values)
    dominant_count = max(counts.values()) if counts else 0
    return {
        "label_counts": dict(sorted(counts.items())),
        "distinct_labels": len(counts),
        "dominant_labels": sorted(
            label for label, count in counts.items() if count == dominant_count
        ),
        "dominant_count": dominant_count,
        "composition": "single_class" if len(counts) == 1 else "mixed_class",
    }


def audit_training_order(
    records: Sequence[Mapping[str, Any]],
    *,
    gradient_accumulation_steps: int = 8,
    block_size: int = 100,
) -> dict[str, Any]:
    """Summarize the exact list order consumed by the explicit M5 loader."""

    if not records:
        raise ValueError("records must not be empty")
    if gradient_accumulation_steps <= 0 or block_size <= 0:
        raise ValueError("window and block sizes must be positive")
    labels = [str(record["expected_response"]["intent"]) for record in records]
    example_ids = [str(record["example_id"]) for record in records]

    blocks = []
    for start in range(0, len(records), block_size):
        end = min(start + block_size, len(records))
        blocks.append({
            "start_index_zero_based": start,
            "end_index_exclusive": end,
            "label_counts": _counts(labels[start:end]),
            "composition": _composition(labels[start:end]),
        })

    windows = []
    for window_number, start in enumerate(
        range(0, len(records), gradient_accumulation_steps),
        start=1,
    ):
        end = min(start + gradient_accumulation_steps, len(records))
        composition = _composition(labels[start:end])
        windows.append({
            "optimizer_step": window_number,
            "start_index_zero_based": start,
            "end_index_exclusive": end,
            "example_ids": example_ids[start:end],
            **composition,
        })

    wrong_item_positions = [
        index for index, label in enumerate(labels) if label == "wrong_item"
    ]
    wrong_item_summary = {
        "first_index_zero_based": min(wrong_item_positions) if wrong_item_positions else None,
        "last_index_zero_based": max(wrong_item_positions) if wrong_item_positions else None,
        "first_position_one_based": min(wrong_item_positions) + 1 if wrong_item_positions else None,
        "last_position_one_based": max(wrong_item_positions) + 1 if wrong_item_positions else None,
        "final_50_count": sum(label == "wrong_item" for label in labels[-50:]),
        "final_100_count": sum(label == "wrong_item" for label in labels[-100:]),
        "final_200_count": sum(label == "wrong_item" for label in labels[-200:]),
        "final_400_count": sum(label == "wrong_item" for label in labels[-400:]),
    }
    contiguous_runs = []
    run_start = 0
    for index in range(1, len(labels) + 1):
        if index == len(labels) or labels[index] != labels[run_start]:
            contiguous_runs.append({
                "label": labels[run_start],
                "start_index_zero_based": run_start,
                "end_index_exclusive": index,
                "count": index - run_start,
            })
            run_start = index

    return {
        "example_count": len(records),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "block_size": block_size,
        "first_50_labels": labels[:50],
        "last_50_labels": labels[-50:],
        "overall_label_counts": _counts(labels),
        "example_ids_sorted_lexicographically": example_ids == sorted(example_ids),
        "labels_sorted_lexicographically": labels == sorted(labels),
        "contiguous_label_runs": contiguous_runs,
        "blocks_of_100": blocks,
        "accumulation_windows": windows,
        "single_class_window_count": sum(
            window["composition"] == "single_class" for window in windows
        ),
        "mixed_class_window_count": sum(
            window["composition"] == "mixed_class" for window in windows
        ),
        "wrong_item": wrong_item_summary,
    }
