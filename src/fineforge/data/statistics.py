"""Dataset inspection statistics with optional tokenizer-only token counts."""

from __future__ import annotations

from collections import Counter
from statistics import mean
from typing import Any, Mapping, Sequence

from .leakage import duplicate_report
from .schema import INTENTS


def _token_count(tokenizer: Any, record: Mapping[str, Any]) -> tuple[int, int]:
    messages = record["messages"]
    try:
        rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
    except TypeError:
        rendered = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    assistant = tokenizer.encode(messages[1]["content"], add_special_tokens=False)
    return len(rendered), len(assistant)


def _split_statistics(records: Sequence[Mapping[str, Any]], tokenizer: Any | None) -> dict[str, Any]:
    chars = [len(record["messages"][0]["content"]) for record in records]
    token_lengths: list[int] = []
    assistant_tokens: list[int] = []
    if tokenizer is not None:
        for record in records:
            inputs, assistant = _token_count(tokenizer, record)
            token_lengths.append(inputs)
            assistant_tokens.append(assistant)
    return {
        "examples": len(records),
        "class_distribution": dict(Counter(record["expected_response"]["intent"] for record in records)),
        "difficulty_distribution": dict(Counter(record["difficulty"] for record in records)),
        "average_user_characters": round(mean(chars), 2) if chars else 0,
        "min_user_characters": min(chars) if chars else 0,
        "max_user_characters": max(chars) if chars else 0,
        "average_tokenizer_input_tokens": round(mean(token_lengths), 2) if token_lengths else None,
        "min_tokenizer_input_tokens": min(token_lengths) if token_lengths else None,
        "max_tokenizer_input_tokens": max(token_lengths) if token_lengths else None,
        "average_assistant_tokens": round(mean(assistant_tokens), 2) if assistant_tokens else None,
        "min_assistant_tokens": min(assistant_tokens) if assistant_tokens else None,
        "max_assistant_tokens": max(assistant_tokens) if assistant_tokens else None,
        "phenomenon_counts": dict(Counter(value for record in records for value in record["phenomena"])),
        "confusable_pair_counts": dict(Counter(
            f"{record['expected_response']['intent']} vs {other}"
            for record in records for other in record["confusable_with"]
        )),
        "scenario_family_counts": dict(Counter(record["scenario_family"] for record in records)),
    }


def dataset_statistics(splits: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer: Any | None = None) -> dict[str, Any]:
    """Return split-level and global duplicate/leakage statistics."""

    result = {
        "splits": {split: _split_statistics(records, tokenizer) for split, records in splits.items()},
        "global": {
            "supported_intents": list(INTENTS),
            "total_examples": sum(len(records) for records in splits.values()),
            **duplicate_report(splits),
        },
    }
    return result
