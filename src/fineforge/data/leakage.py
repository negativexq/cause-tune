"""Duplicate and scenario-family leakage checks for realistic M4 splits."""

from __future__ import annotations

import re
import string
import unicodedata
from collections import defaultdict
from typing import Any, Mapping, Sequence


_ID_RE = re.compile(r"\b(?:order|transaction|txn|case|ticket|reference|ref)[\s:#-]*[a-z]?\d{3,}\b", re.I)
_NUMBER_RE = re.compile(r"(?<![a-z])\d+(?:[.,]\d+)?(?![a-z])", re.I)
_DATE_RE = re.compile(r"\b(?:\d{1,2}[/-]){2}\d{2,4}\b")
_SPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize case, whitespace, punctuation, dates, IDs, and variable numbers."""

    value = unicodedata.normalize("NFKC", text).casefold()
    value = _DATE_RE.sub("<date>", value)
    value = _ID_RE.sub(lambda match: re.sub(r"\d+", "<id>", match.group(0)), value)
    value = _NUMBER_RE.sub("<num>", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    return _SPACE_RE.sub(" ", value).strip()


def record_text(record: Mapping[str, Any]) -> str:
    messages = record.get("messages", [])
    return " ".join(str(message.get("content", "")) for message in messages if isinstance(message, Mapping))


def duplicate_groups(records: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[normalize_text(record_text(record))].append(str(record["example_id"]))
    return {key: ids for key, ids in groups.items() if len(ids) > 1}


def cross_split_leakage(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    by_key: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, records in splits.items():
        for record in records:
            by_key[normalize_text(record_text(record))].append((split, str(record["example_id"])))
    normalized = {
        key: entries for key, entries in by_key.items()
        if len({split for split, _ in entries}) > 1
    }
    families: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, records in splits.items():
        for record in records:
            families[str(record.get("template_family", ""))].append((split, str(record["example_id"])))
    family_overlap = {
        family: entries for family, entries in families.items()
        if family and len({split for split, _ in entries}) > 1
    }
    ood_family_leakage = {
        family: entries for family, entries in family_overlap.items()
        if any(split == "ood_test" for split, _ in entries)
    }
    return {
        "cross_split_normalized_duplicates": normalized,
        "cross_split_template_family_overlap": family_overlap,
        "ood_template_family_leakage": ood_family_leakage,
    }


def duplicate_report(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    all_records = [record for records in splits.values() for record in records]
    exact_texts: dict[str, list[str]] = defaultdict(list)
    for record in all_records:
        exact_texts[record_text(record)].append(str(record["example_id"]))
    exact_groups = {key: ids for key, ids in exact_texts.items() if len(ids) > 1}
    normalized_groups = duplicate_groups(all_records)
    cross = cross_split_leakage(splits)
    return {
        "exact_duplicate_count": sum(len(ids) - 1 for ids in exact_groups.values()),
        "normalized_duplicate_count": sum(len(ids) - 1 for ids in normalized_groups.values()),
        "exact_duplicate_groups": exact_groups,
        "normalized_duplicate_groups": normalized_groups,
        **{key: value for key, value in cross.items()},
    }

