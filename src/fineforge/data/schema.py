"""Schema and content validation for FineForge chat records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


INTENTS: tuple[str, ...] = (
    "refund",
    "duplicate_charge",
    "order_missing",
    "wrong_item",
    "cancel_order",
    "payment_failed",
    "account_locked",
    "subscription_cancel",
    "fraud_suspected",
    "human_escalation",
)
INTENT_SET = frozenset(INTENTS)


class SchemaValidationError(ValueError):
    """Raised when a dataset record violates the expected SFT schema."""


def _fail(example_id: str, message: str) -> None:
    raise SchemaValidationError(f"{example_id}: {message}")


def validate_record(record: Any) -> dict[str, Any]:
    """Validate one two-message user/assistant record.

    Records are not repaired. Every failure is reported to the caller.
    """

    if not isinstance(record, dict):
        raise SchemaValidationError("record must be a JSON object")

    example_id = record.get("example_id")
    if not isinstance(example_id, str) or not example_id.strip():
        raise SchemaValidationError("record: example_id must be a non-empty string")

    messages = record.get("messages")
    if not isinstance(messages, list):
        _fail(example_id, "messages must be a list")
    if len(messages) != 2:
        _fail(example_id, "messages must contain exactly one user and one assistant message")

    roles: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            _fail(example_id, f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            _fail(example_id, f"invalid role at message {index}: {role!r}")
        if not isinstance(content, str) or not content.strip():
            _fail(example_id, f"empty content at message {index}")
        roles.append(role)

    if roles.count("user") != 1:
        _fail(example_id, "missing user message")
    if roles.count("assistant") != 1:
        _fail(example_id, "missing assistant message")
    if roles != ["user", "assistant"]:
        _fail(example_id, "messages must be ordered user then assistant")

    expected = record.get("expected_response")
    if not isinstance(expected, dict):
        _fail(example_id, "expected_response must be an object")
    expected_intent = expected.get("intent")
    if expected_intent not in INTENT_SET:
        _fail(example_id, f"unknown expected intent: {expected_intent!r}")

    assistant_content = messages[1]["content"]
    try:
        parsed_assistant = json.loads(assistant_content)
    except json.JSONDecodeError as exc:
        _fail(example_id, f"malformed assistant JSON: {exc.msg}")
    if not isinstance(parsed_assistant, dict):
        _fail(example_id, "assistant JSON must be an object")
    assistant_intent = parsed_assistant.get("intent")
    if assistant_intent not in INTENT_SET:
        _fail(example_id, f"unknown assistant intent: {assistant_intent!r}")
    if assistant_intent != expected_intent:
        _fail(
            example_id,
            "assistant intent is inconsistent with expected_response intent",
        )

    return record


def validate_records(records: Iterable[Any]) -> list[dict[str, Any]]:
    """Validate all records and reject duplicate IDs."""

    validated: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        validated_record = validate_record(record)
        example_id = validated_record["example_id"]
        if example_id in seen_ids:
            raise SchemaValidationError(f"duplicate example_id: {example_id}")
        seen_ids.add(example_id)
        validated.append(validated_record)
    return validated


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read and validate a JSONL dataset."""

    dataset_path = Path(path)
    records: list[Any] = []
    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise SchemaValidationError(f"{dataset_path}:{line_number}: empty JSONL line")
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(
                f"{dataset_path}:{line_number}: malformed JSON: {exc.msg}"
            ) from exc
    return validate_records(records)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    """Validate and write records as stable UTF-8 JSONL."""

    validated = validate_records(records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in validated:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")

