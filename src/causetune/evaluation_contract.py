"""The single frozen M4 evaluation contract used by base and tuned models."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

PROMPT_VERSION = "m4-frozen-contract-v1"
SYSTEM_INSTRUCTION = (
    "You are an intent classification system. Classify the customer request into "
    "exactly one of these labels: refund, duplicate_charge, order_missing, "
    "wrong_item, cancel_order, payment_failed, account_locked, "
    "subscription_cancel, fraud_suspected, human_escalation. Return exactly one "
    "JSON object in this form: {\"intent\":\"<label>\"}. Return JSON only, with "
    "no explanation and no additional keys."
)


def evaluation_messages(
    record: Mapping[str, Any],
    system_instruction: str = SYSTEM_INSTRUCTION,
) -> list[dict[str, str]]:
    """Build the exact system+user prompt; metadata never enters the prompt."""

    return [
        {"role": "system", "content": system_instruction},
        dict(record["messages"][0]),
    ]


def contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only contract-defining fields for stable fingerprinting."""

    evaluation = config["evaluation_contract"]
    return {
        "prompt_version": config["prompt_version"],
        "system_instruction": evaluation["system_instruction"],
        "tokenizer": config["tokenizer"],
        "generation": evaluation["generation"],
        "parsing": evaluation["parsing"],
        "schema": evaluation["schema"],
    }


def contract_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

