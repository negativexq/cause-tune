"""Versioned evaluation-contract fingerprinting for Experiment 02A."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def contract_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "prompt_version": config["prompt_version"],
        "tokenizer": config["tokenizer"],
        "evaluation_contract": config["evaluation_contract"],
    }


def contract_fingerprint(config: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        contract_payload(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
