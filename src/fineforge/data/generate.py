"""Deterministic local smoke-dataset generation."""

from __future__ import annotations

import json
import random
from itertools import product
from typing import Any

from .schema import INTENTS


_INTENT_CONTENTS: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "refund": (
        (
            "I need a refund for {item}.",
            "Please help me get my money back for {item}.",
            "Can you process a refund for {item}?",
            "I want to return {item} and request a refund.",
        ),
        (
            "a damaged order",
            "an order that did not meet expectations",
            "my recent purchase",
            "an item I no longer need",
            "the order I received yesterday",
        ),
    ),
    "duplicate_charge": (
        (
            "I was charged twice for {item}.",
            "There are two charges for {item} on my account.",
            "Why did I pay twice for {item}?",
            "I see a duplicate payment related to {item}.",
        ),
        (
            "order 83120",
            "my latest order",
            "a card purchase",
            "the same transaction",
            "yesterday's checkout",
        ),
    ),
    "order_missing": (
        (
            "My order {item} has not arrived.",
            "I am still waiting for {item}.",
            "Can you locate {item} for me?",
            "The delivery for {item} is missing.",
        ),
        (
            "order 44018",
            "my package",
            "the shipment placed last week",
            "my latest delivery",
            "the order marked as shipped",
        ),
    ),
    "wrong_item": (
        (
            "I received the wrong item: {item}.",
            "The package contains {item}, not what I ordered.",
            "My delivery is incorrect because of {item}.",
            "Please help with an incorrect item, {item}.",
        ),
        (
            "a blue shirt instead of a black one",
            "a different size",
            "the wrong model",
            "an unrelated product",
            "a substitute I did not request",
        ),
    ),
    "cancel_order": (
        (
            "Please cancel {item}.",
            "I want to stop {item} before it ships.",
            "Can you cancel {item} for me?",
            "I changed my mind about {item} and need it canceled.",
        ),
        (
            "order 91204",
            "my recent purchase",
            "the order I placed this morning",
            "my pending order",
            "the shipment that has not left yet",
        ),
    ),
    "payment_failed": (
        (
            "My payment failed for {item}.",
            "I cannot complete checkout for {item}.",
            "The card payment was declined while buying {item}.",
            "Checkout keeps failing for {item}.",
        ),
        (
            "my cart",
            "order 77301",
            "a subscription renewal",
            "the purchase I am making",
            "my latest checkout attempt",
        ),
    ),
    "account_locked": (
        (
            "My account is locked.",
            "I cannot sign in because my account is locked.",
            "Please help unlock my account.",
            "Too many attempts locked me out of my account.",
        ),
        (
            "after several login attempts",
            "on my phone",
            "since this morning",
            "after changing my password",
            "when I tried to access my profile",
        ),
    ),
    "subscription_cancel": (
        (
            "I want to cancel {item}.",
            "Please stop my recurring {item}.",
            "How do I end {item}?",
            "Cancel my {item} subscription, please.",
        ),
        (
            "monthly plan",
            "premium membership",
            "recurring service",
            "annual subscription",
            "current membership",
        ),
    ),
    "fraud_suspected": (
        (
            "I suspect fraud involving {item}.",
            "I do not recognize {item}.",
            "There may be unauthorized activity related to {item}.",
            "Please investigate a suspicious transaction: {item}.",
        ),
        (
            "a card charge",
            "a login notification",
            "a purchase on my account",
            "an unfamiliar order",
            "activity from an unknown device",
        ),
    ),
    "human_escalation": (
        (
            "I need to speak with a human about {item}.",
            "Please connect me with a support representative for {item}.",
            "Can an agent review {item}?",
            "I would like this issue escalated: {item}.",
        ),
        (
            "a complicated account issue",
            "a problem not solved by the help center",
            "my unresolved order case",
            "a billing dispute",
            "this support request",
        ),
    ),
}


def generate_smoke_records(seed: int, examples_per_intent: int = 20) -> list[dict[str, Any]]:
    """Generate deterministic two-turn records for all configured intents."""

    if examples_per_intent != 20:
        raise ValueError("the smoke dataset requires exactly 20 examples per intent")

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    for intent in INTENTS:
        templates, details = _INTENT_CONTENTS[intent]
        combinations = list(product(templates, details))
        rng.shuffle(combinations)
        for index, (template, detail) in enumerate(combinations, start=1):
            user_content = template.format(item=detail)
            assistant_content = json.dumps(
                {"intent": intent},
                separators=(",", ":"),
                sort_keys=True,
            )
            records.append(
                {
                    "example_id": f"smoke-{intent}-{index:02d}",
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "expected_response": {"intent": intent},
                }
            )
    return records

