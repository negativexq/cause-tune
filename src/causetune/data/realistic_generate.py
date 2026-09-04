"""Reusable deterministic local generation pipeline for the M4 dataset.

The generator separates semantic scenarios, surface realization, perturbation,
metadata, and split writing. It uses no model or external service.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from .leakage import normalize_text
from .perturbations import PHENOMENA, apply_perturbations
from .scenarios import SCENARIOS_BY_INTENT, Scenario
from .schema import INTENTS, write_jsonl
from .taxonomy import validate_taxonomy


GENERATOR_VERSION = "m4-realistic-local-v1"
# Immutable historical identifier: changing it would change the verified M4
# manifest and dataset fingerprint.
DATASET_VERSION = "fineforge-m4-realistic-v1"
SPLITS = ("train", "validation", "id_test", "hard_test", "ood_test")
COUNTS_PER_INTENT = {"train": 200, "validation": 25, "id_test": 25, "hard_test": 25, "ood_test": 25}
DIFFICULTY_COUNTS = {
    "train": {"easy": 100, "medium": 70, "hard": 30},
    "validation": {"easy": 9, "medium": 9, "hard": 7},
    "id_test": {"easy": 8, "medium": 8, "hard": 9},
    "hard_test": {"easy": 2, "medium": 5, "hard": 18},
    "ood_test": {"easy": 4, "medium": 8, "hard": 13},
}

_CORE_TEMPLATES: dict[str, tuple[str, ...]] = {
    "refund": (
        "Please return the money for {subject}.", "I need a refund for {subject}.",
        "Can you reverse the charge for {subject}?", "I am asking for reimbursement on {subject}.",
        "What is the process to get my money back for {subject}?", "The remedy I want is a refund for {subject}.",
        "Could support refund {subject}, please?", "I paid for {subject} and want that payment returned.",
        "Help me undo the purchase of {subject} and send the funds back.", "I would like the charge for {subject} refunded.",
    ),
    "duplicate_charge": (
        "I was charged twice for {subject}.", "There are two completed charges for {subject}.",
        "Why does my statement show {subject} two times?", "The same checkout appears as duplicate debits: {subject}.",
        "I see a repeated payment against {subject}; please remove one.", "My balance dropped twice for {subject}.",
        "The ledger has two entries tied to {subject}.", "One purchase, two charges: {subject}.",
        "Can you investigate the second posting for {subject}?", "I only made one payment, but {subject} is billed twice.",
    ),
    "order_missing": (
        "My {subject} still has not arrived.", "Please locate {subject} for me.",
        "The delivery for {subject} is missing.", "I am waiting on {subject}; tracking has not helped.",
        "Where is {subject}? It was supposed to be here by now.",
        "Can you trace {subject} through the carrier?", "The expected package, {subject}, never reached me.",
        "I cannot find {subject} even though the order was dispatched.",
        "Please check the non-delivery of {subject}.", "My delivery record says something happened, but I have not received {subject}.",
    ),
    "wrong_item": (
        "The package arrived, but it contains {subject}.", "I received {subject}, not what I ordered.",
        "The delivered product is wrong: {subject}.", "Can you fix an incorrect item? The box has {subject}.",
        "My order was fulfilled with {subject} instead of the requested version.", "The courier delivered a box, but the contents are {subject}.",
        "This is a fulfillment mistake; I got {subject}.", "The item in my parcel is {subject}, which is not my order.",
        "I need help with the wrong product, namely {subject}.", "The shipment is here but the SKU is wrong: {subject}.",
    ),
    "cancel_order": (
        "Please cancel {subject} before it ships.", "I want to stop {subject}.",
        "Can you cancel {subject} while it is still pending?", "I changed my mind about {subject}; cancel it please.",
        "Do not dispatch {subject}; I need that order canceled.", "The one-time order {subject} should not go out.",
        "Help me halt fulfillment for {subject}.", "It has not left the warehouse, so cancel {subject}.",
        "I no longer want {subject}. Please cancel the order.", "Can the pending shipment for {subject} be called off?",
    ),
    "payment_failed": (
        "My payment failed for {subject}.", "Checkout will not complete for {subject}.",
        "The card was declined while I tried to pay for {subject}.", "I cannot get the payment for {subject} through.",
        "The purchase of {subject} keeps throwing a payment error.", "The authorization for {subject} was rejected.",
        "I tried to pay for {subject}, but no payment went through.", "Why is checkout refusing the payment for {subject}?",
        "The renewal or purchase {subject} is stuck at payment.", "The payment screen failed again for {subject}.",
    ),
    "account_locked": (
        "My account is locked after trying to sign in.", "Please unlock my account; I cannot log in.",
        "I am locked out of my profile.", "Too many attempts blocked my account.",
        "Sign-in says my account is locked and I need access.", "Can someone restore access to my account?",
        "The login lock is preventing me from opening my account.", "I cannot get past the account lock.",
        "My credentials may be fine, but the account is blocked.", "Help me regain access to the locked profile.",
    ),
    "subscription_cancel": (
        "Cancel my {subject}.", "Please stop the recurring billing for {subject}.",
        "I want to end {subject} before another renewal.", "How do I turn off the {subject} subscription?",
        "Do not renew {subject} again.", "I no longer want this recurring plan: {subject}.",
        "Please terminate future charges for {subject}.", "Make {subject} stop renewing.",
        "I need to cancel the membership called {subject}.", "The next renewal of {subject} should not happen.",
    ),
    "fraud_suspected": (
        "I do not recognize {subject}.", "There may be unauthorized activity involving {subject}.",
        "Please investigate {subject}; I did not make it.", "I suspect fraud around {subject}.",
        "This activity is unfamiliar to me: {subject}.", "Someone may have used my account for {subject}.",
        "Can you flag {subject} as suspicious?", "I did not authorize {subject}.",
        "The account history contains an event I cannot identify, {subject}.", "I think {subject} could be fraudulent.",
    ),
    "human_escalation": (
        "Please connect me with a human about {subject}.", "I need an agent to review {subject}.",
        "Can a support representative take over this {subject} case?", "Please escalate {subject} to a person.",
        "I want to speak with someone about {subject}.", "The automated help was not enough; get me a human for {subject}.",
        "Could an actual support specialist handle {subject}?", "I am requesting human review of {subject}.",
        "Route this {subject} issue to an agent, please.", "I need a person, not another article, for {subject}.",
    ),
}

_OOD_PREFIXES = (
    "Here is the short version from my account history: ",
    "If you read the last line of the statement first, you will see this: ",
    "Back-and-forth compressed into one message: support asked me to write here, and ",
    "I am filing this from the order timeline—",
    "The relevant event happened before the context below: ",
)
_SPLIT_PREFIXES = {
    "train": ("", "For my account, ", "Quick question: ", "I am contacting you because "),
    "validation": ("From the order history: ", "Support, ", "A question about my case: ", "Please note: "),
    "id_test": ("I need help with this record: ", "Looking at today's activity, ", "Customer message: ", "Could you check this: "),
    "hard_test": ("This has become a complicated case: ", "After checking several places, ", "The important part is that ", "I am not sure how to classify this, but "),
    "ood_test": _OOD_PREFIXES,
}
_COLLISION_CONTEXTS = (
    "email receipt", "mobile app", "laptop browser", "bank statement",
    "order-history page", "checkout screen", "delivery dashboard",
    "saved receipt", "notification panel", "support ticket", "account timeline",
    "payment alert", "carrier page", "phone notification", "billing history",
    "purchase confirmation", "case summary", "help-center transcript",
    "shipment view", "profile screen", "card statement", "renewal notice",
    "security page", "warehouse update", "merchant receipt", "status email",
    "customer portal", "checkout history", "delivery email", "account notice",
    "app inbox", "transaction list", "order confirmation", "support transcript",
    "device screen", "membership page", "payment history",
)


def _stable_int(seed: int, *parts: str) -> int:
    digest = hashlib.sha256((str(seed) + "|" + "|".join(parts)).encode()).hexdigest()
    return int(digest[:16], 16)


def _facts(seed: int, split: str, intent: str, index: int) -> dict[str, str]:
    value = _stable_int(seed, split, intent, str(index))
    return {
        "order_id": str(100000 + value % 899999),
        "transaction_id": f"TX-{value % 90000000:08d}",
        "amount": f"{(value % 24000 + 700) / 100:.2f}",
        "date": f"{(value % 25) + 1} {('Jan', 'Mar', 'May', 'Jul', 'Sep', 'Nov')[(value // 25) % 6]} 2026",
        "merchant": ("Northstar Market", "Willow Books", "Harbor Electronics", "Cedar Pharmacy", "Lumen Outfitters")[value % 5],
    }


def _subject(scenario: Scenario, facts: dict[str, str]) -> str:
    return scenario.subject.format(**facts)


def _difficulty_list(split: str) -> list[str]:
    result: list[str] = []
    for difficulty, count in DIFFICULTY_COUNTS[split].items():
        result.extend([difficulty] * count)
    return result


def _phenomena(split: str, difficulty: str, scenario: Scenario, index: int) -> set[str]:
    result: set[str] = set()
    if difficulty == "easy":
        result.add(("polite", "short_request", "order_id", "date")[index % 4])
    elif difficulty == "medium":
        result.add(("informal", "irrelevant_context", "amount", "long_form", "uncertain", "multiple_sentences")[index % 6])
        if index % 4 == 0:
            result.add("prior_failed_attempt")
    else:
        result.add(("typo", "punctuation_error", "incomplete_sentence", "frustrated", "incorrect_terminology")[index % 5])
        result.add("confusable_intent")
        if scenario.multi_issue and difficulty == "hard":
            result.update({"multi_issue", "multiple_sentences"})
        if index % 3 == 0:
            result.add("prior_failed_attempt")
    if scenario.label_rule:
        result.add("confusable_intent")
    if scenario.multi_issue and difficulty == "hard":
        result.update({"multi_issue", "multiple_sentences"})
    if split == "ood_test":
        result.add(("colloquial", "narrative", "chat_compressed", "unusual_ordering", "technical_distraction")[index % 5])
    if "order" in scenario.subject and index % 7 == 0:
        result.add("order_id")
    if "transaction" in scenario.subject:
        result.add("transaction_id")
    return result & PHENOMENA


def _make_text(split: str, intent: str, scenario: Scenario, index: int, facts: dict[str, str], phenomena: set[str]) -> str:
    templates = _CORE_TEMPLATES[intent]
    template = templates[(index + _stable_int(index, split, intent)) % len(templates)]
    if "short_request" in phenomena:
        template = templates[index % 4]
    text = template.format(subject=_subject(scenario, facts))
    prefix = _SPLIT_PREFIXES[split][index % len(_SPLIT_PREFIXES[split])]
    text = prefix + text
    if "amount" in phenomena:
        text += " The amount shown is $" + facts["amount"] + "."
    if "date" in phenomena:
        text += " It appeared on " + facts["date"] + "."
    if "multi_issue" in phenomena:
        additions = {
            "refund": " I also want the original order closed out.",
            "duplicate_charge": " The first attempt had looked unsuccessful.",
            "order_missing": " I would prefer my money back if it cannot be found.",
            "wrong_item": " I need the correct product sent instead.",
            "cancel_order": " I do not need a refund if it can be stopped now.",
            "payment_failed": " I am worried the retry might have posted twice.",
            "account_locked": " The lock appeared after an unfamiliar login alert.",
            "subscription_cancel": " A renewal already appeared on the statement.",
            "fraud_suspected": " The account also briefly stopped accepting my login.",
            "human_escalation": " The underlying issue is a missing order.",
        }
        text += additions[intent]
    if scenario.confusable_with and (split == "hard_test" or "confusable_intent" in phenomena):
        cues = {
            ("duplicate_charge", "fraud_suspected"): " Although the merchant name is unfamiliar, there are two settled postings for this one purchase.",
            ("fraud_suspected", "duplicate_charge"): " There is one unfamiliar posting, not two charges for the same purchase.",
            ("duplicate_charge", "payment_failed"): " The first attempt looked declined, but two completed debits are now present.",
            ("payment_failed", "duplicate_charge"): " No completed debit appears twice; the attempts simply failed.",
            ("refund", "cancel_order"): " This is a request to reverse a completed payment, not merely stop an unshipped order.",
            ("cancel_order", "refund"): " The order has not shipped, and I want it stopped rather than refunded after completion.",
            ("refund", "subscription_cancel"): " The renewal already posted and I want that charge reversed, not just future renewals stopped.",
            ("subscription_cancel", "refund"): " My primary request is to stop future renewals, not only reverse the charge already posted.",
            ("order_missing", "wrong_item"): " Nothing arrived in the box; this is not a case of receiving an incorrect product.",
            ("wrong_item", "order_missing"): " The parcel did arrive, so it is not a missing shipment; its contents are wrong.",
            ("account_locked", "fraud_suspected"): " The sign-in lock is the problem to solve, even though the alert prompted it.",
            ("fraud_suspected", "account_locked"): " I can still sign in; the suspicious activity is the issue, not an access lock.",
            ("human_escalation", "refund"): " The underlying billing matter can be explained, but the requested next step is a human review.",
            ("human_escalation", "order_missing"): " The package issue is background; the requested next step is an agent.",
            ("human_escalation", "wrong_item"): " The fulfillment mistake is background; please have a person take the case.",
            ("human_escalation", "fraud_suspected"): " The security concern is background; I specifically need a human to review it.",
            ("human_escalation", "duplicate_charge"): " The repeated billing is background; I specifically need an agent.",
        }
        for other in scenario.confusable_with:
            cue = cues.get((intent, other))
            if cue:
                text += cue
    if split == "ood_test":
        if "narrative" in phenomena:
            text = "Yesterday, while reconciling a few unrelated account details, I noticed this: " + text
        elif "chat_compressed" in phenomena:
            text = "Customer: I checked the app. Support: please put the issue in one message. Customer: " + text
        elif "unusual_ordering" in phenomena:
            text = text + " That is the outcome I need; the reference was " + facts["transaction_id"] + "."
        elif "technical_distraction" in phenomena:
            text = "The app log showed a timeout and my phone was on a guest network. Separately, " + text
    return apply_perturbations(text, phenomena, index)


def _record(split: str, intent: str, scenario: Scenario, difficulty: str, index: int, seed: int) -> dict[str, Any]:
    facts = _facts(seed, split, intent, index)
    phenomena = _phenomena(split, difficulty, scenario, index)
    text = _make_text(split, intent, scenario, index, facts, phenomena)
    family = f"m4/{split}/{intent}/{scenario.key}"
    template_family = f"m4/{split}/surface/{intent}/{index % 10:02d}"
    return {
        "example_id": f"m4-{split}-{intent}-{index + 1:03d}",
        "messages": [
            {"role": "user", "content": text},
            {"role": "assistant", "content": json.dumps({"intent": intent}, separators=(",", ":"))},
        ],
        "expected_response": {"intent": intent},
        "split": split,
        "difficulty": difficulty,
        "phenomena": sorted(phenomena),
        "confusable_with": list(scenario.confusable_with),
        "label_rule": scenario.label_rule,
        "scenario_family": family,
        "template_family": template_family,
        "semantic_scenario": scenario.key,
        "generator_version": GENERATOR_VERSION,
    }


def generate_realistic_splits(seed: int) -> dict[str, list[dict[str, Any]]]:
    """Generate physically separate, class-balanced M4 split records."""

    validate_taxonomy()
    splits: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    used_normalized: set[str] = set()
    for split in splits:
        difficulties = _difficulty_list(split)
        for intent in INTENTS:
            scenarios = list(SCENARIOS_BY_INTENT[intent])
            rng = random.Random(_stable_int(seed, split, intent))
            rng.shuffle(scenarios)
            for index, difficulty in enumerate(difficulties):
                candidates = [s for s in scenarios if s.hard_negative or s.multi_issue] if split == "hard_test" else scenarios
                candidates = candidates or scenarios
                scenario = candidates[index % len(candidates)]
                record = _record(split, intent, scenario, difficulty, index, seed)
                normalized = normalize_text(record["messages"][0]["content"])
                if normalized in used_normalized:
                    context = _COLLISION_CONTEXTS[index % len(_COLLISION_CONTEXTS)]
                    record["messages"][0]["content"] += " For reference, this came from my " + context + "."
                    normalized = normalize_text(record["messages"][0]["content"])
                if normalized in used_normalized:
                    raise ValueError(f"unable to resolve normalized duplicate for {record['example_id']}")
                used_normalized.add(normalized)
                splits[split].append(record)
    for split in splits:
        splits[split].sort(key=lambda item: item["example_id"])
    return splits


def write_realistic_dataset(output_dir: str | Path, seed: int) -> dict[str, list[dict[str, Any]]]:
    output = Path(output_dir)
    splits = generate_realistic_splits(seed)
    for split, records in splits.items():
        write_jsonl(output / f"{split}.jsonl", records)
    return splits
