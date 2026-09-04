"""Canonical CauseTune M4 intent taxonomy and deterministic ambiguity policy.

This module is the semantic source of truth for realistic dataset generation and
validation.  The metadata is deliberately plain Python data so it can be
inspected without loading a tokenizer or model.
"""

from __future__ import annotations

from typing import Any

from .schema import INTENTS, INTENT_SET


AMBIGUITY_RULES: dict[str, str] = {
    "R1_HUMAN_ESCALATION": (
        "An explicit request for a human, agent, representative, or escalation "
        "wins over the underlying business issue."
    ),
    "R2_DUPLICATE_OVER_FRAUD": (
        "Two or more settled postings for one purchase are duplicate_charge, "
        "even when the user also says the merchant is unfamiliar. A single "
        "unrecognized or unauthorized posting is fraud_suspected."
    ),
    "R3_DUPLICATE_OVER_PAYMENT_FAILED": (
        "If a failed or retried payment resulted in two completed debits, label "
        "duplicate_charge. If no duplicate completed debit is described, label "
        "payment_failed."
    ),
    "R4_SUBSCRIPTION_CANCELLATION_PRIMARY": (
        "A request whose requested remedy is stopping future recurring billing is "
        "subscription_cancel. A request to reverse an already-posted renewal is "
        "refund unless stopping the recurring plan is the explicit primary ask."
    ),
    "R5_ORDER_MISSING_PRIMARY": (
        "A missing or undelivered package is order_missing when the requested "
        "remedy is locating, tracing, or replacing it. If the user explicitly "
        "requests money back as the sole remedy, label refund."
    ),
    "R6_WRONG_ITEM_PRIMARY": (
        "A delivered package containing the wrong product or variant is wrong_item, "
        "even if the user asks what to do next. It is order_missing only when the "
        "expected package/product was not delivered at all."
    ),
    "R7_ACCOUNT_ACCESS_PRIMARY": (
        "When the requested remedy is unlocking or regaining sign-in access, label "
        "account_locked even if the lock followed a suspicious activity alert."
    ),
}


INTENT_SPEC: dict[str, dict[str, Any]] = {
    "refund": {
        "definition": "The user wants money returned for a completed purchase or posted charge.",
        "positive_criteria": ["asks for a refund, reimbursement, reversal, or money back"],
        "exclusion_criteria": ["only wants to stop an unshipped order", "only reports a missing delivery", "only disputes an unauthorized charge"],
        "common_confusable_intents": ["cancel_order", "subscription_cancel", "order_missing", "fraud_suspected"],
        "representative_easy_examples": ["Please refund order 1842.", "I need my money back for the damaged mug."],
        "representative_hard_examples": ["The parcel never arrived; I am done waiting and want the charge reversed now."],
    },
    "duplicate_charge": {
        "definition": "The same purchase or transaction appears as two or more completed charges.",
        "positive_criteria": ["identifies repeated billing for one purchase", "describes two settled postings or a duplicate debit"],
        "exclusion_criteria": ["one unrecognized charge", "a declined payment with no second completed debit"],
        "common_confusable_intents": ["fraud_suspected", "payment_failed"],
        "representative_easy_examples": ["I was charged twice for order 1842.", "There are two identical card entries."],
        "representative_hard_examples": ["I do not recognize the shop, but the ledger has two settled entries for the same checkout."],
    },
    "order_missing": {
        "definition": "An expected physical order or package has not arrived or cannot be located.",
        "positive_criteria": ["reports non-delivery, a lost shipment, or a delivered-but-missing package", "asks to trace or locate the order"],
        "exclusion_criteria": ["package arrived with an incorrect product", "sole remedy is a refund", "issue is a recurring digital subscription"],
        "common_confusable_intents": ["wrong_item", "refund", "human_escalation"],
        "representative_easy_examples": ["My package never arrived.", "Can you locate order 1842?"],
        "representative_hard_examples": ["Tracking says delivered to the reception desk, but I searched the building and have nothing."],
    },
    "wrong_item": {
        "definition": "A delivered order contains a product, model, size, color, or quantity different from what was ordered.",
        "positive_criteria": ["confirms the package arrived", "identifies the received item as different from the order"],
        "exclusion_criteria": ["the package never arrived", "the product is merely unwanted without a fulfillment error"],
        "common_confusable_intents": ["order_missing", "refund", "human_escalation"],
        "representative_easy_examples": ["You sent the wrong size.", "The box has a blue case, not the black one I ordered."],
        "representative_hard_examples": ["The courier marked it delivered and the box is here, but the contents are a different model."],
    },
    "cancel_order": {
        "definition": "The user wants a one-time physical order stopped before fulfillment or shipment.",
        "positive_criteria": ["asks to cancel a particular pending order", "says the order has not shipped or should be stopped"],
        "exclusion_criteria": ["recurring plan cancellation", "refund for a completed order", "already missing delivery"],
        "common_confusable_intents": ["refund", "subscription_cancel", "human_escalation"],
        "representative_easy_examples": ["Cancel order 1842 before it ships.", "I changed my mind; stop my pending order."],
        "representative_hard_examples": ["It is still in processing. Please halt this shipment; I am not asking for a refund yet."],
    },
    "payment_failed": {
        "definition": "A payment attempt, checkout, or renewal was declined or could not be completed.",
        "positive_criteria": ["describes a failed, declined, rejected, or repeatedly unsuccessful payment", "does not describe two completed charges"],
        "exclusion_criteria": ["two settled debits", "a single unfamiliar charge", "account sign-in lockout"],
        "common_confusable_intents": ["duplicate_charge", "fraud_suspected", "subscription_cancel"],
        "representative_easy_examples": ["My card payment was declined.", "Checkout keeps failing."],
        "representative_hard_examples": ["The first attempt errored and the retry was rejected; I see no completed debit, only failed attempts."],
    },
    "account_locked": {
        "definition": "The user cannot access the account because sign-in has been locked or blocked.",
        "positive_criteria": ["asks to unlock or regain access", "reports lockout after too many attempts or a security event"],
        "exclusion_criteria": ["only reports an unfamiliar charge", "payment checkout failure without login lockout"],
        "common_confusable_intents": ["fraud_suspected", "human_escalation"],
        "representative_easy_examples": ["I am locked out of my account.", "Please unlock my sign-in."],
        "representative_hard_examples": ["After the security alert I can no longer log in; the urgent problem is getting my account unlocked."],
    },
    "subscription_cancel": {
        "definition": "The user wants future recurring subscription or membership billing stopped.",
        "positive_criteria": ["asks to end a recurring plan, membership, or renewal", "focuses on preventing future recurring charges"],
        "exclusion_criteria": ["one-time order cancellation", "refund for an already-posted renewal when cancellation is not requested"],
        "common_confusable_intents": ["cancel_order", "refund", "payment_failed"],
        "representative_easy_examples": ["Cancel my monthly plan.", "Please stop the recurring membership."],
        "representative_hard_examples": ["I saw this month's renewal, but the remedy I need is to stop every future renewal of the plan."],
    },
    "fraud_suspected": {
        "definition": "The user suspects an unauthorized, unfamiliar, or fraudulent account, card, order, or login event.",
        "positive_criteria": ["reports one or more unrecognized activities", "asks to investigate unauthorized activity"],
        "exclusion_criteria": ["two charges for the same known purchase", "primary requested remedy is unlocking access", "ordinary payment decline"],
        "common_confusable_intents": ["duplicate_charge", "account_locked", "payment_failed"],
        "representative_easy_examples": ["I do not recognize this card charge.", "There is an unfamiliar login on my account."],
        "representative_hard_examples": ["The merchant name is unfamiliar and the device location is odd; I see one transaction, not two."],
    },
    "human_escalation": {
        "definition": "The user explicitly requests a human agent, representative, review, or escalation.",
        "positive_criteria": ["asks to speak with a person", "requests escalation or an agent review"],
        "exclusion_criteria": ["merely expresses frustration without an explicit human request"],
        "common_confusable_intents": list(INTENTS[:-1]),
        "representative_easy_examples": ["I need a human agent.", "Please connect me with support."],
        "representative_hard_examples": ["My package is missing and the automated replies failed; connect me to a person who can review it."],
    },
}


def validate_taxonomy() -> None:
    """Fail loudly if the canonical taxonomy is incomplete or inconsistent."""

    if set(INTENT_SPEC) != INTENT_SET:
        raise ValueError("INTENT_SPEC must define exactly the supported intents")
    required = {
        "definition", "positive_criteria", "exclusion_criteria",
        "common_confusable_intents", "representative_easy_examples",
        "representative_hard_examples",
    }
    for intent, spec in INTENT_SPEC.items():
        if set(spec) != required:
            raise ValueError(f"taxonomy fields are incomplete for {intent}")
        if intent in spec["common_confusable_intents"]:
            raise ValueError(f"intent cannot be confusable with itself: {intent}")
    for rule_id in AMBIGUITY_RULES:
        if not rule_id.startswith("R"):
            raise ValueError(f"invalid ambiguity rule ID: {rule_id}")


validate_taxonomy()

