"""Controlled, deterministic surface perturbations for M4 records."""

from __future__ import annotations

import re


PHENOMENA = frozenset({
    "short_request", "long_form", "informal", "typo", "punctuation_error",
    "incomplete_sentence", "irrelevant_context", "order_id", "transaction_id",
    "amount", "date", "multiple_sentences", "indirect_wording", "frustrated",
    "polite", "uncertain", "incorrect_terminology", "prior_failed_attempt",
    "confusable_intent", "multi_issue", "colloquial", "narrative", "chat_compressed",
    "unusual_ordering", "technical_distraction",
})


def apply_perturbations(text: str, phenomena: set[str], index: int) -> str:
    """Apply small natural-language changes without changing the gold label."""

    result = text.strip()
    if "polite" in phenomena:
        result = ("Hi support, could you please help? " if index % 2 == 0 else "Please help me with this. ") + result
    if "frustrated" in phenomena:
        result = ("This is really frustrating. " if index % 2 == 0 else "I have already spent too long on this. ") + result
    if "uncertain" in phenomena:
        result = ("I may be misunderstanding the statement, but " if index % 2 == 0 else "I think something is wrong: ") + result
    if "prior_failed_attempt" in phenomena:
        result += " I already tried the help-center steps and the earlier ticket did not resolve it."
    if "irrelevant_context" in phenomena:
        result = result + (" I was commuting when I noticed it, and my phone battery was nearly empty." if index % 2 == 0 else " The app had just updated and I was using hotel Wi-Fi at the time.")
    if "long_form" in phenomena:
        result = ("For context, I checked the order history, my email receipt, and the status page before writing. " + result + " I would appreciate a clear next step rather than another generic article.")
    if "multiple_sentences" in phenomena and "." not in result:
        result += ". Please tell me what happens next"
    if "informal" in phenomena:
        result = result.replace("Please", "pls").replace("I need", "need").replace("because", "bc")
        result = result.replace("Can you", "can u")
    if "incorrect_terminology" in phenomena:
        result += " I might be calling the reference number the wrong thing, but it is the code in my receipt."
    if "typo" in phenomena:
        replacements = {"payment": "paymant", "account": "acount", "package": "pakage", "refund": "refnd", "subscription": "subscrption", "received": "recieved", "transaction": "transacton"}
        for source, target in replacements.items():
            if source in result.lower():
                pattern = re.compile(re.escape(source), re.IGNORECASE)
                result = pattern.sub(target, result, count=1)
                break
    if "punctuation_error" in phenomena:
        result = result.replace(",", "").replace(".", "").replace("?", "")
    if "incomplete_sentence" in phenomena:
        words = result.split()
        result = " ".join(words[:-2]) if len(words) > 8 else result.rstrip(".")
    return result.strip()

