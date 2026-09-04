"""Semantic scenario catalog used by the local M4 surface generator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    intent: str
    subject: str
    confusable_with: tuple[str, ...] = ()
    label_rule: str | None = None
    multi_issue: bool = False
    hard_negative: bool = False


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("refund_damaged", "refund", "a damaged ceramic mug", ("wrong_item",)),
    Scenario("refund_unwanted", "refund", "a recent purchase I no longer need", ("cancel_order",)),
    Scenario("refund_missing_delivery", "refund", "order {order_id} that never arrived", ("order_missing",), "R5_ORDER_MISSING_PRIMARY", True, True),
    Scenario("refund_renewal", "refund", "the renewal charged on {date}", ("subscription_cancel",), "R4_SUBSCRIPTION_CANCELLATION_PRIMARY", True, True),

    Scenario("duplicate_same_order", "duplicate_charge", "order {order_id}", ("fraud_suspected",), "R2_DUPLICATE_OVER_FRAUD", hard_negative=True),
    Scenario("duplicate_retry", "duplicate_charge", "the checkout retry for {order_id}", ("payment_failed",), "R3_DUPLICATE_OVER_PAYMENT_FAILED", True, True),
    Scenario("duplicate_card_ledger", "duplicate_charge", "transaction {transaction_id}", ("fraud_suspected",), "R2_DUPLICATE_OVER_FRAUD", hard_negative=True),
    Scenario("duplicate_pending_settled", "duplicate_charge", "the purchase from {merchant}", ("payment_failed",), "R3_DUPLICATE_OVER_PAYMENT_FAILED", True, True),

    Scenario("missing_late", "order_missing", "order {order_id}", ("refund",)),
    Scenario("missing_delivered", "order_missing", "shipment marked delivered on {date}", ("wrong_item",), "R5_ORDER_MISSING_PRIMARY", True, True),
    Scenario("missing_tracking_stall", "order_missing", "the package from {merchant}", ("human_escalation",)),
    Scenario("missing_address", "order_missing", "the delivery for order {order_id}", ("refund",), "R5_ORDER_MISSING_PRIMARY", True, True),

    Scenario("wrong_color", "wrong_item", "a blue case instead of the black case ordered", ("order_missing",), "R6_WRONG_ITEM_PRIMARY", True, True),
    Scenario("wrong_size", "wrong_item", "a medium shirt instead of the large size", ("order_missing",)),
    Scenario("wrong_model", "wrong_item", "a different laptop model in the delivered box", ("order_missing",), "R6_WRONG_ITEM_PRIMARY", True, True),
    Scenario("wrong_substitute", "wrong_item", "an unrequested substitute from {merchant}", ("refund",)),

    Scenario("cancel_pending", "cancel_order", "pending order {order_id}", ("refund",), "R5_ORDER_MISSING_PRIMARY", hard_negative=True),
    Scenario("cancel_processing", "cancel_order", "the order placed this morning", ("refund",), "R5_ORDER_MISSING_PRIMARY", hard_negative=True),
    Scenario("cancel_gift", "cancel_order", "the shipment intended as a gift", ("wrong_item",)),
    Scenario("cancel_unwanted", "cancel_order", "one-time purchase {order_id}", ("subscription_cancel",), hard_negative=True),

    Scenario("payment_declined", "payment_failed", "the checkout for order {order_id}", ("duplicate_charge",)),
    Scenario("payment_retry_failed", "payment_failed", "a retry at {merchant}", ("duplicate_charge",), "R3_DUPLICATE_OVER_PAYMENT_FAILED", True, True),
    Scenario("payment_renewal_failed", "payment_failed", "the subscription renewal", ("subscription_cancel",)),
    Scenario("payment_no_debit", "payment_failed", "the card payment for {merchant}", ("duplicate_charge",), "R3_DUPLICATE_OVER_PAYMENT_FAILED", True, True),

    Scenario("account_attempts", "account_locked", "my account after too many sign-in attempts", ("fraud_suspected",)),
    Scenario("account_security_lock", "account_locked", "my profile after a security alert", ("fraud_suspected",), "R7_ACCOUNT_ACCESS_PRIMARY", True, True),
    Scenario("account_password", "account_locked", "my account after changing the password", ("fraud_suspected",)),
    Scenario("account_device", "account_locked", "my account on a new phone", ("fraud_suspected",), "R7_ACCOUNT_ACCESS_PRIMARY", True, True),

    Scenario("subscription_monthly", "subscription_cancel", "the monthly premium plan", ("cancel_order",)),
    Scenario("subscription_membership", "subscription_cancel", "my recurring membership", ("refund",), "R4_SUBSCRIPTION_CANCELLATION_PRIMARY", True, True),
    Scenario("subscription_renewal_future", "subscription_cancel", "future renewals on {date}", ("refund",), "R4_SUBSCRIPTION_CANCELLATION_PRIMARY", True, True),
    Scenario("subscription_annual", "subscription_cancel", "the annual service", ("cancel_order",)),

    Scenario("fraud_card", "fraud_suspected", "one unfamiliar card charge from {merchant}", ("duplicate_charge",), "R2_DUPLICATE_OVER_FRAUD", hard_negative=True),
    Scenario("fraud_login", "fraud_suspected", "an unrecognized login from a new device", ("account_locked",), "R7_ACCOUNT_ACCESS_PRIMARY", True, True),
    Scenario("fraud_order", "fraud_suspected", "an order I did not place", ("order_missing",)),
    Scenario("fraud_transfer", "fraud_suspected", "transaction {transaction_id} that I do not recognize", ("duplicate_charge",), "R2_DUPLICATE_OVER_FRAUD", hard_negative=True),

    Scenario("human_billing", "human_escalation", "a billing dispute", ("refund", "duplicate_charge"), "R1_HUMAN_ESCALATION", True, True),
    Scenario("human_missing", "human_escalation", "my missing package", ("order_missing",), "R1_HUMAN_ESCALATION", True, True),
    Scenario("human_wrong_item", "human_escalation", "the wrong item delivered", ("wrong_item",), "R1_HUMAN_ESCALATION", True, True),
    Scenario("human_security", "human_escalation", "a suspicious account event", ("fraud_suspected", "account_locked"), "R1_HUMAN_ESCALATION", True, True),
)


SCENARIOS_BY_INTENT: dict[str, tuple[Scenario, ...]] = {}
for _scenario in SCENARIOS:
    SCENARIOS_BY_INTENT.setdefault(_scenario.intent, tuple())
    SCENARIOS_BY_INTENT[_scenario.intent] += (_scenario,)

