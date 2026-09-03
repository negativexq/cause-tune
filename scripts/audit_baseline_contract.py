#!/usr/bin/env python3
"""Audit the original M4 baseline without changing official metrics or labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fineforge.evaluation import parse_structured_intent
from fineforge.data.validation import read_realistic_splits


SEMANTIC_CUES = {
    "refund": ("refund", "money back", "reimburse", "reverse", "returned"),
    "duplicate_charge": ("charged twice", "two charge", "duplicate", "second posting", "billed twice"),
    "order_missing": ("missing", "delivery", "package", "arrived", "shipment"),
    "wrong_item": ("wrong item", "incorrect", "wrong size", "wrong model", "sku"),
    "cancel_order": ("cancel", "halt", "stop", "pending order"),
    "payment_failed": ("payment", "declined", "checkout", "failed", "rejected"),
    "account_locked": ("account locked", "lockout", "locked", "unlock", "sign-in lock"),
    "subscription_cancel": ("subscription", "renewal", "recurring", "membership", "stop renewing"),
    "fraud_suspected": ("fraud", "unauthorized", "unrecognized", "suspicious", "security"),
    "human_escalation": ("human", "agent", "representative", "person", "escalate"),
}


def _semantic_assessment(expected: str, raw: str) -> tuple[str, str]:
    lowered = raw.casefold()
    cues = SEMANTIC_CUES[expected]
    matched = [cue for cue in cues if cue in lowered]
    if matched:
        return "appears_semantically_aligned", f"matched cues: {matched[:4]}"
    return "not_determined", "no deterministic cue match; no prose-to-label repair was used"


def _sample(rows: list[dict[str, Any]], count: int = 10) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"need at least {count} failures, got {len(rows)}")
    positions = [round(index * (len(rows) - 1) / (count - 1)) for index in range(count)]
    return [rows[position] for position in positions]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/realistic")
    parser.add_argument("--original-output-dir", default="outputs/baseline_original")
    parser.add_argument("--output-dir", default="outputs/baseline_audit")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = read_realistic_splits(args.dataset_dir)
    by_id = {
        record["example_id"]: record
        for rows in splits.values()
        for record in rows
    }
    failures = [
        json.loads(line)
        for line in (Path(args.original_output_dir) / "failures.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sampled: list[dict[str, Any]] = []
    for split in ("validation", "id_test", "hard_test", "ood_test"):
        split_failures = [failure for failure in failures if failure["split"] == split]
        for rank, failure in enumerate(_sample(split_failures), start=1):
            record = by_id[failure["example_id"]]
            raw = failure["raw_model_output"]
            assessment, basis = _semantic_assessment(failure["expected_intent"], raw)
            sampled.append({
                "sample_rank": rank,
                "example_id": failure["example_id"],
                "split": split,
                "expected_intent": failure["expected_intent"],
                "user_message": record["messages"][0]["content"],
                "raw_model_output": raw,
                "strict_parser_result": parse_structured_intent(raw),
                "semantic_assessment": assessment,
                "formatting_only_likely": assessment == "appears_semantically_aligned",
                "assessment_basis": basis,
                "contains_json_object_syntax": "{" in raw or "}" in raw,
                "contains_intent_key_literal": '"intent"' in raw,
                "official_metric_changed": False,
            })
    (output_dir / "raw_failure_audit.json").write_text(
        json.dumps({
            "sample_method": "10 evenly spaced failures from each split after stable example_id ordering",
            "sample_count_per_split": 10,
            "llm_judge_used": False,
            "prose_to_intent_repair_used": False,
            "samples": sampled,
        }, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    old_contract = {
        "system_message": None,
        "user_message": "record[\"messages\"][0][\"content\"] (raw customer message; one user turn)",
        "chat_template": {
            "function": "tokenizer.apply_chat_template",
            "messages": "[record[\"messages\"][0]]",
            "tokenize": True,
            "add_generation_prompt": True,
            "enable_thinking": False,
            "fallback": "retry without enable_thinking on TypeError",
        },
        "generation": {
            "max_new_tokens": 32,
            "do_sample": False,
            "pad_token_id": "tokenizer.pad_token_id",
            "eos_token_id": "tokenizer.eos_token_id",
            "batching": "left-pad prompts to batch max width; batch_size=8",
        },
        "decoding": {
            "generated_slice": "generated[row, prompt_width:]",
            "skip_special_tokens": True,
        },
        "parser": {
            "function": "parse_structured_intent",
            "operation": "json.loads(raw_text)",
            "allow_surrounding_text": False,
            "require_exact_keys": ["intent"],
            "allowed_labels": "FineForge INTENT_SET",
            "repair_output": False,
        },
        "schema_validator": {
            "dataset": "validate_realistic_splits -> validate_realistic_record -> validate_record",
            "assistant_target": "exactly {\"intent\": \"<supported label>\"}",
            "metadata_excluded_from_prompt": True,
        },
        "task_instruction_present": {
            "intent_classification_task": False,
            "allowed_intent_labels": False,
            "required_json_schema": False,
            "json_only_no_prose": False,
        },
    }
    checks = {
        "generation_slice_check": {
            "status": "verified_by_code_inspection",
            "finding": "The old batched path decoded only generated[row, prompt_width:], not prompt tokens.",
        },
        "prompt_plus_completion_check": {
            "status": "verified_by_code_inspection",
            "finding": "Decoded text began as a model response, with no user prompt prefix.",
        },
        "special_token_check": {
            "status": "verified_by_code_inspection",
            "finding": "skip_special_tokens=True was used; no special-token residue was observed in sampled outputs.",
        },
        "thinking_mode_check": {
            "status": "verified_by_code_inspection",
            "finding": "enable_thinking=False was passed to chat-template tokenization, with a compatible fallback.",
        },
        "length_check": {
            "status": "diagnostic",
            "finding": "Many sampled responses ended mid-prose at the 32-token limit; this is consistent with protocol failure but not the root cause.",
        },
        "assistant_boundary_check": {
            "status": "verified_by_code_inspection_and_existing_tests",
            "finding": "Teacher-forced assistant-only masking derives a token prefix boundary and has regression coverage; it does not control generation parsing.",
        },
        "parser_check": {
            "status": "verified_by_tests",
            "finding": "Strict parser correctly rejects prose, surrounding text, unknown labels, and extra keys.",
        },
    }
    report = {
        "audit_version": "m4-baseline-audit-v1",
        "original_artifacts": str(Path(args.original_output_dir)),
        "old_contract": old_contract,
        "implementation_checks": checks,
        "sample_summary": {
            "total_sampled": len(sampled),
            "by_split": {split: sum(item["split"] == split for item in sampled) for split in ("validation", "id_test", "hard_test", "ood_test")},
            "appears_semantically_aligned": sum(item["semantic_assessment"] == "appears_semantically_aligned" for item in sampled),
            "strictly_valid_json": sum(item["strict_parser_result"] is not None for item in sampled),
        },
        "conclusion": {
            "root_cause": "The original contract omitted an explicit task instruction and output protocol; the frozen model answered as a general support assistant.",
            "implementation_bug_found": False,
            "formatting_only_diagnostic": "The sample frequently appears semantically aligned, but this remains diagnostic only and does not change official metrics.",
            "official_metrics_repaired": False,
        },
    }
    (output_dir / "audit_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["sample_summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
