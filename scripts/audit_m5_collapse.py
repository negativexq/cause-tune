#!/usr/bin/env python3
"""Audit the completed M5 order, labels, loss chronology, and contracts.

This script is CPU-only. It reads persisted M5 artifacts and may load the
tokenizer for label decoding, but never loads model or adapter weights.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from transformers import AutoTokenizer

from fineforge.data.preprocess import decode_trainable_assistant, preprocess_records
from fineforge.evaluation_contract import SYSTEM_INSTRUCTION, contract_fingerprint
from fineforge.m5_audit import audit_training_order


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/realistic/train.jsonl")
    parser.add_argument("--output", default="outputs/realistic_qlora_v1/training_order_audit.json")
    parser.add_argument("--loss-history", default="outputs/realistic_qlora_v1/loss_history.json")
    parser.add_argument("--resolved-config", default="outputs/realistic_qlora_v1/resolved_training_config.json")
    parser.add_argument("--evaluation-config", default="configs/baseline_eval.json")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.dataset).read_text().splitlines()]
    resolved = json.loads(Path(args.resolved_config).read_text())
    loss_history = json.loads(Path(args.loss_history).read_text())
    audit = audit_training_order(
        records,
        gradient_accumulation_steps=resolved["training"]["gradient_accumulation_steps"],
        block_size=100,
    )
    if len(loss_history) != len(audit["accumulation_windows"]):
        raise RuntimeError("loss history length does not match optimizer windows")

    loss_by_label: dict[str, list[float]] = defaultdict(list)
    for window, loss_record in zip(audit["accumulation_windows"], loss_history):
        window["loss"] = float(loss_record["loss"])
        window["micro_steps"] = int(loss_record["micro_steps"])
        if window["micro_steps"] != resolved["training"]["gradient_accumulation_steps"]:
            raise RuntimeError(f"unexpected micro-step count at optimizer step {window['optimizer_step']}")
        for label in window["label_counts"]:
            loss_by_label[label].append(float(loss_record["loss"]))

    decoded_by_intent: dict[str, Counter[str]] = defaultdict(Counter)
    decoded_exact_counts: Counter[str] = Counter()
    tokenizer = AutoTokenizer.from_pretrained(resolved["model_id"])
    preprocessed = preprocess_records(
        records,
        tokenizer,
        resolved["training"]["max_sequence_length"],
        system_message=SYSTEM_INSTRUCTION,
    )
    for record, example in zip(records, preprocessed):
        intent = record["expected_response"]["intent"]
        decoded = decode_trainable_assistant(example, tokenizer).strip()
        decoded_by_intent[intent][decoded] += 1
        if decoded.startswith("{\"intent\":\"") and decoded.split("\"")[3] == intent:
            decoded_exact_counts[intent] += 1

    evaluation_config = json.loads(Path(args.evaluation_config).read_text())
    audit.update({
        "loader": {
            "implementation": "torch.utils.data.DataLoader",
            "batch_size": resolved["training"]["micro_batch_size"],
            "shuffle": False,
            "sampler": "implicit_sequential_sampler",
            "split_manifest_order_preserved": True,
            "random_seed": resolved["seed"],
            "order_source": "data/realistic/train.jsonl",
        },
        "loss_history": {
            "optimizer_steps": len(loss_history),
            "per_label": {
                label: {
                    "optimizer_steps": len(values),
                    "first_loss": values[0],
                    "last_loss": values[-1],
                    "mean_loss": mean(values),
                    "min_loss": min(values),
                    "max_loss": max(values),
                }
                for label, values in sorted(loss_by_label.items())
            },
            "final_25_step_label_counts": [
                window["label_counts"] for window in audit["accumulation_windows"][-25:]
            ],
        },
        "label_integrity": {
            "decoded_supervised_completion_counts": {
                intent: dict(counts) for intent, counts in sorted(decoded_by_intent.items())
            },
            "decoded_expected_json_count_per_intent": dict(sorted(decoded_exact_counts.items())),
            "preprocessed_example_count": len(preprocessed),
            "zero_supervised_examples": sum(
                example.trainable_assistant_token_count == 0 for example in preprocessed
            ),
            "all_expected_responses_match_record_labels": all(
                set(counts) == {f'{{"intent":"{intent}"}}<|im_end|>'}
                or all(f'"{intent}"' in text for text in counts)
                for intent, counts in decoded_by_intent.items()
            ),
        },
        "chat_contract": {
            "training_system_instruction": SYSTEM_INSTRUCTION,
            "training_messages": ["system", "user", "assistant"],
            "training_prompt_messages_before_assistant": ["system", "user"],
            "training_prompt_add_generation_prompt": True,
            "training_full_conversation_add_generation_prompt": False,
            "training_enable_thinking": False,
            "training_supervised_portion": "assistant completion only; prompt labels -100",
            "evaluation_system_instruction": evaluation_config["evaluation_contract"]["system_instruction"],
            "evaluation_prompt_messages": ["system", "user"],
            "evaluation_add_generation_prompt": evaluation_config["tokenizer"]["add_generation_prompt"],
            "evaluation_enable_thinking": evaluation_config["tokenizer"]["enable_thinking"],
            "same_system_instruction": (
                SYSTEM_INSTRUCTION == evaluation_config["evaluation_contract"]["system_instruction"]
            ),
            "evaluation_contract_fingerprint": contract_fingerprint(evaluation_config),
        },
        "optimizer": {
            "optimizer": "torch.optim.AdamW",
            "learning_rate": resolved["training"]["learning_rate"],
            "scheduler": "none",
            "weight_decay": 0.01,
            "gradient_clipping": None,
            "warmup": None,
            "optimizer_steps": len(loss_history),
            "learning_rate_constant_in_config": True,
        },
        "notes": [
            "M5 used shuffle=False, so each 8-example accumulation window is sequential.",
            "This audit does not modify the training implementation and is not an M6 run.",
            "The persisted training_metrics.json does not record optimizer hyperparameters beyond learning rate; AdamW defaults are reported from torch.optim.AdamW.",
        ],
    })
    _write_json(Path(args.output), audit)
    print(json.dumps({
        "output": args.output,
        "examples": audit["example_count"],
        "shuffle": audit["loader"]["shuffle"],
        "single_class_windows": audit["single_class_window_count"],
        "mixed_class_windows": audit["mixed_class_window_count"],
        "wrong_item": audit["wrong_item"],
        "decoded_expected_json_count_per_intent": audit["label_integrity"]["decoded_expected_json_count_per_intent"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
