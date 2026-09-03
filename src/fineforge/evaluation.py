"""Teacher-forced and greedy structured evaluation for SFT examples."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .data.preprocess import IGNORE_INDEX, PreprocessedExample
from .telemetry import synchronize_cuda


def collate_preprocessed(
    examples: Sequence[PreprocessedExample],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pad already-preprocessed examples without changing their labels."""

    max_length = max(len(example.input_ids) for example in examples)
    input_ids = torch.full(
        (len(examples), max_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    labels = torch.full(
        (len(examples), max_length),
        fill_value=IGNORE_INDEX,
        dtype=torch.long,
    )
    attention_mask = torch.zeros((len(examples), max_length), dtype=torch.long)
    for row, example in enumerate(examples):
        length = len(example.input_ids)
        input_ids[row, :length] = torch.tensor(example.input_ids, dtype=torch.long)
        labels[row, :length] = torch.tensor(example.labels, dtype=torch.long)
        attention_mask[row, :length] = 1
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def _model_device(model: Any) -> torch.device:
    return next(model.parameters()).device


def teacher_forced_metrics(
    model: Any,
    tokenizer: Any,
    examples: Sequence[PreprocessedExample],
) -> dict[str, float | int]:
    """Compute assistant-only causal loss over a preprocessed split."""

    if not examples:
        raise ValueError("cannot evaluate an empty split")
    model.eval()
    loader = DataLoader(
        list(examples),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda batch: collate_preprocessed(batch, tokenizer.pad_token_id),
    )
    loss_function = CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction="sum")
    total_loss = 0.0
    total_tokens = 0
    device = _model_device(model)
    with torch.inference_mode():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            total_loss += float(
                loss_function(
                    shift_logits.float().view(-1, shift_logits.shape[-1]),
                    shift_labels.view(-1),
                )
            )
            total_tokens += int((shift_labels != IGNORE_INDEX).sum())
    return {
        "teacher_forced_loss": total_loss / total_tokens,
        "assistant_loss_tokens": total_tokens,
    }


def parse_structured_intent(text: str) -> str | None:
    """Parse JSON output without repairing malformed or extra-text responses."""

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    intent = parsed.get("intent")
    return intent if isinstance(intent, str) else None


def _prompt_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    try:
        tokenized = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        tokenized = tokenizer.apply_chat_template(
            list(messages),
            tokenize=True,
            add_generation_prompt=True,
        )
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if isinstance(tokenized, Mapping):
        tokenized = tokenized["input_ids"]
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    return [int(token_id) for token_id in tokenized]


def greedy_intent_metrics(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    max_new_tokens: int = 32,
) -> dict[str, float | int]:
    """Evaluate structured intent output with deterministic greedy decoding."""

    if not records:
        raise ValueError("cannot generate metrics for an empty split")
    model.eval()
    device = _model_device(model)
    correct = 0
    valid_json = 0
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for record in records:
            prompt = _prompt_ids(tokenizer, record["messages"][:1])
            input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            synchronize_cuda()
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            synchronize_cuda()
            text = tokenizer.decode(
                generated[0, input_ids.shape[1] :],
                skip_special_tokens=True,
            )
            predicted_intent = parse_structured_intent(text)
            is_valid = predicted_intent is not None
            valid_json += int(is_valid)
            expected_intent = record["expected_response"]["intent"]
            correct += int(predicted_intent == expected_intent)
            predictions.append(
                {
                    "example_id": record["example_id"],
                    "raw_text": text,
                    "parsed_intent": predicted_intent,
                    "expected_intent": expected_intent,
                    "valid_json": is_valid,
                    "correct": predicted_intent == expected_intent,
                }
            )
    total = len(records)
    return {
        "intent_accuracy": correct / total,
        "valid_json_rate": valid_json / total,
        "generation_examples": total,
        "predictions": predictions,
    }


def evaluate_split(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    examples: Sequence[PreprocessedExample],
) -> dict[str, Any]:
    """Combine loss and deterministic structured metrics for one split."""

    teacher_forced = teacher_forced_metrics(model, tokenizer, examples)
    greedy = greedy_intent_metrics(model, tokenizer, records)
    return {
        **teacher_forced,
        "intent_accuracy": greedy["intent_accuracy"],
        "valid_json_rate": greedy["valid_json_rate"],
        "generation_examples": greedy["generation_examples"],
        "predictions": greedy["predictions"],
    }
