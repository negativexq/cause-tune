"""Teacher-forced and greedy structured evaluation for SFT examples."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from typing import Iterable
from typing import Any, Sequence

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from .data.preprocess import IGNORE_INDEX, PreprocessedExample
from .data.schema import INTENTS, INTENT_SET
from .evaluation_contract import SYSTEM_INSTRUCTION, evaluation_messages
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
    """Parse exactly {"intent": "..."} without repairing model output."""

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict) or set(parsed) != {"intent"}:
        return None
    intent = parsed.get("intent")
    return intent if isinstance(intent, str) and intent in INTENT_SET else None


def diagnose_generation_output(text: str, expected_intent: str) -> str:
    """Classify generation protocol outcomes without repairing official output."""

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        stripped = text.strip()
        try:
            _, end = json.JSONDecoder().raw_decode(stripped)
        except (json.JSONDecodeError, TypeError):
            return "invalid JSON"
        return "unexpected extra text" if stripped[end:].strip() else "invalid JSON"
    if not isinstance(parsed, dict):
        return "invalid JSON"
    if set(parsed) != {"intent"}:
        return "unexpected additional keys"
    intent = parsed.get("intent")
    if not isinstance(intent, str) or intent not in INTENT_SET:
        return "unknown label"
    return "valid JSON + correct intent" if intent == expected_intent else "valid JSON + wrong intent"


def classification_metrics(
    expected: Iterable[str],
    predicted: Iterable[str | None],
    labels: Sequence[str] = INTENTS,
) -> dict[str, Any]:
    """Calculate deterministic multiclass metrics, treating parse failures as wrong."""

    expected_list = list(expected)
    predicted_list = list(predicted)
    if len(expected_list) != len(predicted_list):
        raise ValueError("expected and predicted lengths differ")
    label_set = set(labels)
    matrix = {actual: {label: 0 for label in labels} for actual in labels}
    per_class: dict[str, dict[str, float | int]] = {}
    valid = sum(prediction is not None for prediction in predicted_list)
    correct = sum(actual == prediction for actual, prediction in zip(expected_list, predicted_list))
    for actual, prediction in zip(expected_list, predicted_list):
        if actual not in label_set:
            raise ValueError(f"unknown expected label: {actual}")
        if prediction in label_set:
            matrix[actual][prediction] += 1
    for label in labels:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in labels if actual != label)
        fn = sum(matrix[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "support": sum(matrix[label].values()) + sum(
                1 for actual, prediction in zip(expected_list, predicted_list)
                if actual == label and prediction not in label_set
            ),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    return {
        "intent_accuracy": correct / len(expected_list) if expected_list else 0.0,
        "macro_f1": sum(float(item["f1"]) for item in per_class.values()) / len(labels),
        "valid_json_rate": valid / len(expected_list) if expected_list else 0.0,
        "exact_schema_compliance_rate": valid / len(expected_list) if expected_list else 0.0,
        "per_class": per_class,
        "confusion_matrix": matrix,
        "count": len(expected_list),
    }


def failure_category(
    record: Mapping[str, Any],
    predicted: str | None,
    raw_text: str,
) -> str:
    """Assign a transparent aggregate category to one baseline failure."""

    phenomena = set(record.get("phenomena", []))
    if predicted is None:
        return "json/schema failure"
    if "multi_issue" in phenomena:
        return "multi-issue error"
    if record.get("label_rule") and predicted in record.get("confusable_with", []):
        return "ambiguity-rule error"
    if "typo" in phenomena:
        return "typo/noise sensitivity"
    if "irrelevant_context" in phenomena or "technical_distraction" in phenomena:
        return "irrelevant-context distraction"
    if record.get("split") == "ood_test":
        return "OOD phrasing failure"
    if predicted in record.get("confusable_with", []):
        return "wrong confusable intent"
    return "other classification error"


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
    batch_size: int = 8,
    system_message: str | None = None,
) -> dict[str, float | int]:
    """Evaluate structured intent output with deterministic greedy decoding."""

    if not records:
        raise ValueError("cannot generate metrics for an empty split")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    device = _model_device(model)
    correct = 0
    valid_json = 0
    predictions: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start : start + batch_size]
            prompts = [
                _prompt_ids(
                    tokenizer,
                    evaluation_messages(record, system_message or SYSTEM_INSTRUCTION)
                    if system_message is not None
                    else record["messages"][:1],
                )
                for record in batch_records
            ]
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.full(
                (len(prompts), width),
                fill_value=tokenizer.pad_token_id,
                dtype=torch.long,
                device=device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
                attention_mask[row, width - len(prompt) :] = 1
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
            for record, output in zip(batch_records, generated):
                text = tokenizer.decode(output[width:], skip_special_tokens=True)
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
                        "diagnostic_type": diagnose_generation_output(text, expected_intent),
                        "failure_category": (
                            None
                            if predicted_intent == expected_intent
                            else failure_category(record, predicted_intent, text)
                        ),
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
    system_message: str | None = None,
    max_new_tokens: int = 32,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Combine loss and deterministic structured metrics for one split."""

    teacher_forced = teacher_forced_metrics(model, tokenizer, examples)
    greedy = greedy_intent_metrics(
        model,
        tokenizer,
        records,
        max_new_tokens=max_new_tokens,
        batch_size=batch_size,
        system_message=system_message,
    )
    expected = [record["expected_response"]["intent"] for record in records]
    predicted = [item["parsed_intent"] for item in greedy["predictions"]]
    structured = classification_metrics(expected, predicted)
    diagnostic_counts = dict(Counter(
        prediction["diagnostic_type"] for prediction in greedy["predictions"]
    ))
    return {
        **teacher_forced,
        **structured,
        "generation_examples": greedy["generation_examples"],
        "generation_diagnostic_counts": diagnostic_counts,
        "predictions": greedy["predictions"],
    }


def aggregate_failures(
    split: str,
    records: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Return per-example failures and category counts for an evaluated split."""

    failures: list[dict[str, Any]] = []
    categories: Counter[str] = Counter()
    for record, prediction in zip(records, predictions):
        if prediction["correct"]:
            continue
        category = str(prediction.get("failure_category") or "other classification error")
        categories[category] += 1
        failures.append({
            "example_id": record["example_id"],
            "split": split,
            "expected_intent": record["expected_response"]["intent"],
            "predicted_intent": prediction.get("parsed_intent"),
            "raw_model_output": prediction.get("raw_text", ""),
            "diagnostic_type": prediction.get("diagnostic_type"),
            "difficulty": record["difficulty"],
            "phenomena": record["phenomena"],
            "confusable_with": record["confusable_with"],
            "scenario_family": record["scenario_family"],
            "label_rule": record.get("label_rule"),
            "failure_category": category,
        })
    return failures, dict(categories)
