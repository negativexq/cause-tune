"""Chat-template tokenization and assistant-only causal-LM labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .schema import validate_record


IGNORE_INDEX = -100


class PreprocessingError(ValueError):
    """Raised when tokenization cannot safely produce supervised labels."""


class ChatTemplatePrefixError(PreprocessingError):
    """Raised when the prompt tokenization is not a full-conversation prefix."""


class SequenceLengthError(PreprocessingError):
    """Raised when the full assistant completion cannot fit the configured limit."""


@dataclass(frozen=True)
class PreprocessedExample:
    example_id: str
    messages: tuple[dict[str, str], ...]
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    attention_mask: tuple[int, ...]
    prompt_token_count: int
    masked_token_count: int
    trainable_assistant_token_count: int
    rendered_conversation: str


def _token_ids(value: Any) -> list[int]:
    """Normalize tokenizer output without requiring a tensor dependency."""

    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        raise PreprocessingError("tokenizer did not return a list of input IDs")
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise PreprocessingError("tokenizer returned more than one sequence")
        value = value[0]
    if not all(isinstance(token_id, int) for token_id in value):
        raise PreprocessingError("tokenizer returned non-integer input IDs")
    return value


def _chat_template_ids(
    tokenizer: Any,
    messages: Sequence[Mapping[str, str]],
    add_generation_prompt: bool,
) -> list[int]:
    try:
        try:
            tokenized = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            # Small tokenizer doubles and older compatible tokenizers may not
            # expose Qwen's optional official thinking switch.
            tokenized = tokenizer.apply_chat_template(
                list(messages),
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
    except TypeError as exc:
        raise PreprocessingError(
            "tokenizer must support apply_chat_template(..., tokenize=True, "
            "add_generation_prompt=...)"
        ) from exc
    return _token_ids(tokenized)


def _render_chat_template(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    try:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        rendered = tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
        )
    if not isinstance(rendered, str):
        raise PreprocessingError("tokenizer did not return rendered chat text")
    return rendered


def build_preprocessed_example(
    record: Mapping[str, Any],
    tokenizer: Any,
    max_sequence_length: int,
) -> PreprocessedExample:
    """Create labels that train only on the assistant completion.

    The assistant boundary is derived by comparing tokenized chat-template
    outputs, never by searching for a hard-coded assistant token or string.
    """

    if max_sequence_length <= 0:
        raise ValueError("max_sequence_length must be positive")
    validated = validate_record(dict(record))
    messages = tuple(validated["messages"])
    prompt_ids = _chat_template_ids(tokenizer, messages[:1], add_generation_prompt=True)
    full_ids = _chat_template_ids(tokenizer, messages, add_generation_prompt=False)

    if len(prompt_ids) >= len(full_ids):
        raise PreprocessingError(
            f"{validated['example_id']}: chat template produced no assistant response tokens"
        )
    if full_ids[: len(prompt_ids)] != prompt_ids:
        mismatch = next(
            (
                index
                for index, (prompt_token, full_token) in enumerate(
                    zip(prompt_ids, full_ids)
                )
                if prompt_token != full_token
            ),
            min(len(prompt_ids), len(full_ids)),
        )
        raise ChatTemplatePrefixError(
            f"{validated['example_id']}: tokenized user prompt is not a prefix of "
            f"the full conversation at token {mismatch}; refusing to guess the boundary"
        )
    if len(full_ids) > max_sequence_length:
        raise SequenceLengthError(
            f"{validated['example_id']}: full conversation has {len(full_ids)} tokens, "
            f"exceeding max_sequence_length={max_sequence_length}; refusing to "
            "truncate the assistant completion"
        )

    labels = [IGNORE_INDEX] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    trainable_count = sum(label != IGNORE_INDEX for label in labels)
    if any(label != IGNORE_INDEX for label in labels[: len(prompt_ids)]):
        raise AssertionError(f"{validated['example_id']}: prompt labels are not fully masked")
    if trainable_count == 0:
        raise PreprocessingError(
            f"{validated['example_id']}: truncation left zero trainable assistant tokens"
        )
    if len(full_ids) != len(labels):
        raise AssertionError(f"{validated['example_id']}: input_ids/labels length mismatch")

    return PreprocessedExample(
        example_id=validated["example_id"],
        messages=messages,
        input_ids=tuple(full_ids),
        labels=tuple(labels),
        attention_mask=tuple(1 for _ in full_ids),
        prompt_token_count=len(prompt_ids),
        masked_token_count=sum(label == IGNORE_INDEX for label in labels),
        trainable_assistant_token_count=trainable_count,
        rendered_conversation=_render_chat_template(tokenizer, messages),
    )


def preprocess_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    max_sequence_length: int,
) -> list[PreprocessedExample]:
    """Preprocess records and fail on the first unsafe example."""

    return [
        build_preprocessed_example(record, tokenizer, max_sequence_length)
        for record in records
    ]


def decode_trainable_assistant(example: PreprocessedExample, tokenizer: Any) -> str:
    """Decode only token IDs whose labels contribute to the loss."""

    trainable_ids = [
        token_id
        for token_id, label in zip(example.input_ids, example.labels)
        if label != IGNORE_INDEX
    ]
    return tokenizer.decode(trainable_ids, skip_special_tokens=False)
