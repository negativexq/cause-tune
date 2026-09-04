from __future__ import annotations

import unittest

from causetune.data.generate import generate_smoke_records
from causetune.data.preprocess import (
    IGNORE_INDEX,
    ChatTemplatePrefixError,
    SequenceLengthError,
    build_preprocessed_example,
)


class FakeChatTokenizer:
    """Small tokenizer double; tests never load a real model or tokenizer."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = ""
        for message in messages:
            rendered += f"<{message['role']}>{message['content']}"
        if add_generation_prompt:
            rendered += "<assistant>"
        else:
            rendered += "<eos>"
        if tokenize:
            return [ord(character) for character in rendered]
        return rendered

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class NonPrefixTokenizer(FakeChatTokenizer):
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        rendered = super().apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        if tokenize and not add_generation_prompt:
            rendered = rendered.replace("<user>", "<USER>")
        return [ord(character) for character in rendered] if tokenize else rendered


class PreprocessTests(unittest.TestCase):
    def test_assistant_only_masking_and_lengths(self) -> None:
        record = generate_smoke_records(seed=42)[0]
        example = build_preprocessed_example(
            record, FakeChatTokenizer(), max_sequence_length=1024
        )
        boundary = example.prompt_token_count
        self.assertTrue(all(label == IGNORE_INDEX for label in example.labels[:boundary]))
        self.assertTrue(all(label != IGNORE_INDEX for label in example.labels[boundary:]))
        self.assertGreater(example.trainable_assistant_token_count, 0)
        self.assertEqual(len(example.input_ids), len(example.labels))
        self.assertEqual(len(example.input_ids), len(example.attention_mask))
        self.assertEqual(example.masked_token_count, boundary)

    def test_non_prefix_tokenization_fails_loudly(self) -> None:
        record = generate_smoke_records(seed=42)[0]
        with self.assertRaisesRegex(ChatTemplatePrefixError, "not a prefix"):
            build_preprocessed_example(
                record, NonPrefixTokenizer(), max_sequence_length=1024
            )

    def test_max_length_rejects_assistant_truncation(self) -> None:
        record = generate_smoke_records(seed=42)[0]
        tokenizer = FakeChatTokenizer()
        full_length = len(
            tokenizer.apply_chat_template(
                record["messages"], tokenize=True, add_generation_prompt=False
            )
        )
        with self.assertRaisesRegex(SequenceLengthError, "refusing to truncate"):
            build_preprocessed_example(record, tokenizer, max_sequence_length=full_length - 1)
