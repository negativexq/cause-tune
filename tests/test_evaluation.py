from __future__ import annotations

import unittest

from causetune.data.generate import generate_smoke_records
from causetune.data.preprocess import build_preprocessed_example
from causetune.evaluation import collate_preprocessed, parse_structured_intent

from test_preprocess import FakeChatTokenizer


class EvaluationTests(unittest.TestCase):
    def test_parser_does_not_repair_invalid_json(self) -> None:
        self.assertEqual(parse_structured_intent('{"intent":"refund"}'), "refund")
        self.assertIsNone(parse_structured_intent("not json"))
        self.assertIsNone(parse_structured_intent('{"intent":"refund"} trailing'))
        self.assertIsNone(parse_structured_intent('["refund"]'))
        self.assertIsNone(parse_structured_intent('{"wrong_key":"refund"}'))

    def test_collator_preserves_preprocessed_labels(self) -> None:
        tokenizer = FakeChatTokenizer()
        records = generate_smoke_records(seed=42)[:2]
        examples = [
            build_preprocessed_example(record, tokenizer, max_sequence_length=1024)
            for record in records
        ]
        batch = collate_preprocessed(examples, pad_token_id=0)
        self.assertEqual(tuple(batch["input_ids"].shape), tuple(batch["labels"].shape))
        self.assertEqual(tuple(batch["input_ids"].shape), tuple(batch["attention_mask"].shape))
        for row, example in enumerate(examples):
            length = len(example.input_ids)
            self.assertEqual(
                batch["labels"][row, :length].tolist(),
                list(example.labels),
            )
            self.assertTrue((batch["labels"][row, length:] == -100).all())

