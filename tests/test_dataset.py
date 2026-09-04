from __future__ import annotations

import copy
import unittest

from causetune.data.generate import generate_smoke_records
from causetune.data.schema import INTENTS, SchemaValidationError, validate_records


class DatasetTests(unittest.TestCase):
    def test_dataset_has_exact_counts_and_unique_ids(self) -> None:
        records = validate_records(generate_smoke_records(seed=42))
        self.assertEqual(len(records), 200)
        self.assertEqual(len({record["example_id"] for record in records}), 200)
        counts = {intent: 0 for intent in INTENTS}
        for record in records:
            counts[record["expected_response"]["intent"]] += 1
        self.assertEqual(counts, {intent: 20 for intent in INTENTS})

    def test_generation_is_deterministic_and_seeded(self) -> None:
        self.assertEqual(generate_smoke_records(seed=42), generate_smoke_records(seed=42))
        self.assertNotEqual(generate_smoke_records(seed=42), generate_smoke_records(seed=43))

    def test_schema_validation_catches_malformed_records(self) -> None:
        mutators = [
            lambda record: record["messages"].__setitem__(
                0, {"role": "assistant", "content": '{"intent":"refund"}'}
            ),
            lambda record: record["messages"].__setitem__(
                1, {"role": "user", "content": "missing assistant"}
            ),
            lambda record: record["messages"].__setitem__(
                0, {"role": "system", "content": "x"}
            ),
            lambda record: record["messages"].__setitem__(
                1, {"role": "assistant", "content": ""}
            ),
            lambda record: record["messages"].__setitem__(
                1, {"role": "assistant", "content": "not json"}
            ),
            lambda record: record["expected_response"].__setitem__("intent", "unknown"),
            lambda record: record["messages"].__setitem__(
                1, {"role": "assistant", "content": '{"intent":"duplicate_charge"}'}
            ),
        ]
        for mutator in mutators:
            with self.subTest(mutator=mutator):
                malformed = copy.deepcopy(generate_smoke_records(seed=42)[0])
                mutator(malformed)
                with self.assertRaises(SchemaValidationError):
                    validate_records([malformed])

        records = generate_smoke_records(seed=42)
        records[1]["example_id"] = records[0]["example_id"]
        with self.assertRaises(SchemaValidationError):
            validate_records(records[:2])
