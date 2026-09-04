from __future__ import annotations

import unittest

from causetune.data.generate import generate_smoke_records
from causetune.data.schema import validate_records
from causetune.data.split import split_ids, split_records


class SplitTests(unittest.TestCase):
    def test_split_counts_reproducibility_and_non_overlap(self) -> None:
        records = validate_records(generate_smoke_records(seed=42))
        first = split_records(records, seed=42)
        second = split_records(records, seed=42)
        self.assertEqual(split_ids(first), split_ids(second))
        self.assertEqual(
            {name: len(items) for name, items in first.items()},
            {"train": 160, "validation": 20, "test": 20},
        )
        all_ids = [record["example_id"] for items in first.values() for record in items]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(all_ids), 200)
        self.assertEqual(set(all_ids), {record["example_id"] for record in records})

        all_intents = {record["expected_response"]["intent"] for record in records}
        for items in first.values():
            self.assertEqual(
                {record["expected_response"]["intent"] for record in items},
                all_intents,
            )
