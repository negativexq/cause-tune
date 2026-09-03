from __future__ import annotations

import copy
import unittest

from fineforge.data.leakage import cross_split_leakage, duplicate_groups, normalize_text
from fineforge.data.realistic_generate import generate_realistic_splits
from fineforge.data.statistics import dataset_statistics
from fineforge.data.taxonomy import AMBIGUITY_RULES, INTENT_SPEC, validate_taxonomy
from fineforge.data.validation import validate_realistic_record, validate_realistic_splits
from fineforge.evaluation import classification_metrics, parse_structured_intent


class M4DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.splits = generate_realistic_splits(20260904)

    def test_deterministic_generation_and_stable_ids(self) -> None:
        self.assertEqual(self.splits, generate_realistic_splits(20260904))
        self.assertNotEqual(self.splits, generate_realistic_splits(20260905))
        ids = [row["example_id"] for rows in self.splits.values() for row in rows]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(item.startswith("m4-") for item in ids))

    def test_taxonomy_and_ambiguity_policy(self) -> None:
        validate_taxonomy()
        self.assertEqual(len(INTENT_SPEC), 10)
        self.assertIn("R2_DUPLICATE_OVER_FRAUD", AMBIGUITY_RULES)
        self.assertIn("R1_HUMAN_ESCALATION", AMBIGUITY_RULES)

    def test_counts_coverage_and_metadata(self) -> None:
        report = validate_realistic_splits(self.splits)
        self.assertEqual(report["split_sizes"], {
            "train": 2000, "validation": 250, "id_test": 250,
            "hard_test": 250, "ood_test": 250,
        })
        for rows in self.splits.values():
            self.assertEqual({row["expected_response"]["intent"] for row in rows}, set(INTENT_SPEC))
            for row in rows:
                validate_realistic_record(row, row["split"])
                self.assertEqual(set(row["messages"][1].keys()), {"role", "content"})

    def test_hard_negatives_multi_issue_and_ood_family_exclusion(self) -> None:
        hard = self.splits["hard_test"]
        self.assertGreaterEqual(sum("confusable_intent" in row["phenomena"] for row in hard), 150)
        self.assertGreaterEqual(sum("multi_issue" in row["phenomena"] for row in hard), 20)
        families = {
            row["template_family"]
            for rows in self.splits.values()
            if rows is not self.splits["ood_test"]
            for row in rows
        }
        self.assertTrue(all(row["template_family"] not in families for row in self.splits["ood_test"]))

    def test_duplicate_and_template_detection(self) -> None:
        first = copy.deepcopy(self.splits["train"][0])
        second = copy.deepcopy(first)
        first["messages"][0]["content"] = "I was charged twice for order 123"
        second["messages"][0]["content"] = "I was charged twice for order 891"
        second["example_id"] = "other"
        self.assertEqual(len(duplicate_groups([first, second])), 1)
        self.assertEqual(normalize_text("I was charged twice for order 123"), normalize_text("I was charged twice for order 891"))
        leaks = cross_split_leakage({"train": [first], "ood_test": [second]})
        self.assertTrue(leaks["cross_split_normalized_duplicates"])

    def test_statistics_without_tokenizer(self) -> None:
        report = dataset_statistics(self.splits)
        self.assertEqual(report["global"]["total_examples"], 3000)
        self.assertEqual(report["splits"]["hard_test"]["examples"], 250)
        self.assertIn("confusable_intent", report["splits"]["hard_test"]["phenomenon_counts"])

    def test_metadata_and_schema_fail_loudly(self) -> None:
        malformed = copy.deepcopy(self.splits["train"][0])
        malformed["messages"][1]["content"] = '{"intent":"refund","extra":1}'
        with self.assertRaises(ValueError):
            validate_realistic_record(malformed)
        malformed = copy.deepcopy(self.splits["train"][0])
        malformed["label_rule"] = "unsupported"
        with self.assertRaises(ValueError):
            validate_realistic_record(malformed)


class M4MetricTests(unittest.TestCase):
    def test_invalid_json_is_not_repaired_and_schema_is_exact(self) -> None:
        self.assertEqual(parse_structured_intent('{"intent":"refund"}'), "refund")
        self.assertIsNone(parse_structured_intent('{"intent":"refund","extra":"x"}'))
        self.assertIsNone(parse_structured_intent('prefix {"intent":"refund"}'))
        self.assertIsNone(parse_structured_intent('{"intent":"not_an_intent"}'))

    def test_macro_f1_and_confusion_matrix(self) -> None:
        metrics = classification_metrics(
            ["refund", "refund", "fraud_suspected", "payment_failed"],
            ["refund", "fraud_suspected", "fraud_suspected", None],
            labels=("refund", "fraud_suspected", "payment_failed"),
        )
        self.assertAlmostEqual(metrics["intent_accuracy"], 0.5)
        self.assertEqual(metrics["valid_json_rate"], 0.75)
        self.assertEqual(metrics["confusion_matrix"]["refund"]["fraud_suspected"], 1)
        self.assertGreaterEqual(metrics["macro_f1"], 0.0)
        self.assertLessEqual(metrics["macro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
