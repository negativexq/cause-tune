from __future__ import annotations

import copy
import json
import unittest

from causetune.incident_benchmark import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    FAILURE_FAMILIES,
    SLICES,
    benchmark_fingerprint,
    generate_benchmark,
    normalize_incident_text,
    validate_benchmark,
)
from causetune.incident_evaluation import evaluate_incidents, failure_patterns_for_splits, parse_incident_diagnosis
from causetune.incident_contract import contract_fingerprint
from causetune.incident_taxonomy import ACTIONS


class IncidentBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs, cls.truth = generate_benchmark(20260904)

    def test_deterministic_generation_and_fingerprint(self) -> None:
        inputs2, truth2 = generate_benchmark(20260904)
        self.assertEqual(self.inputs, inputs2)
        self.assertEqual(self.truth, truth2)
        self.assertNotEqual(self.truth, generate_benchmark(20260905)[1])
        self.assertEqual(benchmark_fingerprint(self.inputs, self.truth), benchmark_fingerprint(inputs2, truth2))

    def test_counts_and_family_balance(self) -> None:
        report = validate_benchmark(self.inputs, self.truth)
        self.assertEqual(report["benchmark_version"], BENCHMARK_VERSION)
        self.assertEqual(report["generator_version"], GENERATOR_VERSION)
        self.assertEqual(report["split_counts"], {"standard": 72, "hard": 48, "transfer": 24})
        self.assertEqual(report["failure_family_counts"], {family: 12 for family in sorted(FAILURE_FAMILIES)})
        self.assertEqual(report["hard_red_herring_count"], 48)
        self.assertEqual(report["literal_label_leakage_count"], 0)

    def test_no_duplicates_and_transfer_topology_separation(self) -> None:
        ids = [row["incident_id"] for split in SLICES for row in self.inputs[split]]
        self.assertEqual(len(ids), len(set(ids)))
        normalized = [normalize_incident_text(row["incident_packet"]) for split in SLICES for row in self.inputs[split]]
        self.assertEqual(len(normalized), len(set(normalized)))
        core = {row["metadata"]["topology_family"] for split in ("standard", "hard") for row in self.inputs[split]}
        transfer = {row["metadata"]["topology_family"] for row in self.inputs["transfer"]}
        self.assertTrue(core.isdisjoint(transfer))

    def test_validation_catches_leakage_and_invalid_references(self) -> None:
        bad_inputs = copy.deepcopy(self.inputs)
        bad_inputs["hard"][0]["incident_packet"] += "\nNote: db_connection_pool_exhaustion"
        with self.assertRaises(ValueError):
            validate_benchmark(bad_inputs, self.truth)
        bad_inputs = copy.deepcopy(self.inputs)
        bad_inputs["standard"][0]["metadata"]["evidence_ids"] = ["M99", "E1"]
        with self.assertRaises(ValueError):
            validate_benchmark(bad_inputs, self.truth)
        bad_inputs = copy.deepcopy(self.inputs)
        bad_inputs["hard"][0]["incident_packet"] = bad_inputs["standard"][0]["incident_packet"]
        with self.assertRaises(ValueError):
            validate_benchmark(bad_inputs, self.truth)

    def test_validation_catches_unknown_ground_truth_component(self) -> None:
        bad_truth = copy.deepcopy(self.truth)
        bad_truth[0]["culprit_service"] = "component-not-in-packet"
        with self.assertRaises(ValueError):
            validate_benchmark(self.inputs, bad_truth)


class IncidentEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        inputs, truths = generate_benchmark(20260904)
        cls.inputs = inputs["standard"][:2]
        truth_map = {row["incident_id"]: row for row in truths}
        cls.truth = {row["incident_id"]: truth_map[row["incident_id"]] for row in cls.inputs}

    def test_strict_parser_rejects_extra_keys_and_bad_enums(self) -> None:
        components = self.inputs[0]["metadata"]["present_components"]
        evidence = self.inputs[0]["metadata"]["evidence_ids"]
        valid = json.dumps({
            "culprit_service": components[0],
            "failure_mode": "memory_leak",
            "recommended_action": ACTIONS[1],
            "evidence_ids": evidence,
        })
        parsed, category, json_valid = parse_incident_diagnosis(valid, components, evidence)
        self.assertIsNotNone(parsed)
        self.assertEqual(category, "valid JSON")
        self.assertTrue(json_valid)
        self.assertIsNone(parse_incident_diagnosis(valid[:-1] + ', "extra": 1}', components, evidence)[0])
        self.assertIsNone(parse_incident_diagnosis("plain prose", components, evidence)[0])
        invalid_mode = valid.replace("memory_leak", "not_a_failure_family")
        self.assertEqual(parse_incident_diagnosis(invalid_mode, components, evidence)[1], "unknown failure mode")

    def test_packet_evidence_is_not_limited_to_canonical_ground_truth_evidence(self) -> None:
        incident = self.inputs[0]
        truth = self.truth[incident["incident_id"]]
        output = json.dumps({
            "culprit_service": truth["culprit_service"],
            "failure_mode": truth["failure_mode"],
            "recommended_action": truth["recommended_action"],
            "evidence_ids": ["M2", "E2"],
        })
        metrics = evaluate_incidents(
            [incident], self.truth, {incident["incident_id"]: output}
        )
        self.assertEqual(metrics["json_compliance"]["count"], 1)

    def test_metrics_require_culprit_and_failure_mode_for_primary_match(self) -> None:
        outputs = {}
        for incident in self.inputs:
            truth = self.truth[incident["incident_id"]]
            outputs[incident["incident_id"]] = json.dumps({
                "culprit_service": truth["culprit_service"],
                "failure_mode": truth["failure_mode"],
                "recommended_action": truth["recommended_action"],
                "evidence_ids": truth["evidence_ids"],
            })
        metrics = evaluate_incidents(self.inputs, self.truth, outputs)
        self.assertEqual(metrics["diagnosis_exact_match"]["count"], 2)
        self.assertEqual(metrics["resolution_exact_match"]["count"], 2)
        self.assertEqual(metrics["evidence"]["f1"], 1.0)
        self.assertEqual(metrics["failure_mode_macro_f1"], 1.0 / 12.0)

    def test_failure_patterns_support_aggregate_all_without_split_lookup(self) -> None:
        incident = self.inputs[0]
        truth = self.truth[incident["incident_id"]]
        output = json.dumps({
            "culprit_service": truth["culprit_service"],
            "failure_mode": truth["failure_mode"],
            "recommended_action": truth["recommended_action"],
            "evidence_ids": truth["evidence_ids"],
        })
        evaluation = evaluate_incidents([incident], self.truth, {incident["incident_id"]: output})
        patterns = failure_patterns_for_splits(
            {"standard": [incident]}, {"all": evaluation}, [incident]
        )
        self.assertEqual(patterns["all"]["total_failures"], 0)

    def test_contract_is_explicit_and_not_the_experiment_01_contract(self) -> None:
        with open("configs/incident_diagnosis_eval.json", encoding="utf-8") as handle:
            config = json.load(handle)
        self.assertEqual(config["contract_fingerprint"], contract_fingerprint(config))
        self.assertNotEqual(
            config["contract_fingerprint"],
            "1b00a333c26c4cbd03b3e04d990fad3b4adf0d9a03443c9fd5f183ee7e8ef94d",
        )
