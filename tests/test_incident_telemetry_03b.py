from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from causetune.incident_telemetry import (
    ARCHETYPE_BY_ID,
    ScenarioGenerator,
    ScenarioRequest,
    SplitIntegrityError,
    audit_dataset,
    build_reference_corpus,
    build_split_manifest,
    canonical_json,
    scenario_fingerprint,
    split_manifest_fingerprint,
    validate_counterfactual_pair,
    validate_generated_scenario,
    write_reference_corpus,
)
from causetune.incident_telemetry.generator import validate_state_invariants
from causetune.incident_telemetry.reference import REFERENCE_SEED


class ScenarioGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = ScenarioGenerator()

    def test_same_request_is_byte_and_fingerprint_stable(self) -> None:
        request = ScenarioRequest("db_query_regression", "CHECKOUT", "java_spring", "STANDARD", 17)
        first = self.generator.generate(request)
        second = self.generator.generate(request)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(canonical_json(first.to_dict()), canonical_json(second.to_dict()))
        self.assertEqual(scenario_fingerprint(first), scenario_fingerprint(second))
        self.assertEqual(first.provenance.canonical_scenario_hash, scenario_fingerprint(first))
        mutated = replace(first, expected_runbook_id="runbook.mutated")
        self.assertNotEqual(scenario_fingerprint(first), scenario_fingerprint(mutated))

    def test_explicit_seed_is_independent_of_generation_order(self) -> None:
        requests = (
            ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "STANDARD", 11),
            ScenarioRequest("cpu_throttling", "WEB_DB", "python_fastapi", "HARD", 12),
        )
        first_order = {request.archetype_id: self.generator.generate(request) for request in requests}
        second_order = {request.archetype_id: self.generator.generate(request) for request in reversed(requests)}
        self.assertEqual(first_order, second_order)
        self.assertNotEqual(first_order[requests[0].archetype_id].to_dict(), first_order[requests[1].archetype_id].to_dict())

    def test_compatibility_and_state_invariants_are_enforced(self) -> None:
        scenario = self.generator.generate(
            ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "STANDARD", 23)
        )
        validate_generated_scenario(scenario)
        state = {item.name: item.value for item in scenario.causal_state}
        state["working_set"] = state["memory_limit"] - 1
        with self.assertRaises(ValueError):
            validate_state_invariants(ARCHETYPE_BY_ID["container_memory_exhaustion"], state)
        with self.assertRaises(ValueError):
            self.generator.generate(
                ScenarioRequest("kafka_consumer_lag", "WEB_DB", "python_fastapi", "STANDARD", 23)
            )

    def test_difficulty_changes_evidence_availability_and_distractors(self) -> None:
        standard = self.generator.generate(
            ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "STANDARD", 31)
        )
        hard = self.generator.generate(
            ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "HARD", 31)
        )
        incomplete = self.generator.generate(
            ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "INCOMPLETE", 31)
        )
        self.assertEqual(standard.answerability, "ANSWERABLE")
        self.assertEqual(len(standard.distractors), 0)
        self.assertEqual(len(hard.distractors), 2)
        self.assertEqual(incomplete.answerability, "INSUFFICIENT_EVIDENCE")
        self.assertTrue(any(item.causal_role == "MISSING_REQUIRED" and item.may_be_removed for item in incomplete.causal_evidence))
        self.assertTrue(
            {item.evidence_id for item in hard.distractors}.isdisjoint(
                {item.evidence_id for item in hard.causal_evidence}
            )
        )

    def test_counterfactual_pair_metadata_is_truthful(self) -> None:
        first, second, pair = self.generator.generate_counterfactual_pair(
            "container_memory_exhaustion",
            "cpu_throttling",
            topology_family="CHECKOUT",
            runtime_family="python_fastapi",
            pair_seed=41,
            split_group_id="pair-group",
        )
        validate_counterfactual_pair(first, second, pair)
        self.assertEqual(first.topology, second.topology)
        self.assertEqual(first.runtime, second.runtime)
        self.assertNotEqual(first.root_cause_id, second.root_cause_id)
        malformed = replace(pair, variables_held_constant=("working_set",))
        with self.assertRaises(ValueError):
            validate_counterfactual_pair(first, second, malformed)


class ReferenceCorpusTests(unittest.TestCase):
    def test_reference_corpus_is_small_auditable_and_grouped(self) -> None:
        scenarios, manifest, pairs = build_reference_corpus(REFERENCE_SEED)
        report = audit_dataset(
            scenarios,
            split_manifest=manifest,
            archetypes=ARCHETYPE_BY_ID,
            counterfactual_pairs=pairs,
        )
        self.assertEqual(report["status"], "pass")
        self.assertEqual(len(scenarios), 52)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(manifest.manifest_fingerprint, split_manifest_fingerprint(manifest))
        self.assertEqual(report["split"]["split_counts"]["SEALED_HOLDOUT"], 4)
        self.assertEqual(report["split"]["split_counts"]["ABSTENTION_TEST"], 4)

    def test_reference_artifacts_are_byte_identical_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            write_reference_corpus(first_dir, REFERENCE_SEED)
            write_reference_corpus(second_dir, REFERENCE_SEED)
            first_files = sorted(Path(first_dir).glob("*"))
            second_files = sorted(Path(second_dir).glob("*"))
            self.assertEqual([item.name for item in first_files], [item.name for item in second_files])
            for first, second in zip(first_files, second_files):
                self.assertEqual(first.read_bytes(), second.read_bytes(), first.name)

    def test_audit_rejects_corrupt_authoritative_scenario(self) -> None:
        scenarios, manifest, pairs = build_reference_corpus(REFERENCE_SEED)
        corrupted = scenarios[0]
        object.__setattr__(corrupted, "root_cause_id", "cpu_throttling")
        with self.assertRaises(ValueError):
            audit_dataset(
                scenarios,
                split_manifest=manifest,
                archetypes=ARCHETYPE_BY_ID,
                counterfactual_pairs=pairs,
            )

    def test_audit_rejects_contradictory_distractor_fixture(self) -> None:
        scenarios, manifest, pairs = build_reference_corpus(REFERENCE_SEED)
        hard = next(item for item in scenarios if item.difficulty == "HARD")
        replacement = replace(hard.distractors[0], approved_noise=False)
        corrupted = replace(hard, distractors=(replacement, *hard.distractors[1:]))
        corrupted_scenarios = [corrupted if item.scenario_id == hard.scenario_id else item for item in scenarios]
        with self.assertRaises(ValueError):
            audit_dataset(
                corrupted_scenarios,
                split_manifest=manifest,
                archetypes=ARCHETYPE_BY_ID,
                counterfactual_pairs=pairs,
            )

    def test_group_leakage_and_unknown_split_fail_closed(self) -> None:
        scenarios, _manifest, _pairs = build_reference_corpus(REFERENCE_SEED)
        first = replace(scenarios[0], split_group_id="shared-group")
        second = replace(scenarios[1], split_group_id="shared-group")
        with self.assertRaises(SplitIntegrityError):
            build_split_manifest(
                (first, second),
                {first.scenario_id: "TRAIN", second.scenario_id: "VALIDATION"},
            )
        with self.assertRaises(ValueError):
            ScenarioGenerator().generate(
                ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "UNKNOWN", 1)
            )
