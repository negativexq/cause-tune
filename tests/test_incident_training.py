from __future__ import annotations

import json
import unittest
from pathlib import Path

from causetune.incident_benchmark import read_benchmark
from causetune.incident_contract import contract_fingerprint
from causetune.incident_training import (
    CheckpointSelectionPolicy,
    TRAIN_SEED,
    VALIDATION_SEED,
    contamination_report,
    checkpoint_path,
    earliest_within_tolerance,
    generate_training_split,
    training_fingerprint,
    validate_training_manifest,
    validate_training_split,
    validation_schedule,
)
from causetune.data.preprocess import IGNORE_INDEX, build_preprocessed_messages


class TinyChatTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking=False):
        rendered = "".join(f"<{item['role']}>{item['content']}" for item in messages)
        rendered += "<assistant>" if add_generation_prompt else "<eos>"
        return [ord(char) for char in rendered] if tokenize else rendered

    def decode(self, token_ids, skip_special_tokens=False):
        return "".join(chr(token_id) for token_id in token_ids)


class IncidentTrainingTests(unittest.TestCase):
    def test_exact_balanced_split_sizes_and_difficulty(self) -> None:
        train, train_truth = generate_training_split("train", 200, TRAIN_SEED, "train")
        validation, validation_truth = generate_training_split("validation", 24, VALIDATION_SEED, "validation")
        train_report = validate_training_split(train, train_truth, expected_count=2400, expected_per_family=200)
        validation_report = validate_training_split(validation, validation_truth, expected_count=288, expected_per_family=24)
        self.assertEqual(len(train), 2400)
        self.assertEqual(len(validation), 288)
        self.assertEqual(set(train_report["family_counts"].values()), {200})
        self.assertEqual(set(validation_report["family_counts"].values()), {24})
        self.assertEqual(train_report["difficulty_counts"], {"hard": 720, "standard": 1440, "transfer": 240})
        self.assertEqual(validation_report["difficulty_counts"], {"hard": 96, "standard": 168, "transfer": 24})

    def test_generation_and_fingerprints_are_deterministic(self) -> None:
        first = generate_training_split("train", 2, TRAIN_SEED, "train")
        second = generate_training_split("train", 2, TRAIN_SEED, "train")
        self.assertEqual(first, second)
        self.assertEqual(training_fingerprint(*first), training_fingerprint(*second))
        different = generate_training_split("train", 2, TRAIN_SEED + 1, "train")
        self.assertNotEqual(first, different)

    def test_contamination_against_frozen_02a_is_zero(self) -> None:
        train, _ = generate_training_split("train", 200, TRAIN_SEED, "train")
        validation, _ = generate_training_split("validation", 24, VALIDATION_SEED, "validation")
        report = contamination_report(train, validation, Path("data/incident_diagnosis"))
        self.assertEqual(report["status"], "pass")
        for pair in report["pairs"].values():
            self.assertEqual(pair, {"exact_text_overlap": 0, "normalized_text_overlap": 0, "canonical_structure_overlap": 0, "incident_id_overlap": 0})

    def test_checkpoint_selection_is_validation_only_and_deterministic(self) -> None:
        policy = CheckpointSelectionPolicy()
        self.assertFalse(policy.observe(0, {"diagnosis_exact_match": .20, "resolution_exact_match": .1, "failure_mode_macro_f1": .2, "teacher_forced_loss": 2.0}))
        self.assertFalse(policy.observe(25, {"diagnosis_exact_match": .20, "resolution_exact_match": .2, "failure_mode_macro_f1": .3, "teacher_forced_loss": 1.0}))
        self.assertFalse(policy.observe(50, {"diagnosis_exact_match": .20, "resolution_exact_match": .2, "failure_mode_macro_f1": .3, "teacher_forced_loss": 1.0}))
        self.assertFalse(policy.observe(75, {"diagnosis_exact_match": .20, "resolution_exact_match": .2, "failure_mode_macro_f1": .3, "teacher_forced_loss": 1.0}))
        self.assertTrue(policy.observe(100, {"diagnosis_exact_match": .20, "resolution_exact_match": .2, "failure_mode_macro_f1": .3, "teacher_forced_loss": 1.0}))
        self.assertEqual(policy.state(), {"best_step": 25, "best_metric": .2, "last_improvement_step": 0, "patience_counter": 3, "stop_reason": "validation_no_improvement"})

    def test_earliest_within_tolerance(self) -> None:
        history = [
            {"step": 0, "diagnosis_exact_match": .50},
            {"step": 25, "diagnosis_exact_match": .70},
            {"step": 50, "diagnosis_exact_match": .76},
            {"step": 75, "diagnosis_exact_match": .755},
        ]
        self.assertEqual(earliest_within_tolerance(history), {"best_diagnosis_exact_match": .76, "earliest_step": 50, "tolerance": .01, "updates_potentially_avoided": 25})

    def test_validation_schedule_and_adapter_checkpoint_naming(self) -> None:
        self.assertEqual(validation_schedule(100), (0, 25, 50, 75, 100))
        self.assertEqual(checkpoint_path("outputs/incident_diagnosis_02b", 25).as_posix(), "outputs/incident_diagnosis_02b/checkpoint-step-000025")

    def test_manifest_fingerprint_invariants(self) -> None:
        manifest = json.loads(Path("data/incident_diagnosis_training/resolved_training_manifest.json").read_text())
        evaluation = json.loads(Path("configs/incident_diagnosis_eval.json").read_text())
        validate_training_manifest(manifest, expected_benchmark_fingerprint="5e9a6d74ba881af7aa1146a23f4b837e139a4e8e5a77dba7fe40c3bb2c9c0a94", expected_evaluation_fingerprint=contract_fingerprint(evaluation))
        self.assertNotIn("data/incident_diagnosis/", manifest["training_sources"]["train"])

    def test_incident_formatting_masks_only_assistant_target(self) -> None:
        train, truths = generate_training_split("train", 1, TRAIN_SEED, "train")
        example_id = train[0]["incident_id"]
        truth = next(item for item in truths if item["incident_id"] == example_id)
        target = json.dumps({"culprit_service": truth["culprit_service"], "failure_mode": truth["failure_mode"], "recommended_action": truth["recommended_action"], "evidence_ids": truth["evidence_ids"]}, separators=(",", ":"))
        example = build_preprocessed_messages(example_id, [{"role": "user", "content": train[0]["incident_packet"]}, {"role": "assistant", "content": target}], TinyChatTokenizer(), 4096, "fixed system instruction")
        self.assertTrue(all(label == IGNORE_INDEX for label in example.labels[:example.prompt_token_count]))
        self.assertGreater(example.trainable_assistant_token_count, 0)
        self.assertEqual(TinyChatTokenizer().decode([token for token, label in zip(example.input_ids, example.labels) if label != IGNORE_INDEX]), target + "<eos>")


if __name__ == "__main__":
    unittest.main()
