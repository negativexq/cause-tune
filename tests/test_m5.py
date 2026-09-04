from __future__ import annotations

import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from fineforge.m5 import (
    EXPECTED_COUNTS,
    EXPECTED_EVALUATION_FINGERPRINT,
    adapter_artifact_metadata,
    confusion_delta,
    dataset_fingerprint,
    metric_delta,
    parameter_structure,
    prediction_transitions,
    validate_m5_config,
)
from fineforge.m5_audit import audit_training_order, deterministic_shuffled_indices
from fineforge.m6 import validate_m6_config


class _MockLoraTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(1), requires_grad=True)
        self.lora_B = nn.Parameter(torch.ones(1), requires_grad=True)


class _MockPackedPEFTModel(nn.Module):
    """Small stand-in for a 4-bit PEFT model; no Qwen weights are loaded."""

    def __init__(self, target_modules: tuple[str, ...]) -> None:
        super().__init__()
        self.packed_base = nn.Parameter(torch.zeros(7), requires_grad=False)
        for target in target_modules:
            self.add_module(target, _MockLoraTarget())

    def get_nb_trainable_parameters(self) -> tuple[int, int]:
        return 33_030_144, 4_055_498_240


class M5IntegrityTests(unittest.TestCase):
    def test_m6_config_changes_only_order_and_identity(self) -> None:
        m5 = json.loads(Path("configs/realistic_qlora_v1.json").read_text(encoding="utf-8"))
        m6 = json.loads(Path("configs/realistic_qlora_m6.json").read_text(encoding="utf-8"))
        diff = validate_m6_config(m5, m6)
        self.assertTrue(diff["fixed_fields_unchanged"])
        self.assertEqual(
            diff["changed_fields"],
            ["experiment", "output_dir", "training_order.shuffle", "training_order.sampler"],
        )

    def test_m6_order_has_balanced_mixed_accumulation_windows(self) -> None:
        records = [
            {"example_id": f"{label}-{index:03d}", "expected_response": {"intent": label}}
            for label in ("a", "b", "c", "d")
            for index in range(50)
        ]
        order = deterministic_shuffled_indices(len(records), 42)
        audit = audit_training_order(
            [records[index] for index in order], gradient_accumulation_steps=8
        )
        self.assertEqual(audit["overall_label_counts"], {"a": 50, "b": 50, "c": 50, "d": 50})
        self.assertGreater(audit["mixed_class_window_count"], 0)
        self.assertEqual(audit["single_class_window_count"], 0)

    def test_future_sampler_is_deterministic_and_changes_order(self) -> None:
        first = deterministic_shuffled_indices(2000, 42)
        same = deterministic_shuffled_indices(2000, 42)
        other = deterministic_shuffled_indices(2000, 43)
        self.assertEqual(first, same)
        self.assertNotEqual(first, other)
        self.assertNotEqual(first, list(range(2000)))

    def test_future_shuffled_epoch_preserves_balance_and_mixes_blocks(self) -> None:
        labels = [label for label in ("a", "b", "c", "d") for _ in range(50)]
        order = deterministic_shuffled_indices(len(labels), 42)
        shuffled = [labels[index] for index in order]
        self.assertEqual(dict(sorted(Counter(shuffled).items())), {"a": 50, "b": 50, "c": 50, "d": 50})
        self.assertTrue(all(len(set(shuffled[start:start + 25])) > 1 for start in range(0, 200, 25)))

    def test_training_order_audit_detects_terminal_class_block(self) -> None:
        records = [
            {"example_id": f"x-{label}-{index:03d}", "expected_response": {"intent": label}}
            for label in ("a", "b")
            for index in range(8)
        ]
        audit = audit_training_order(records, gradient_accumulation_steps=4)
        self.assertEqual(audit["single_class_window_count"], 4)
        self.assertEqual(audit["mixed_class_window_count"], 0)
        self.assertEqual(audit["last_50_labels"][-8:], ["b"] * 8)

    def test_packed_numel_is_not_used_as_logical_peft_total(self) -> None:
        targets = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
        model = _MockPackedPEFTModel(targets)

        structure = parameter_structure(model, targets)

        self.assertEqual(structure["counting_method"], "peft.get_nb_trainable_parameters")
        self.assertEqual(structure["trainable_parameter_count"], 33_030_144)
        self.assertEqual(structure["logical_parameter_count"], 4_055_498_240)
        self.assertLess(structure["packed_parameter_numel"], structure["logical_parameter_count"])
        self.assertAlmostEqual(structure["trainable_percentage"], 0.8144534, places=5)
        self.assertTrue(all(structure["lora_target_modules_present"].values()))
        self.assertTrue(structure["base_parameters_frozen"])
        self.assertTrue(structure["lora_parameters_trainable"])

    def test_config_and_fingerprint_are_enforced(self) -> None:
        config = json.loads(Path("configs/realistic_qlora_v1.json").read_text(encoding="utf-8"))
        self.assertEqual(validate_m5_config(config), EXPECTED_EVALUATION_FINGERPRINT)
        self.assertEqual(EXPECTED_COUNTS["train"], 2000)
        altered = json.loads(json.dumps(config))
        altered["evaluation_contract_fingerprint"] = "changed"
        with self.assertRaises(ValueError):
            validate_m5_config(altered)

    def test_dataset_fingerprint_has_all_immutable_split_files(self) -> None:
        fingerprint = dataset_fingerprint("data/realistic")
        self.assertEqual(len(fingerprint["aggregate_sha256"]), 64)
        self.assertEqual(set(fingerprint["files"]), {
            "train", "validation", "id_test", "hard_test", "ood_test", "manifest.json"
        })

    def test_metric_and_confusion_deltas(self) -> None:
        base = {
            "intent_accuracy": 0.8,
            "macro_f1": 0.7,
            "valid_json_rate": 1.0,
            "exact_schema_compliance_rate": 1.0,
            "teacher_forced_loss": 2.0,
        }
        tuned = {
            "intent_accuracy": 0.85,
            "macro_f1": 0.75,
            "valid_json_rate": 1.0,
            "exact_schema_compliance_rate": 1.0,
            "teacher_forced_loss": 1.5,
        }
        delta = metric_delta(base, tuned)
        self.assertAlmostEqual(delta["accuracy_delta_pp"], 5.0)
        self.assertAlmostEqual(delta["macro_f1_delta_pp"], 5.0)
        self.assertEqual(delta["teacher_loss_delta"], -0.5)
        pair = confusion_delta(
            {"a": {"b": 3}},
            {"a": {"b": 1}},
            (("a", "b"),),
        )["a -> b"]
        self.assertEqual(pair, {"baseline": 3, "tuned": 1, "delta": -2})

    def test_prediction_transitions_expose_new_regressions(self) -> None:
        record = {
            "example_id": "x",
            "expected_response": {"intent": "refund"},
            "difficulty": "hard",
            "phenomena": [],
            "confusable_with": [],
            "scenario_family": "family",
        }
        base = [
            {"correct": False, "parsed_intent": "cancel_order", "raw_text": "base"},
            {"correct": True, "parsed_intent": "refund", "raw_text": "base"},
        ]
        tuned = [
            {"correct": True, "parsed_intent": "refund", "raw_text": "tuned"},
            {"correct": False, "parsed_intent": "cancel_order", "raw_text": "tuned"},
        ]
        result = prediction_transitions([record, record], base, tuned)
        self.assertEqual(result["counts"]["base_wrong_tuned_correct"], 1)
        self.assertEqual(result["counts"]["base_correct_tuned_wrong"], 1)

    def test_adapter_artifact_manifest_requires_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            for name in (
                "adapter_model.safetensors", "adapter_config.json",
                "tokenizer_config.json", "resolved_training_config.json",
                "training_metrics.json", "loss_history.json",
                "reload_verification.json", "post_training_metrics.json",
                "base_vs_tuned.json",
            ):
                (path / name).write_text("{}", encoding="utf-8")
            metadata = adapter_artifact_metadata(path)
            self.assertEqual(metadata["missing_files"], [])
            self.assertEqual(len(metadata["adapter_model_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
