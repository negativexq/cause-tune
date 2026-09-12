from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from causetune.experiment_contract import (
    ExperimentContractError,
    field_classification,
    legacy_config_to_contract,
    load_experiment_contract,
    metadata_only_fields,
    resolve_experiment_config,
    training_affecting_fields,
)


def contract_payload() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "incident-02b",
        "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-abc123"},
        "data": {
            "train": {"path": "data/train.jsonl", "fingerprint": "train-sha"},
            "validation": {"path": "data/validation.jsonl", "fingerprint": "validation-sha"},
            "benchmark": {"path": "data/benchmark.jsonl", "fingerprint": "benchmark-sha"},
        },
        "training": {
            "seed": 42,
            "quantization": {
                "load_in_4bit": True,
                "quant_type": "nf4",
                "compute_dtype": "bfloat16",
                "double_quant": True,
            },
            "lora": {
                "rank": 16,
                "alpha": 32,
                "dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
            },
            "optimizer": {
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 8,
                "learning_rate": 2e-4,
                "max_epochs": 2,
                "max_sequence_length": 1024,
                "gradient_checkpointing": True,
                "gradient_checkpointing_use_reentrant": False,
            },
            "checkpoint_policy": {
                "selection_split": "validation",
                "validation_only": True,
                "primary_metric": "diagnosis_exact_match",
                "interval_steps": 25,
            },
            "stopping_policy": {
                "mode": "early_stopping",
                "max_steps": 600,
                "patience": 3,
                "min_delta": 0.005,
                "eval_interval_steps": 25,
            },
            "preprocessing": {
                "deterministic": True,
                "shuffle": True,
                "seed": 42,
                "version": "incident-v1",
            },
        },
        "evaluation": {"contract_version": "incident-v1", "scorer_version": "scorer-v1"},
        "output": {"output_dir": "runs/incident-02b"},
    }


class ExperimentContractTests(unittest.TestCase):
    def test_defaults_and_stable_serialization(self) -> None:
        first = resolve_experiment_config(contract_payload())
        reordered = copy.deepcopy(contract_payload())
        reordered["training"] = {
            key: reordered["training"][key]
            for key in reversed(tuple(reordered["training"]))
        }
        second = resolve_experiment_config(reordered)

        self.assertEqual(first.canonical_json(), second.canonical_json())
        self.assertEqual(first.sha256(), second.sha256())
        self.assertEqual(json.loads(first.resolved_json()), first.to_dict())
        self.assertTrue(first.resolved_json().endswith("\n"))

    def test_resolved_file_is_deterministic(self) -> None:
        contract = resolve_experiment_config(contract_payload())
        destination = Path(".pytest_cache/m8-resolved-config.json")
        contract.write_resolved(destination)
        self.assertEqual(destination.read_text(encoding="utf-8"), contract.resolved_json())

    def test_unknown_keys_fail_at_every_contract_level(self) -> None:
        for location in ("root", "model", "data", "training", "lora", "optimizer"):
            raw = copy.deepcopy(contract_payload())
            if location == "root":
                raw["ignored_training_option"] = True
            elif location == "model":
                raw["model"]["alias"] = "wrong"
            elif location == "data":
                raw["data"]["train"] = {"path": "data/train.jsonl", "role": "train"}
            elif location == "training":
                raw["training"]["ignored"] = True
            elif location == "lora":
                raw["training"]["lora"]["bias"] = "none"
            else:
                raw["training"]["optimizer"]["weight_decay"] = 0.0
            with self.subTest(location=location):
                with self.assertRaisesRegex(ExperimentContractError, "unknown config key"):
                    resolve_experiment_config(raw)

    def test_invalid_combinations_and_isolation_fail_closed(self) -> None:
        cases = []
        raw = copy.deepcopy(contract_payload())
        raw["training"]["quantization"]["load_in_4bit"] = False
        cases.append(raw)
        raw = copy.deepcopy(contract_payload())
        raw["training"]["checkpoint_policy"]["selection_split"] = "benchmark"
        cases.append(raw)
        raw = copy.deepcopy(contract_payload())
        raw["training"]["stopping_policy"] = {"mode": "early_stopping", "max_steps": 100}
        cases.append(raw)
        raw = copy.deepcopy(contract_payload())
        raw["data"]["benchmark"]["path"] = raw["data"]["train"]["path"]
        cases.append(raw)
        for invalid in cases:
            with self.assertRaises(ExperimentContractError):
                resolve_experiment_config(invalid)

    def test_training_fields_are_exhaustively_classified(self) -> None:
        classification = field_classification()
        self.assertTrue(classification)
        self.assertEqual(
            set(classification.values()), {"training-affecting", "metadata-only"}
        )
        self.assertTrue(training_affecting_fields())
        self.assertTrue(metadata_only_fields())
        self.assertIn("training.seed", training_affecting_fields())
        self.assertIn("data.benchmark", metadata_only_fields())
        self.assertNotIn("data.benchmark", training_affecting_fields())

    def test_roles_are_immutable_after_resolution(self) -> None:
        contract = resolve_experiment_config(contract_payload())
        with self.assertRaises(TypeError):
            contract.data["train"] = contract.data["validation"]  # type: ignore[index]

    def test_existing_e01_e02_config_shapes_remain_representable(self) -> None:
        for path in (
            Path("configs/sft_smoke.json"),
            Path("configs/realistic_qlora_v1.json"),
            Path("configs/realistic_qlora_m6.json"),
            Path("configs/incident_diagnosis_training.json"),
        ):
            raw = json.loads(path.read_text(encoding="utf-8"))
            adapted = legacy_config_to_contract(raw, experiment_id=path.stem)
            with self.subTest(path=path):
                contract = resolve_experiment_config(adapted)
                self.assertEqual(contract.model.revision_policy, "legacy_unpinned")

    def test_load_experiment_contract_uses_resolved_defaults(self) -> None:
        contract = load_experiment_contract(Path("configs/m8_contract_fixture.json"))
        self.assertEqual(contract.experiment_id, "m8-fixture")
        self.assertEqual(contract.training.preprocessing.seed, 42)
        self.assertEqual(contract.output.output_dir, "runs/m8-fixture")


if __name__ == "__main__":
    unittest.main()
