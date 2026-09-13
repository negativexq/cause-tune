from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def load_config() -> dict:
    return json.loads((ROOT / "configs/experiment_06/e06-training.json").read_text(encoding="utf-8"))


def test_e06_training_protocol_is_frozen_and_uses_the_selected_e04_recipe() -> None:
    config = load_config()
    assert config["model"] == {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "revision": "cfbefacb99257ffa30c83adab238a50856ac3083",
        "revision_policy": "pinned",
    }
    assert config["data"]["train"]["fingerprint"] == "156c9862ebab8a2e3ccef69cc3879c1dc6cb88666d439320216b69bd56f629ed"
    assert config["data"]["validation"]["fingerprint"] == "375ae9ae09de3397f84789a8628a7d368e09f3b6e23f11b5d0b074d8e39efdd7"
    assert config["training"]["lora"] == {
        "rank": 8,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
    }
    assert config["training"]["optimizer"]["learning_rate"] == 0.0001
    assert config["training"]["checkpoint_policy"]["validation_only"] is True
    assert config["metadata"]["tags"] == "e06-cross-model-replication"


def test_e06_training_preflight_records_native_loading_and_capability_gap() -> None:
    record = json.loads((ROOT / "results/experiment_06/training_preflight.json").read_text(encoding="utf-8"))
    assert record["status"] == "PASS"
    assert record["protocol_status"] == "FROZEN_BEFORE_TRAINING"
    assert record["model"]["trust_remote_code"] is False
    assert record["capability_gap_screen"]["decision"] == "CAPABILITY_GAP_PRESENT"
    assert record["benchmark_used_for_training_decisions"] is False
