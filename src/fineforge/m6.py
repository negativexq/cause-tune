"""Pure integrity helpers for the single deterministic-shuffle M6 run."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

from .m5 import sha256_file


EXPECTED_M4_DATASET_FINGERPRINT = "0efaacae27cbfeb1c304c8ee359384239d4526470a32aded8f0eda39908e9d06"
EXPECTED_M6_ORDER = {
    "shuffle": True,
    "sampler": "deterministic_python_random_v1",
}


def _fixed_config(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    value.pop("experiment", None)
    value.pop("output_dir", None)
    value.pop("training_order", None)
    return value


def validate_m6_config(
    m5_config: Mapping[str, Any],
    m6_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Require that M6 differs from M5 only by order and output identity."""

    if _fixed_config(m5_config) != _fixed_config(m6_config):
        raise ValueError("M6 changes a fixed M5 dataset/model/training/evaluation field")
    order = m6_config.get("training_order")
    expected = {**EXPECTED_M6_ORDER, "seed": m5_config["seed"]}
    if order != expected:
        raise ValueError(f"M6 training_order must be exactly {expected}")
    if m6_config.get("experiment") == m5_config.get("experiment"):
        raise ValueError("M6 experiment identity must differ from M5")
    if m6_config.get("output_dir") == m5_config.get("output_dir"):
        raise ValueError("M6 output directory must differ from M5")
    return {
        "m5": {
            "experiment": m5_config.get("experiment"),
            "output_dir": m5_config.get("output_dir"),
            "training_order": {
                "shuffle": False,
                "sampler": "implicit_sequential_sampler",
                "seed": m5_config["seed"],
            },
        },
        "m6": {
            "experiment": m6_config.get("experiment"),
            "output_dir": m6_config.get("output_dir"),
            "training_order": order,
        },
        "changed_fields": [
            "experiment",
            "output_dir",
            "training_order.shuffle",
            "training_order.sampler",
        ],
        "fixed_fields_unchanged": True,
    }


def m6_artifact_metadata(output_dir: str | Path) -> dict[str, Any]:
    """Validate the M6 artifact set without requiring M5-only filenames."""

    directory = Path(output_dir)
    required = (
        "adapter_model.safetensors",
        "adapter_config.json",
        "tokenizer_config.json",
        "resolved_training_config.json",
        "training_metrics.json",
        "loss_history.json",
        "reload_verification.json",
        "post_training_metrics.json",
        "base_vs_m5_vs_m6.json",
        "experiment_diff.json",
        "shuffled_order.json",
        "order_audit.json",
        "optimizer_steps.jsonl",
        "validation_progression.json",
    )
    missing = [name for name in required if not (directory / name).exists()]
    if missing:
        raise ValueError(f"missing M6 artifacts: {missing}")
    return {
        "required_files": list(required),
        "missing_files": missing,
        "adapter_model_sha256": sha256_file(directory / "adapter_model.safetensors"),
    }
