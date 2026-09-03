"""Configuration for the FineForge SFT smoke milestone.

The configuration is deliberately dependency-free so experiment settings can be
loaded, printed, and tested without loading a model.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QuantizationConfig:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(frozen=True)
class SFTConfig:
    model_id: str
    seed: int
    dataset_path: str
    max_examples: int = 200
    train_ratio: float = 0.80
    validation_ratio: float = 0.10
    test_ratio: float = 0.10
    max_sequence_length: int = 1024
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    num_epochs: int = 1
    quantization: QuantizationConfig = QuantizationConfig()
    lora: LoraConfig = LoraConfig()
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False
    output_dir: str = "outputs/sft_smoke"

    def __post_init__(self) -> None:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_examples <= 0:
            raise ValueError("max_examples must be positive")
        if self.max_sequence_length <= 0:
            raise ValueError("max_sequence_length must be positive")
        if self.micro_batch_size <= 0:
            raise ValueError("micro_batch_size must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        ratios = (self.train_ratio, self.validation_ratio, self.test_ratio)
        if any(ratio < 0 for ratio in ratios):
            raise ValueError("split ratios must be non-negative")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ValueError("train/validation/test ratios must sum to 1.0")
        if self.quantization.quant_type.lower() != "nf4":
            raise ValueError("the SFT smoke configuration must use NF4")
        if self.quantization.compute_dtype.lower() != "bfloat16":
            raise ValueError("the SFT smoke configuration must use bfloat16 compute")
        if self.lora.rank <= 0 or self.lora.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive")
        if not 0 <= self.lora.dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        if not self.lora.target_modules:
            raise ValueError("LoRA target_modules must not be empty")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the configuration."""

        result = asdict(self)
        result["lora"]["target_modules"] = list(self.lora.target_modules)
        return result

    def human_readable(self) -> str:
        """Return stable, indented JSON suitable for experiment evidence."""

        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


def config_from_dict(raw: dict[str, Any]) -> SFTConfig:
    """Construct and validate SFTConfig from decoded JSON data."""

    if not isinstance(raw, dict):
        raise ValueError("configuration root must be a JSON object")

    quantization_raw = raw.get("quantization", {})
    lora_raw = raw.get("lora", {})
    if not isinstance(quantization_raw, dict):
        raise ValueError("quantization must be a JSON object")
    if not isinstance(lora_raw, dict):
        raise ValueError("lora must be a JSON object")

    quantization = QuantizationConfig(**quantization_raw)
    if "target_modules" in lora_raw:
        lora_raw = {
            **lora_raw,
            "target_modules": tuple(lora_raw["target_modules"]),
        }
    lora = LoraConfig(**lora_raw)

    top_level = {
        key: value
        for key, value in raw.items()
        if key not in {"quantization", "lora"}
    }
    return SFTConfig(**top_level, quantization=quantization, lora=lora)


def load_config(path: str | Path) -> SFTConfig:
    """Load a JSON configuration file and validate it."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON configuration: {config_path}: {exc}") from exc
    return config_from_dict(raw)

