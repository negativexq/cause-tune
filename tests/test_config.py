from __future__ import annotations

import unittest
from pathlib import Path

from causetune.config import load_config


class ConfigurationTests(unittest.TestCase):
    def test_configuration_loading(self) -> None:
        config = load_config(Path("configs/sft_smoke.json"))
        self.assertEqual(config.max_examples, 200)
        self.assertEqual(config.train_ratio, 0.8)
        self.assertEqual(config.validation_ratio, 0.1)
        self.assertEqual(config.test_ratio, 0.1)
        self.assertEqual(config.max_sequence_length, 1024)
        self.assertFalse(config.gradient_checkpointing_use_reentrant)
        self.assertEqual(config.quantization.quant_type, "nf4")
        self.assertEqual(config.quantization.compute_dtype, "bfloat16")
        self.assertEqual(
            config.lora.target_modules,
            (
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ),
        )
        self.assertIn('"model_id": "Qwen/Qwen3-4B"', config.human_readable())
