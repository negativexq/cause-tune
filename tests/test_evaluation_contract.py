from __future__ import annotations

import unittest
import json
from pathlib import Path

import torch

from fineforge.data.generate import generate_smoke_records
from fineforge.data.preprocess import build_preprocessed_example
from fineforge.evaluation import (
    diagnose_generation_output,
    greedy_intent_metrics,
)
from fineforge.evaluation_contract import (
    PROMPT_VERSION,
    SYSTEM_INSTRUCTION,
    contract_fingerprint,
    evaluation_messages,
)

from test_preprocess import FakeChatTokenizer


class RecordingTokenizer(FakeChatTokenizer):
    pad_token_id = 0
    eos_token_id = 2

    def __init__(self) -> None:
        self.seen_messages = []

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.seen_messages.append(list(messages))
        return super().apply_chat_template(messages, tokenize, add_generation_prompt)

    def decode(self, token_ids, skip_special_tokens=False):
        return '{"intent":"refund"}'


class FixedGenerationModel(torch.nn.Module):
    def __init__(self, tokenizer: RecordingTokenizer) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.tokenizer = tokenizer

    def generate(self, input_ids, attention_mask, **kwargs):
        completion = torch.tensor(
            [ord(character) for character in '{"intent":"refund"}'],
            dtype=torch.long,
            device=input_ids.device,
        )
        return torch.cat(
            [input_ids, completion.repeat(input_ids.shape[0], 1)],
            dim=1,
        )


class EvaluationContractTests(unittest.TestCase):
    def test_contract_is_explicit_and_fingerprinted(self) -> None:
        self.assertEqual(PROMPT_VERSION, "m4-frozen-contract-v1")
        self.assertIn("intent classification", SYSTEM_INSTRUCTION)
        self.assertIn("JSON only", SYSTEM_INSTRUCTION)
        self.assertIn("no additional keys", SYSTEM_INSTRUCTION)
        config = {
            "prompt_version": PROMPT_VERSION,
            "tokenizer": {
                "enable_thinking": False,
                "add_generation_prompt": True,
                "skip_special_tokens": True,
                "padding_side": "left",
            },
            "evaluation_contract": {
                "system_instruction": SYSTEM_INSTRUCTION,
                "generation": {"do_sample": False, "max_new_tokens": 32, "batch_size": 8},
                "parsing": {"allow_surrounding_text": False, "repair_malformed_output": False},
                "schema": {"exact_keys": ["intent"]},
            },
        }
        self.assertEqual(len(contract_fingerprint(config)), 64)
        messages = evaluation_messages({"messages": [{"role": "user", "content": "hello"}]})
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], SYSTEM_INSTRUCTION)
        self.assertEqual(messages[1]["content"], "hello")

    def test_persisted_baseline_config_matches_contract(self) -> None:
        config = json.loads(Path("configs/baseline_eval.json").read_text(encoding="utf-8"))
        self.assertEqual(config["evaluation_contract"]["system_instruction"], SYSTEM_INSTRUCTION)
        self.assertEqual(config["contract_fingerprint"], contract_fingerprint(config))
        self.assertFalse(config["tokenizer"]["enable_thinking"])
        self.assertFalse(config["evaluation_contract"]["generation"]["do_sample"])

    def test_preprocessing_accepts_same_system_prompt(self) -> None:
        record = generate_smoke_records(seed=42)[0]
        tokenizer = FakeChatTokenizer()
        plain = build_preprocessed_example(record, tokenizer, 1024)
        instructed = build_preprocessed_example(record, tokenizer, 1024, SYSTEM_INSTRUCTION)
        self.assertGreater(instructed.prompt_token_count, plain.prompt_token_count)
        self.assertEqual(
            instructed.trainable_assistant_token_count,
            plain.trainable_assistant_token_count,
        )

    def test_generation_decodes_only_completion_after_left_padding(self) -> None:
        tokenizer = RecordingTokenizer()
        records = generate_smoke_records(seed=42)[:2]
        results = greedy_intent_metrics(
            FixedGenerationModel(tokenizer),
            tokenizer,
            records,
            batch_size=2,
            system_message=SYSTEM_INSTRUCTION,
        )
        self.assertEqual(results["intent_accuracy"], 1.0)
        self.assertEqual(results["valid_json_rate"], 1.0)
        self.assertEqual(results["predictions"][0]["raw_text"], '{"intent":"refund"}')
        self.assertEqual(tokenizer.seen_messages[0][0]["role"], "system")
        self.assertTrue(all(not item["raw_text"].startswith("<") for item in results["predictions"]))

    def test_diagnostic_categories_do_not_repair(self) -> None:
        self.assertEqual(diagnose_generation_output('{"intent":"refund"}', "refund"), "valid JSON + correct intent")
        self.assertEqual(diagnose_generation_output('{"intent":"refund"}', "wrong_item"), "valid JSON + wrong intent")
        self.assertEqual(diagnose_generation_output('{"intent":"unknown"}', "refund"), "unknown label")
        self.assertEqual(diagnose_generation_output('{"intent":"refund","x":1}', "refund"), "unexpected additional keys")
        self.assertEqual(diagnose_generation_output('prefix {\"intent\":\"refund\"}', "refund"), "invalid JSON")
        self.assertEqual(diagnose_generation_output('{\"intent\":\"refund\"} trailing', "refund"), "unexpected extra text")
        self.assertEqual(diagnose_generation_output("plain prose", "refund"), "invalid JSON")


if __name__ == "__main__":
    unittest.main()
