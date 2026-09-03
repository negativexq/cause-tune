#!/usr/bin/env python3
"""Inspect one preprocessed example without loading model weights."""

from __future__ import annotations

import argparse

from transformers import AutoTokenizer

from fineforge.config import load_config
from fineforge.data.preprocess import decode_trainable_assistant, preprocess_records
from fineforge.data.schema import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sft_smoke.json")
    parser.add_argument("--dataset", help="override config.dataset_path")
    parser.add_argument("--tokenizer-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--example-id")
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_jsonl(args.dataset or config.dataset_path)
    if args.example_id:
        records = [record for record in records if record["example_id"] == args.example_id]
        if not records:
            raise ValueError(f"example_id not found: {args.example_id}")
    else:
        records = records[:1]

    print(f"Loading tokenizer only: {args.tokenizer_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)
    examples = preprocess_records(records, tokenizer, config.max_sequence_length)
    for example in examples:
        print("\n--- PREPROCESSING INSPECTION ---")
        print("example_id:", example.example_id)
        print("rendered conversation:")
        print(example.rendered_conversation)
        print("input token count:", len(example.input_ids))
        print("masked token count:", example.masked_token_count)
        print("trainable assistant tokens:", example.trainable_assistant_token_count)
        print("decoded trainable assistant portion:")
        print(decode_trainable_assistant(example, tokenizer))


if __name__ == "__main__":
    main()

