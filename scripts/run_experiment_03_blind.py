#!/usr/bin/env python3
"""Run the frozen Experiment 03 base/tuned blind one-shot evaluation."""

from __future__ import annotations

import argparse
import gc
import json
import random
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from causetune.benchmark_blind_v2 import SCORER_VERSION, scorer_fingerprint
from causetune.evidence import sha256_path
from causetune.incident_evaluation import evaluate_incidents
from causetune.model import load_adapter, load_frozen_quantized_base, load_tokenizer_for_model


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _prompt_ids(tokenizer: Any, system_instruction: str, packet: str) -> list[int]:
    messages = [{"role": "system", "content": system_instruction}, {"role": "user", "content": packet}]
    try:
        values = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        values = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
    if isinstance(values, Mapping):
        values = values["input_ids"]
    if hasattr(values, "tolist"):
        values = values.tolist()
    if values and isinstance(values[0], list):
        values = values[0]
    return [int(value) for value in values]


def _generate(model: Any, tokenizer: Any, records: list[dict[str, Any]], system_instruction: str) -> dict[str, str]:
    model.eval()
    device = next(model.parameters()).device
    outputs: dict[str, str] = {}
    generation = {"max_new_tokens": 96, "batch_size": 4}
    with torch.inference_mode():
        for start in range(0, len(records), generation["batch_size"]):
            batch = records[start : start + generation["batch_size"]]
            prompts = [_prompt_ids(tokenizer, system_instruction, item["incident_packet"]) for item in batch]
            width = max(len(prompt) for prompt in prompts)
            input_ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            attention = torch.zeros_like(input_ids)
            for row, prompt in enumerate(prompts):
                input_ids[row, width - len(prompt) :] = torch.tensor(prompt, dtype=torch.long, device=device)
                attention[row, width - len(prompt) :] = 1
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                max_new_tokens=generation["max_new_tokens"],
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            for item, sequence in zip(batch, generated):
                outputs[item["incident_id"]] = tokenizer.decode(sequence[width:], skip_special_tokens=True)
            print(f"generated={min(start + len(batch), len(records))}/{len(records)}", flush=True)
    return outputs


def _without_predictions(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "predictions"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="data/incident_diagnosis_blind_v2")
    parser.add_argument("--adapter", default="outputs/incident_diagnosis_02b2/checkpoint-step-000100")
    parser.add_argument("--output-dir", default="results/experiment_03_blind")
    parser.add_argument("--model-id", default="Qwen/Qwen3-4B")
    parser.add_argument("--seed", type=int, default=20260912)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    manifest = json.loads((dataset_dir / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("frozen") or manifest.get("scorer_version") != SCORER_VERSION:
        raise ValueError("blind benchmark or scorer is not frozen")
    records = [record for split in ("standard", "hard", "transfer") for record in _read_jsonl(dataset_dir / f"{split}.jsonl")]
    truth = _read_jsonl(dataset_dir / "ground_truth.jsonl")
    truth_by_id = {row["incident_id"]: row for row in truth}
    config = json.loads(Path("configs/incident_diagnosis_eval.json").read_text(encoding="utf-8"))
    system_instruction = config["evaluation_contract"]["system_instruction"]
    protocol = {
        "experiment": "E03",
        "protocol_version": "blind-one-shot-v1",
        "benchmark_fingerprint": manifest["fingerprint"],
        "scorer_version": SCORER_VERSION,
        "scorer_fingerprint": scorer_fingerprint(),
        "prompt_frozen": True,
        "checkpoint_switching": False,
        "model_id": args.model_id,
        "adapter": args.adapter,
    }
    _write_json(output_dir / "protocol.json", protocol)
    if not torch.cuda.is_available():
        raise RuntimeError("E03 requires CUDA; no scientific CPU fallback is allowed")
    tokenizer = load_tokenizer_for_model(args.model_id)
    base_model = load_frozen_quantized_base(args.model_id, **config["quantization"])
    base_raw = _generate(base_model, tokenizer, records, system_instruction)
    base_metrics = evaluate_incidents(records, truth_by_id, base_raw)
    _write_jsonl(output_dir / "base_predictions.jsonl", base_metrics["predictions"])
    _write_json(output_dir / "base_evaluation.json", _without_predictions(base_metrics))
    del base_model
    gc.collect()
    torch.cuda.empty_cache()
    tuned_base = load_frozen_quantized_base(args.model_id, **config["quantization"])
    tuned_model = load_adapter(tuned_base, args.adapter)
    tuned_raw = _generate(tuned_model, tokenizer, records, system_instruction)
    tuned_metrics = evaluate_incidents(records, truth_by_id, tuned_raw)
    _write_jsonl(output_dir / "tuned_predictions.jsonl", tuned_metrics["predictions"])
    _write_json(output_dir / "tuned_evaluation.json", _without_predictions(tuned_metrics))
    comparison = {
        "benchmark_fingerprint": manifest["fingerprint"],
        "base": _without_predictions(base_metrics),
        "tuned": _without_predictions(tuned_metrics),
        "diagnosis_exact_delta": tuned_metrics["diagnosis_exact_match"]["rate"] - base_metrics["diagnosis_exact_match"]["rate"],
        "resolution_exact_delta": tuned_metrics["resolution_exact_match"]["rate"] - base_metrics["resolution_exact_match"]["rate"],
    }
    _write_json(output_dir / "comparison.json", comparison)
    _write_json(
        output_dir / "artifact_fingerprints.json",
        {"benchmark_dir_sha256": sha256_path(dataset_dir), "artifacts": {path.name: sha256_path(path) if path.is_dir() else __import__("hashlib").sha256(path.read_bytes()).hexdigest() for path in sorted(output_dir.iterdir()) if path.is_file()}},
    )
    print(json.dumps({"status": "PASS", "output_dir": str(output_dir), "comparison": comparison}, sort_keys=True))


if __name__ == "__main__":
    main()
