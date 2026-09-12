"""Disposable model-backed compatibility smoke for CI.

This is intentionally not a quality experiment. It proves that the current
Transformers, PEFT and Trainer path can tokenize, mask assistant supervision,
step an optimizer, persist an adapter, reload it into a fresh base and run
generation on CPU.
"""

from __future__ import annotations

import gc
import importlib.metadata
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SmokeResult:
    model_id: str
    optimizer_steps: int
    train_loss: float
    supervised_examples: int
    adapter_dir: str
    generated_token_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": "PASS",
            "model_id": self.model_id,
            "optimizer_steps": self.optimizer_steps,
            "train_loss": self.train_loss,
            "supervised_examples": self.supervised_examples,
            "adapter_dir": self.adapter_dir,
            "generated_token_count": self.generated_token_count,
        }


def _seed(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)


def _target_modules(model: Any) -> list[str]:
    candidates = ("c_attn", "q_proj", "v_proj", "query_key_value")
    names = {name.rsplit(".", 1)[-1] for name, _module in model.named_modules()}
    targets = [candidate for candidate in candidates if candidate in names]
    if not targets:
        raise RuntimeError("could not find a PEFT target module in tiny smoke model")
    return targets[:2] if "c_attn" not in targets else ["c_attn"]


def run_model_backed_smoke(
    *,
    model_id: str = "hf-internal-testing/tiny-random-GPT2",
    output_dir: str | Path = "outputs/model_backed_smoke",
    local_files_only: bool = False,
    seed: int = 7,
) -> SmokeResult:
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator

    _seed(seed)
    destination = Path(output_dir)
    adapter_dir = destination / "adapter"
    destination.mkdir(parents=True, exist_ok=True)
    if any(adapter_dir.iterdir()) if adapter_dir.exists() else False:
        raise RuntimeError(f"smoke adapter directory is not empty: {adapter_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
    model.config.use_cache = False
    targets = _target_modules(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=2,
            lora_alpha=4,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
        ),
    )

    examples: list[dict[str, Any]] = []
    for prompt, answer in (("user: alpha\nassistant:", " A"), ("user: beta\nassistant:", " B")):
        full = tokenizer(prompt + answer, add_special_tokens=True, truncation=True, max_length=32)
        prompt_ids = tokenizer(prompt, add_special_tokens=True, truncation=True, max_length=32)["input_ids"]
        labels = list(full["input_ids"])
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        if not any(label != -100 for label in labels):
            raise AssertionError("smoke fixture has no assistant supervision")
        examples.append({"input_ids": full["input_ids"], "attention_mask": full["attention_mask"], "labels": labels})

    training_arguments = {
        "output_dir": str(destination / "trainer"),
        "max_steps": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 1e-3,
        "logging_steps": 1,
        "logging_strategy": "steps",
        "save_strategy": "no",
        "report_to": "none",
        "remove_unused_columns": False,
        "optim": "adamw_torch",
        "seed": seed,
    }
    # ``use_cpu`` is the current name; ``no_cuda`` keeps this smoke runnable
    # on the oldest supported Transformers minor without changing its scope.
    import inspect

    if "use_cpu" in inspect.signature(TrainingArguments).parameters:
        training_arguments["use_cpu"] = True
    else:
        training_arguments["no_cuda"] = True
    args = TrainingArguments(**training_arguments)
    trainer = Trainer(model=model, args=args, train_dataset=examples, data_collator=default_data_collator)
    result = trainer.train()
    train_loss = float(result.training_loss)
    if not math.isfinite(train_loss):
        raise RuntimeError(f"model-backed smoke loss is not finite: {train_loss}")
    model.save_pretrained(adapter_dir, safe_serialization=True)
    tokenizer.save_pretrained(adapter_dir)
    del trainer, model
    gc.collect()

    fresh_base = AutoModelForCausalLM.from_pretrained(model_id, local_files_only=local_files_only)
    fresh_model = PeftModel.from_pretrained(fresh_base, adapter_dir, is_trainable=False)
    fresh_model.eval()
    prompt = tokenizer("user: gamma\nassistant:", return_tensors="pt")
    with torch.no_grad():
        generated = fresh_model.generate(**prompt, max_new_tokens=2, do_sample=False)
    generated_count = int(generated.shape[-1] - prompt["input_ids"].shape[-1])
    if generated_count <= 0:
        raise RuntimeError("fresh adapter reload produced no generated tokens")
    del fresh_model, fresh_base
    gc.collect()
    return SmokeResult(
        model_id=model_id,
        optimizer_steps=2,
        train_loss=train_loss,
        supervised_examples=len(examples),
        adapter_dir=str(adapter_dir),
        generated_token_count=generated_count,
    )


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("torch", "transformers", "peft", "accelerate"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions
