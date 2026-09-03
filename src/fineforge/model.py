"""Quantized QLoRA model construction for the FineForge smoke run."""

from __future__ import annotations

from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import (
    LoraConfig as PeftLoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from .config import SFTConfig


def load_tokenizer(config: SFTConfig) -> Any:
    """Load only tokenizer assets; this function does not load model weights."""

    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def make_quantization_config(config: SFTConfig) -> BitsAndBytesConfig:
    """Build the resolved NF4/double-quantization configuration."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA smoke run")
    compute_dtype = getattr(torch, config.quantization.compute_dtype)
    return BitsAndBytesConfig(
        load_in_4bit=config.quantization.load_in_4bit,
        bnb_4bit_quant_type=config.quantization.quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.quantization.double_quant,
    )


def load_quantized_base(config: SFTConfig) -> Any:
    """Load a fresh Qwen3-4B NF4 base model on the local CUDA device."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the QLoRA smoke run")
    device_index = torch.cuda.current_device()
    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        quantization_config=make_quantization_config(config),
        dtype=torch.bfloat16,
        device_map={"": device_index},
    )
    model.config.use_cache = False
    return model


def attach_lora(model: Any, config: SFTConfig) -> Any:
    """Prepare a quantized model and attach the configured PEFT adapter."""

    model.config.use_cache = False
    checkpointing_kwargs = None
    if config.gradient_checkpointing:
        checkpointing_kwargs = {
            "use_reentrant": config.gradient_checkpointing_use_reentrant,
        }
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=checkpointing_kwargs,
    )
    lora_config = PeftLoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(config.lora.target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.config.use_cache = False
    return model


def load_adapter(base_model: Any, adapter_dir: str) -> Any:
    """Attach a saved adapter to a fresh quantized base model."""

    return PeftModel.from_pretrained(base_model, adapter_dir, is_trainable=False)


def adapter_parameter_count(model: Any) -> int:
    """Count materialized LoRA parameters, including frozen reloaded adapters."""

    return sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if "lora_" in name
    )

