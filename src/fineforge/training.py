"""Small explicit PyTorch QLoRA training loop for the smoke milestone."""

from __future__ import annotations

import time
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from .config import SFTConfig
from .data.preprocess import PreprocessedExample
from .evaluation import collate_preprocessed
from .telemetry import (
    assert_finite,
    cuda_memory_snapshot,
    json_safe_memory,
    synchronize_cuda,
)


class QLoRATrainingOOMError(RuntimeError):
    """A non-retried CUDA OOM with stage and memory evidence attached."""

    def __init__(self, stage: str, snapshot: dict[str, Any], cause: BaseException):
        super().__init__(
            f"CUDA OOM during {stage}; peak allocated={snapshot['peak_allocated_gib']:.3f} GiB, "
            f"peak reserved={snapshot['peak_reserved_gib']:.3f} GiB"
        )
        self.stage = stage
        self.snapshot = snapshot
        self.cause = cause


def count_supervised_tokens_after_shift(labels: torch.Tensor) -> int:
    """Count non-ignored causal targets after the model's next-token shift."""

    if labels.ndim == 1:
        labels = labels.unsqueeze(0)
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, sequence], got {labels.shape}")
    return int((labels[..., 1:] != -100).sum().item())


def count_model_input_tokens(attention_mask: torch.Tensor) -> int:
    """Count real input positions, excluding right-padding."""

    return int(attention_mask.sum().item())


def count_preprocessed_supervised_tokens(
    examples: Sequence[PreprocessedExample],
) -> int:
    """Count supervised targets from preprocessed examples after causal shift."""

    return sum(
        sum(label != -100 for label in example.labels[1:])
        for example in examples
    )


def count_preprocessed_input_tokens(
    examples: Sequence[PreprocessedExample],
) -> int:
    """Count unpadded model input tokens in preprocessed examples."""

    return sum(len(example.input_ids) for example in examples)


def train_qlora(
    model: Any,
    tokenizer: Any,
    train_examples: Sequence[PreprocessedExample],
    config: SFTConfig,
) -> tuple[dict[str, Any], list[dict[str, float | int]]]:
    """Run exactly one configured epoch and return optimizer-step telemetry."""

    if not train_examples:
        raise ValueError("training split is empty")
    if any(example.trainable_assistant_token_count < 1 for example in train_examples):
        raise ValueError("every training example must have a non-ignored assistant label")

    loader = DataLoader(
        list(train_examples),
        batch_size=config.micro_batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_preprocessed(batch, tokenizer.pad_token_id),
    )
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    optimizer.zero_grad(set_to_none=True)
    model.train()
    device = next(model.parameters()).device
    model_loaded_memory = json_safe_memory(cuda_memory_snapshot())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_history: list[dict[str, float | int]] = []
    micro_losses: list[float] = []
    processed_examples = 0
    supervised_assistant_tokens_processed = 0
    model_input_tokens_processed = 0
    optimizer_steps = 0
    stage = "training setup"

    synchronize_cuda()
    start_time = time.perf_counter()
    try:
        for micro_step, batch in enumerate(loader, start=1):
            processed_examples += len(batch["input_ids"])
            batch = {key: value.to(device) for key, value in batch.items()}
            stage = f"forward/backward micro-step {micro_step}"
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss
            loss_value = float(loss.detach())
            assert_finite(loss_value, f"micro-step {micro_step} loss")
            micro_losses.append(loss_value)
            supervised_assistant_tokens_processed += count_supervised_tokens_after_shift(
                batch["labels"]
            )
            model_input_tokens_processed += count_model_input_tokens(
                batch["attention_mask"]
            )
            (loss / config.gradient_accumulation_steps).backward()

            is_accumulation_boundary = (
                micro_step % config.gradient_accumulation_steps == 0
                or micro_step == len(loader)
            )
            if is_accumulation_boundary:
                stage = f"optimizer step {optimizer_steps + 1}"
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                step_loss = sum(micro_losses) / len(micro_losses)
                assert_finite(step_loss, f"optimizer step {optimizer_steps} loss")
                loss_history.append(
                    {
                        "optimizer_step": optimizer_steps,
                        "loss": step_loss,
                        "micro_steps": len(micro_losses),
                    }
                )
                micro_losses.clear()
    except torch.cuda.OutOfMemoryError as exc:
        synchronize_cuda()
        snapshot = json_safe_memory(cuda_memory_snapshot())
        raise QLoRATrainingOOMError(stage, snapshot, exc) from exc

    synchronize_cuda()
    duration = time.perf_counter() - start_time
    snapshot = json_safe_memory(cuda_memory_snapshot())
    losses = [float(item["loss"]) for item in loss_history]
    all_finite = all(torch.isfinite(torch.tensor(loss)) for loss in losses)
    if not losses:
        raise RuntimeError("training completed without an optimizer step")
    if supervised_assistant_tokens_processed != count_preprocessed_supervised_tokens(
        train_examples
    ):
        raise AssertionError("batched supervised-token count differs from preprocessing")
    if model_input_tokens_processed != count_preprocessed_input_tokens(train_examples):
        raise AssertionError("batched input-token count differs from preprocessing")
    return {
        "trainable_parameter_count": model.get_nb_trainable_parameters()[0]
        if hasattr(model, "get_nb_trainable_parameters")
        else sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "logical_parameter_count": model.get_nb_trainable_parameters()[1]
        if hasattr(model, "get_nb_trainable_parameters")
        else sum(parameter.numel() for parameter in model.parameters()),
        "training_start_model_loaded_vram": model_loaded_memory,
        "training_peak_cuda_vram": snapshot,
        "wall_clock_training_seconds": duration,
        "optimizer_steps": optimizer_steps,
        "processed_training_examples": processed_examples,
        "effective_training_tokens": supervised_assistant_tokens_processed,
        "supervised_assistant_tokens_processed": supervised_assistant_tokens_processed,
        "causal_shifted_supervised_assistant_tokens": supervised_assistant_tokens_processed,
        "model_input_tokens_processed": model_input_tokens_processed,
        "supervised_assistant_tokens_per_second": (
            supervised_assistant_tokens_processed / duration
        ),
        "input_tokens_per_second": model_input_tokens_processed / duration,
        # Retain the original key as a compatibility alias with an explicit
        # supervised-token definition.
        "training_tokens_per_second": supervised_assistant_tokens_processed / duration,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "all_losses_finite": bool(all_finite),
    }, loss_history
