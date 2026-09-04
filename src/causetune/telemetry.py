"""CUDA and training telemetry helpers."""

from __future__ import annotations

import math
from typing import Any

import torch


def synchronize_cuda() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cuda_memory_snapshot() -> dict[str, int]:
    """Return allocated/reserved CUDA bytes and tracked peaks."""

    if not torch.cuda.is_available():
        return {
            "allocated_bytes": 0,
            "reserved_bytes": 0,
            "peak_allocated_bytes": 0,
            "peak_reserved_bytes": 0,
        }
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def assert_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        raise FloatingPointError(f"{name} is not finite: {value}")


def json_safe_memory(snapshot: dict[str, int]) -> dict[str, Any]:
    """Add GiB values while retaining exact byte measurements."""

    result: dict[str, Any] = dict(snapshot)
    for name, value in snapshot.items():
        if name.endswith("_bytes"):
            result[name.replace("_bytes", "_gib")] = round(value / 1024**3, 6)
    return result

