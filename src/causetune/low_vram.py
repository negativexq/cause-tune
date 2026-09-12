"""Admission gates for the optional low-VRAM/streaming research track."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


class LowVramAdmissionError(ValueError):
    """Raised when a low-VRAM backend has not earned experimental admission."""


PARITY_CHECKS = ("forward_loss", "gradient", "lora_gradient", "optimizer_update")


@dataclass(frozen=True)
class ParityCheck:
    name: str
    passed: bool
    max_abs_error: float
    tolerance: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "max_abs_error": self.max_abs_error,
            "tolerance": self.tolerance,
        }


def _max_abs(left: Any, right: Any) -> float:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left) != set(right):
            return math.inf
        return max((_max_abs(left[key], right[key]) for key in left), default=0.0)
    if isinstance(left, Sequence) and not isinstance(left, (str, bytes)) and isinstance(right, Sequence) and not isinstance(right, (str, bytes)):
        if len(left) != len(right):
            return math.inf
        return max((_max_abs(a, b) for a, b in zip(left, right)), default=0.0)
    try:
        return abs(float(left) - float(right))
    except (TypeError, ValueError):
        return 0.0 if left == right else math.inf


def evaluate_glv0_parity(
    resident: Mapping[str, Any],
    streamed: Mapping[str, Any],
    *,
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Require loss, gradients and optimizer update parity—not loss alone."""

    checks: list[ParityCheck] = []
    for name in PARITY_CHECKS:
        if name not in resident or name not in streamed:
            checks.append(ParityCheck(name, False, math.inf, tolerance))
            continue
        error = _max_abs(resident[name], streamed[name])
        checks.append(ParityCheck(name, error <= tolerance, error, tolerance))
    return {
        "gate": "GLV0",
        "checks": [check.to_dict() for check in checks],
        "passed": all(check.passed for check in checks),
        "loss_only_is_sufficient": False,
    }


def evaluate_glv1_systems(
    *,
    resident: Mapping[str, float],
    streamed: Mapping[str, float],
    quality_tolerance: float,
) -> dict[str, Any]:
    required = ("peak_allocated_vram", "host_ram", "disk_io", "wall_time", "tokens_per_second", "quality")
    if any(key not in resident or key not in streamed for key in required):
        raise LowVramAdmissionError("GLV1 requires the complete controlled systems metric set")
    return {
        "gate": "GLV1",
        "resident": dict(resident),
        "streamed": dict(streamed),
        "quality_delta": streamed["quality"] - resident["quality"],
        "quality_within_tolerance": abs(streamed["quality"] - resident["quality"]) <= quality_tolerance,
        "memory_reduction": resident["peak_allocated_vram"] - streamed["peak_allocated_vram"],
        "quality_tolerance": quality_tolerance,
    }


def evaluate_glv2_admission(
    glv0: Mapping[str, Any],
    glv1: Mapping[str, Any],
    *,
    repeated: bool,
) -> dict[str, Any]:
    correctness = bool(glv0.get("passed"))
    meaningful_memory_reduction = float(glv1.get("memory_reduction", 0)) > 0
    quality_ok = bool(glv1.get("quality_within_tolerance"))
    result = {
        "gate": "GLV2",
        "correctness_pass": correctness,
        "meaningful_memory_reduction": meaningful_memory_reduction,
        "quality_within_tolerance": quality_ok,
        "repeatable": repeated,
        "admitted": correctness and meaningful_memory_reduction and quality_ok and repeated,
        "backend_status": "experimental" if correctness and meaningful_memory_reduction and quality_ok and repeated else "not_admitted",
    }
    return result
