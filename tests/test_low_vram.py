from __future__ import annotations

import pytest

from causetune.low_vram import evaluate_glv0_parity, evaluate_glv1_systems, evaluate_glv2_admission


def _trace(value: float = 1.0) -> dict:
    return {
        "forward_loss": value,
        "gradient": [value, value + 0.1],
        "lora_gradient": {"a": [value + 0.2]},
        "optimizer_update": [value + 0.3, value + 0.4],
    }


def test_glv0_requires_all_parity_layers() -> None:
    result = evaluate_glv0_parity(_trace(), _trace())
    assert result["passed"] is True
    assert {check["name"] for check in result["checks"]} == {
        "forward_loss", "gradient", "lora_gradient", "optimizer_update"
    }
    streamed = _trace()
    streamed["optimizer_update"][0] += 0.1
    assert evaluate_glv0_parity(_trace(), streamed)["passed"] is False


def test_glv1_requires_controlled_metrics() -> None:
    resident = {"peak_allocated_vram": 8, "host_ram": 10, "disk_io": 1, "wall_time": 4, "tokens_per_second": 10, "quality": 0.8}
    streamed = {"peak_allocated_vram": 4, "host_ram": 12, "disk_io": 3, "wall_time": 8, "tokens_per_second": 5, "quality": 0.79}
    result = evaluate_glv1_systems(resident=resident, streamed=streamed, quality_tolerance=0.02)
    assert result["memory_reduction"] == 4
    assert result["quality_within_tolerance"] is True


def test_glv2_admission_requires_correctness_memory_quality_and_repeatability() -> None:
    glv0 = evaluate_glv0_parity(_trace(), _trace())
    glv1 = evaluate_glv1_systems(
        resident={"peak_allocated_vram": 8, "host_ram": 10, "disk_io": 1, "wall_time": 4, "tokens_per_second": 10, "quality": 0.8},
        streamed={"peak_allocated_vram": 4, "host_ram": 12, "disk_io": 3, "wall_time": 8, "tokens_per_second": 5, "quality": 0.79},
        quality_tolerance=0.02,
    )
    assert evaluate_glv2_admission(glv0, glv1, repeated=True)["admitted"] is True
    assert evaluate_glv2_admission(glv0, glv1, repeated=False)["admitted"] is False
