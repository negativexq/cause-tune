from __future__ import annotations

from causetune.e04c_selection import VARIANT_ORDER, select_variants


def _row(variant: str, learning_rate: float, diagnosis: float, wall: float) -> dict:
    return {
        "variant": variant,
        "learning_rate": learning_rate,
        "metrics": {
            "diagnosis_exact_match": {"rate": diagnosis},
            "resolution_exact_match": {"rate": 1.0},
            "failure_mode_macro_f1": 1.0,
            "action_accuracy": {"rate": 1.0},
            "evidence_f1": 1.0,
            "valid_json": 1.0,
            "strict_json": {"rate": 1.0},
        },
        "failure_family_metrics": {"family": {"failure_mode_f1": 1.0}},
        "checkpoint_stability": {"stable_validation_count": 4, "earliest_within_tolerance_step": 25, "selected_checkpoint_step": 100},
        "evidence": {"verification": "PASS"},
        "cost": {"wall_clock_training_seconds": wall},
    }


def test_cost_breaks_a_quality_and_stability_tie() -> None:
    rows = [_row("lr1e-4", 1e-4, 1.0, 10.0), _row("lr2e-4", 2e-4, 1.0, 20.0), _row("lr4e-4", 4e-4, 1.0, 30.0)]
    result = select_variants(rows)
    assert result["reference_variant"] == "lr1e-4"
    assert result["selected_variant"] == "lr1e-4"
    assert result["eligible_variants"] == list(VARIANT_ORDER)


def test_quality_priority_beats_training_cost() -> None:
    rows = [_row("lr1e-4", 1e-4, 0.98, 1.0), _row("lr2e-4", 2e-4, 1.0, 20.0), _row("lr4e-4", 4e-4, 1.0, 30.0)]
    result = select_variants(rows)
    assert result["selected_variant"] == "lr2e-4"
    assert "lr1e-4" in result["rejected_variants"]
