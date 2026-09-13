from __future__ import annotations

from causetune.e04b_selection import VARIANT_ORDER, select_variants


def _row(variant: str, rank: int, diagnosis: float) -> dict:
    return {
        "variant": variant,
        "rank": rank,
        "metrics": {
            "diagnosis_exact_match": {"rate": diagnosis},
            "resolution_exact_match": {"rate": 1.0},
            "failure_mode_macro_f1": 1.0,
            "valid_json": 1.0,
            "strict_json": {"rate": 1.0},
        },
        "failure_family_metrics": {"family": {"failure_mode_f1": 1.0}},
        "evidence": {"verification": "PASS"},
        "cost": {"trainable_parameters": rank * 100},
    }


def test_smallest_eligible_rank_is_selected() -> None:
    rows = [_row("r8", 8, 1.0), _row("r16", 16, 0.99), _row("r32", 32, 1.0)]
    result = select_variants(rows)
    assert result["reference_variant"] == "r8"
    assert result["selected_rank"] == 8
    assert result["eligible_variants"] == list(VARIANT_ORDER)


def test_rank_with_quality_regression_is_rejected() -> None:
    rows = [_row("r8", 8, 0.98), _row("r16", 16, 0.99), _row("r32", 32, 1.0)]
    result = select_variants(rows)
    assert result["selected_rank"] == 16
    assert "r8" in result["rejected_variants"]
