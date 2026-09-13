from __future__ import annotations

from causetune.e04a_selection import VARIANT_ORDER, select_variants


def _row(variant: str, fraction: float, diagnosis: float, resolution: float, f1: float) -> dict:
    families = {"family": {"failure_mode_f1": f1}}
    return {
        "variant": variant,
        "fraction": fraction,
        "subset_hash": f"hash-{variant}",
        "metrics": {
            "diagnosis_exact_match": {"rate": diagnosis},
            "resolution_exact_match": {"rate": resolution},
            "failure_mode_macro_f1": f1,
            "valid_json": 1.0,
            "strict_json": {"rate": 1.0},
        },
        "failure_family_metrics": families,
        "evidence": {"verification": "PASS"},
    }


def test_selection_uses_smallest_eligible_fraction() -> None:
    rows = [
        _row("025pct", 0.25, 1.0, 1.0, 1.0),
        _row("050pct", 0.50, 0.99, 1.0, 1.0),
        _row("075pct-retry-01", 0.75, 0.99, 0.99, 1.0),
        _row("100pct", 1.0, 1.0, 1.0, 1.0),
    ]
    result = select_variants(rows)
    assert result["reference_variant"] == "025pct"
    assert result["selected_variant"] == "025pct"
    assert result["eligible_variants"] == list(VARIANT_ORDER)


def test_selection_rejects_more_than_one_point_regression() -> None:
    rows = [
        _row("025pct", 0.25, 0.98, 1.0, 1.0),
        _row("050pct", 0.50, 0.98, 1.0, 1.0),
        _row("075pct-retry-01", 0.75, 0.99, 0.99, 1.0),
        _row("100pct", 1.0, 1.0, 1.0, 1.0),
    ]
    result = select_variants(rows)
    assert result["selected_variant"] == "075pct-retry-01"
    assert "050pct" in result["rejected_variants"]
