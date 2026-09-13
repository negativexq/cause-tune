from __future__ import annotations

from causetune.benchmark_e05 import generate_e05_benchmark


def test_e05_has_declared_size_origins_and_new_topology_namespace() -> None:
    inputs, truth, provenance = generate_e05_benchmark()
    assert sum(len(rows) for rows in inputs.values()) == 120
    assert len(truth) == len(provenance) == 120
    assert {row["origin"] for row in provenance} == {"generated", "static"}
    assert sum(row["origin"] == "generated" for row in provenance) == 96
    assert sum(row["origin"] == "static" for row in provenance) == 24
    assert len({row["topology_family"] for row in provenance}) == 8
    assert all(row["incident_id"].startswith("e05-") for row in provenance)


def test_e05_has_no_internal_packet_or_normalized_duplicates() -> None:
    inputs, _, _ = generate_e05_benchmark()
    rows = [row for split in ("standard", "hard", "transfer") for row in inputs[split]]
    assert len({row["incident_packet"] for row in rows}) == 120
