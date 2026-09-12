"""Independent blind generalization benchmark for Experiment 03.

This module intentionally does not call the original incident generator. It
reuses only the frozen target taxonomy and packet-validation boundary; wording,
case ordering, narrative templates and topology assignment are defined here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .incident_benchmark import packet_evidence_ids, validate_input_record
from .incident_taxonomy import ACTION_SET, FAILURE_FAMILY_SET, FAILURE_FAMILIES, FAILURE_SPECS, SLICES, TOPOLOGIES


BENCHMARK_VERSION = "incident-blind-v2"
GENERATOR_VERSION = "independent-blind-namespace-v1"
SCORER_VERSION = "incident-scorer-v1"
_SPLIT_BY_CASE = ("standard", "standard", "hard", "hard", "transfer")
_NARRATIVES = (
    "The handoff was written after mitigation, so the first causal signal matters more than the loudest alert.",
    "The incident room lists a recovered dependency beside the failing path; treat recovery as evidence, not cause.",
    "A customer-facing symptom arrived before the platform notification and the operator preserved the order of events.",
    "The packet uses unfamiliar service names and a different timeline style from the training corpus.",
    "One correlated signal is deliberately noisy; choose the component supported by the independent health checks.",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(inputs: Mapping[str, list[Mapping[str, Any]]], truth: Iterable[Mapping[str, Any]]) -> str:
    payload = bytearray()
    for split in SLICES:
        payload.extend(split.encode())
        payload.extend(b"\n")
        payload.extend(b"\n".join(_canonical(row) for row in inputs[split]))
        payload.extend(b"\n")
    payload.extend(b"truth\n")
    payload.extend(b"\n".join(_canonical(row) for row in truth))
    return hashlib.sha256(bytes(payload)).hexdigest()


def scorer_fingerprint() -> str:
    return hashlib.sha256(f"{SCORER_VERSION}:strict incident JSON: no repair".encode()).hexdigest()


def _packet(case_id: str, split: str, family_index: int, case_index: int, topology_name: str) -> tuple[str, str, list[str]]:
    topology = TOPOLOGIES[topology_name]
    components = topology["components"]
    family = FAILURE_FAMILIES[family_index]
    spec = FAILURE_SPECS[family]
    culprit = components[spec.culprit_role]
    distractor = components["gateway"] if spec.culprit_role != "gateway" else components["app"]
    t0 = 11 + ((family_index * 5 + case_index * 7) % 40)
    start = f"14:{t0:02d}"
    signal = f"14:{(t0 + 2) % 60:02d}"
    change = f"14:{(t0 + 5) % 60:02d}"
    intro = _NARRATIVES[(family_index * 2 + case_index) % len(_NARRATIVES)]
    packet = "\n".join(
        [
            "INCIDENT",
            f"id: {case_id}",
            f"{intro} The first user-visible error appeared at {start}.",
            "",
            "TOPOLOGY",
            *topology["edges"],
            "COMPONENTS",
            *[f"{role}: {name}" for role, name in sorted(components.items())],
            "",
            "RECENT CHANGES",
            f"{change} {distractor} policy revision was observed, but the packet's primary signal predates it.",
            f"{signal} the distinctive symptom became correlated with {culprit} health.",
            "",
            "METRICS",
            f"M1 {spec.metric_label}: {spec.metric_change}",
            f"M2 {culprit} dependency signal: {spec.secondary_signal}",
            f"M3 {distractor} request rate: stable while the customer error rate rose",
            "",
            "LOGS / EVENTS",
            f"E1 {culprit}: {spec.primary_log}",
            f"E2 {distractor}: one correlated warning without a corresponding failure signature",
            "",
            "ALERTS / DEPENDENCY HEALTH",
            f"A1 {culprit}: {spec.alert_signal}",
            f"A2 unrelated dependencies: health checks remain within their normal range",
        ]
    )
    return packet, culprit, ["M1", "E1", "A1"]


def _validate_blind_truth(target: Mapping[str, Any], record: Mapping[str, Any]) -> None:
    if set(target) != {"incident_id", "culprit_service", "failure_mode", "recommended_action", "evidence_ids", "metadata"}:
        raise ValueError("blind ground truth schema mismatch")
    family = target["failure_mode"]
    if family not in FAILURE_FAMILY_SET or target["recommended_action"] not in ACTION_SET:
        raise ValueError("blind ground truth taxonomy mismatch")
    if target["recommended_action"] != FAILURE_SPECS[family].action:
        raise ValueError("blind ground truth action mismatch")
    if target["incident_id"] != record["incident_id"] or target["culprit_service"] not in record["metadata"]["present_components"]:
        raise ValueError("blind ground truth identity/component mismatch")
    evidence_ids = target["evidence_ids"]
    if not isinstance(evidence_ids, list) or len(evidence_ids) < 2 or len(evidence_ids) > 4:
        raise ValueError("blind ground truth evidence schema mismatch")
    if len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids).issubset(packet_evidence_ids(record["incident_packet"])):
        raise ValueError("blind ground truth evidence mismatch")


def generate_blind_benchmark(seed: int = 20260912) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    inputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SLICES}
    truth: list[dict[str, Any]] = []
    topology_names = tuple(TOPOLOGIES)
    case_number = 0
    for family_index, family in enumerate(FAILURE_FAMILIES):
        for case_index in range(5):
            case_number += 1
            split = _SPLIT_BY_CASE[case_index]
            topology_name = topology_names[(seed + family_index * 3 + case_index * 2) % len(topology_names)]
            case_id = f"blind-v2-{split}-{case_number:03d}"
            packet, culprit, evidence_ids = _packet(case_id, split, family_index, case_index, topology_name)
            components = TOPOLOGIES[topology_name]["components"]
            record = {
                "incident_id": case_id,
                "slice": split,
                "incident_packet": packet,
                "metadata": {
                    "difficulty": split,
                    "topology_family": topology_name,
                    "red_herring": case_index == 2 or case_index == 3,
                    "present_components": sorted(set(components.values())),
                    "evidence_ids": evidence_ids,
                },
            }
            target = {
                "incident_id": case_id,
                "culprit_service": culprit,
                "failure_mode": family,
                "recommended_action": FAILURE_SPECS[family].action,
                "evidence_ids": evidence_ids,
                "metadata": {
                    "difficulty": split,
                    "failure_family": family,
                    "topology_family": topology_name,
                    "red_herring": record["metadata"]["red_herring"],
                    "generator_version": GENERATOR_VERSION,
                },
            }
            validate_input_record(record)
            _validate_blind_truth(target, record)
            inputs[split].append(record)
            truth.append(target)
    return inputs, truth


def contamination_audit(
    inputs: Mapping[str, list[Mapping[str, Any]]],
    *source_paths: str | Path,
) -> dict[str, Any]:
    blind_packets = {record["incident_packet"] for rows in inputs.values() for record in rows}
    exact_overlap: list[str] = []
    source_counts: dict[str, int] = {}
    for source_path in source_paths:
        path = Path(source_path)
        if not path.exists():
            source_counts[str(path)] = 0
            continue
        count = 0
        for file_path in sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]:
            for line in file_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                value = json.loads(line)
                packet = value.get("incident_packet") if isinstance(value, Mapping) else None
                if packet in blind_packets:
                    exact_overlap.append(str(value.get("incident_id", "unknown")))
                    count += 1
        source_counts[str(path)] = count
    return {
        "status": "pass" if not exact_overlap else "fail",
        "exact_packet_overlap_count": len(exact_overlap),
        "overlap_ids": sorted(exact_overlap),
        "source_counts": source_counts,
    }


def write_blind_benchmark(
    output_dir: str | Path,
    *,
    seed: int = 20260912,
    contamination_sources: tuple[str | Path, ...] = (),
) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    inputs, truth = generate_blind_benchmark(seed)
    audit = contamination_audit(inputs, *contamination_sources)
    if audit["status"] != "pass":
        raise ValueError("blind benchmark contamination audit failed")
    fingerprint = _fingerprint(inputs, truth)
    for split in SLICES:
        (directory / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inputs[split]),
            encoding="utf-8",
        )
    (directory / "ground_truth.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in truth),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "scorer_version": SCORER_VERSION,
        "scorer_fingerprint": scorer_fingerprint(),
        "generation_seed": seed,
        "fingerprint": fingerprint,
        "counts": {split: len(inputs[split]) for split in SLICES},
        "contamination_audit": audit,
        "frozen": True,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
