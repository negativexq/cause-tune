#!/usr/bin/env python3
"""Freeze the mixed sufficient/insufficient Experiment 07 boundary challenge."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

from causetune.evidence import sha256_path
from causetune.incident_benchmark import benchmark_fingerprint, generate_benchmark, normalize_incident_text, packet_evidence_ids
from causetune.incident_taxonomy import FAILURE_SPECS, SLICES


SEED = 20260915
TOTAL = 60
CASES = (
    ("sufficient", 12),
    ("insufficient_evidence", 12),
    ("contradictory_evidence", 8),
    ("multiple_plausible_culprits", 8),
    ("missing_topology", 6),
    ("missing_metrics", 6),
    ("out_of_taxonomy", 8),
)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _rename_packet(packet: str, old_id: str, new_id: str) -> str:
    return packet.replace(f"id: {old_id}", f"id: {new_id}")


def _mutate(packet: str, kind: str) -> str:
    if kind == "sufficient":
        return packet
    if kind == "insufficient_evidence":
        lines = [line for line in packet.splitlines() if not re.match(r"^(M1|E1|A1) ", line)]
        lines.insert(lines.index("METRICS") + 1, "MISSING: the primary component measurement was not collected")
        return "\n".join(lines)
    if kind == "contradictory_evidence":
        marker = "MISSING: independent probes disagree about the first causal signal"
        lines = packet.splitlines()
        lines.insert(lines.index("LOGS / EVENTS"), marker)
        lines.insert(lines.index("LOGS / EVENTS") + 2, "E9 an independent trace attributes the same symptom to a different component")
        return "\n".join(lines)
    if kind == "multiple_plausible_culprits":
        lines = packet.splitlines()
        lines.insert(lines.index("ALERTS / DEPENDENCY HEALTH") + 1, "A9 a second component shows a matching primary-looking alert during the same window")
        return "\n".join(lines)
    if kind == "missing_topology":
        lines = packet.splitlines()
        start = lines.index("TOPOLOGY") + 1
        end = lines.index("RECENT CHANGES")
        kept = [line for line in lines[start:end] if line.startswith("components:")]
        return "\n".join(lines[:start] + kept + lines[end:])
    if kind == "missing_metrics":
        lines = packet.splitlines()
        start = lines.index("METRICS") + 1
        end = lines.index("LOGS / EVENTS")
        return "\n".join(lines[:start] + ["MISSING: metric collection was unavailable during the incident window"] + lines[end:])
    if kind == "out_of_taxonomy":
        lines = packet.splitlines()
        replacements = {
            "M1 ": "M1 unsupported protocol anomaly: client negotiation fails",
            "E1 ": "E1 unsupported protocol anomaly: the failure is outside the evaluation taxonomy",
            "A1 ": "A1 unsupported protocol anomaly: no allowed failure family matches",
        }
        output = []
        for line in lines:
            replaced = line
            for prefix, value in replacements.items():
                if line.startswith(prefix):
                    replaced = value
                    break
            output.append(replaced)
        return "\n".join(output)
    raise ValueError(kind)


def _canonical_hash(rows: Iterable[Mapping[str, Any]]) -> str:
    payload = "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def _source_records() -> list[dict[str, Any]]:
    inputs, truths = generate_benchmark(SEED)
    truth_by_id = {row["incident_id"]: row for row in truths}
    return [dict(record, _truth=truth_by_id[record["incident_id"]]) for split in SLICES for record in inputs[split]]


def _build() -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    source = _source_records()
    selected = source[:TOTAL]
    inputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SLICES}
    truths: list[dict[str, Any]] = []
    cursor = 0
    for kind, count in CASES:
        for index in range(count):
            source_record = selected[cursor]
            cursor += 1
            old_id = source_record["incident_id"]
            new_id = f"e07-{kind}-{index + 1:03d}"
            truth = source_record["_truth"]
            packet = _rename_packet(_mutate(source_record["incident_packet"], kind), old_id, new_id)
            available = sorted(packet_evidence_ids(packet))
            difficulty = source_record["slice"]
            input_record = {
                "incident_id": new_id,
                "slice": difficulty,
                "incident_packet": packet,
                "metadata": {
                    "difficulty": difficulty,
                    "topology_family": source_record["metadata"]["topology_family"],
                    "red_herring": source_record["metadata"]["red_herring"],
                    "present_components": source_record["metadata"]["present_components"],
                    "evidence_ids": available,
                    "boundary_category": kind,
                },
            }
            expected_status = "diagnose" if kind == "sufficient" else ("out_of_taxonomy" if kind == "out_of_taxonomy" else "insufficient_evidence" if kind in {"insufficient_evidence", "missing_topology", "missing_metrics"} else "ambiguous_evidence")
            truth_record = {
                "incident_id": new_id,
                "expected_status": expected_status,
                "culprit_service": truth["culprit_service"] if kind == "sufficient" else None,
                "failure_mode": truth["failure_mode"] if kind == "sufficient" else None,
                "recommended_action": truth["recommended_action"] if kind == "sufficient" else None,
                "evidence_ids": truth["evidence_ids"] if kind == "sufficient" else [],
                "metadata": {
                    "boundary_category": kind,
                    "difficulty": difficulty,
                    "underlying_failure_family": truth["failure_mode"],
                    "topology_family": source_record["metadata"]["topology_family"],
                    "generator_version": "experiment-07-boundary-v1",
                },
            }
            inputs[difficulty].append(input_record)
            truths.append(truth_record)
    for split in SLICES:
        inputs[split].sort(key=lambda row: row["incident_id"])
    truths.sort(key=lambda row: row["incident_id"])
    return inputs, truths, {"source_seed": SEED, "case_counts": dict(CASES)}


def _contamination(inputs: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    current = [row for split in SLICES for row in inputs[split]]
    sources = {
        "training": [*(_read_jsonl(Path("data/incident_diagnosis_training/train.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis_training/validation.jsonl")))],
        "e02": [*(_read_jsonl(Path("data/incident_diagnosis/standard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis/hard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis/transfer.jsonl")))],
        "e05": [*(_read_jsonl(Path("data/incident_diagnosis_e05/standard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis_e05/hard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis_e05/transfer.jsonl")))],
        "e06_final": [*(_read_jsonl(Path("data/incident_diagnosis_e06_final/standard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis_e06_final/hard.jsonl"))), *(_read_jsonl(Path("data/incident_diagnosis_e06_final/transfer.jsonl")))],
    }
    exact = {}
    normalized = {}
    for name, rows in sources.items():
        exact[name] = sum(any(row["incident_packet"] == other["incident_packet"] for other in rows) for row in current)
        normalized[name] = sum(any(normalize_incident_text(row["incident_packet"]) == normalize_incident_text(other["incident_packet"]) for other in rows) for row in current)
    return {"exact_packet_duplicates": exact, "normalized_packet_duplicates": normalized, "status": "PASS" if not any(exact.values()) else "FAIL"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data/incident_diagnosis_e07_boundary")
    parser.add_argument("--protocol", default="results/experiment_07/protocol.json")
    args = parser.parse_args()
    output = Path(args.output_dir)
    protocol_path = Path(args.protocol)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite frozen E07 directory: {output}")
    inputs, truths, provenance = _build()
    output.mkdir(parents=True, exist_ok=False)
    for split in SLICES:
        _write_jsonl(output / f"{split}.jsonl", inputs[split])
    _write_jsonl(output / "ground_truth.jsonl", truths)
    fingerprint = benchmark_fingerprint(inputs, truths)
    contamination = _contamination(inputs)
    manifest = {
        "schema_version": 1,
        "experiment": "E07",
        "benchmark_version": "experiment-07-boundary-v1",
        "generator_version": "experiment-07-boundary-v1",
        "generation_seed": SEED,
        "case_count": TOTAL,
        "case_counts": provenance["case_counts"],
        "slice_counts": {split: len(inputs[split]) for split in SLICES},
        "fingerprint": fingerprint,
        "contamination": contamination,
        "ground_truth_file": "ground_truth.jsonl",
        "inputs_frozen": True,
        "truths_frozen": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    prompt = Path("configs/e07_boundary_prompt.txt")
    protocol = {
        "schema_version": 1,
        "experiment": "E07",
        "question": "Does specialization cause confident incorrect diagnoses when evidence is insufficient or ambiguous?",
        "benchmark_dir": str(output),
        "benchmark_fingerprint": fingerprint,
        "case_count": TOTAL,
        "inputs_frozen": True,
        "truths_frozen": True,
        "prompt_path": str(prompt),
        "prompt_sha256": sha256_path(prompt),
        "decoding": {"max_new_tokens": 128, "batch_size": 4, "do_sample": False},
        "systems": ["base", "e02", "e04"],
        "abstention_semantics": {
            "diagnose": "sufficient evidence for one defensible taxonomy diagnosis",
            "insufficient_evidence": "required causal evidence is missing",
            "ambiguous_evidence": "evidence supports multiple plausible or contradictory explanations",
            "out_of_taxonomy": "evidence does not map to the frozen failure-family taxonomy",
        },
        "selection_exclusion": "evaluation-only; excluded from all training and checkpoint decisions",
        "scorer": "deterministic E07 boundary scorer, no LLM judge",
        "contamination_audit": contamination,
        "benchmark_hash": sha256_path(output),
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": fingerprint, "contamination": contamination}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
