#!/usr/bin/env python3
"""Freeze the separate held-out E06 final challenge before evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from causetune.benchmark_blind_v2 import generate_blind_benchmark, scorer_fingerprint
from causetune.benchmark_e05 import _structural_signature
from causetune.incident_benchmark import normalize_incident_text


SEED = 20260914
SOURCES = (
    "data/incident_diagnosis_training",
    "data/incident_diagnosis",
    "data/incident_diagnosis_blind_v2",
    "data/incident_diagnosis_e05",
    "data/incident_diagnosis_e06_screen",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _fingerprint(inputs: Mapping[str, list[Mapping[str, Any]]], truth: list[Mapping[str, Any]]) -> str:
    payload = bytearray()
    for split in ("standard", "hard", "transfer"):
        payload.extend(split.encode("utf-8")); payload.extend(b"\n")
        payload.extend(b"\n".join(_canonical(row) for row in inputs[split])); payload.extend(b"\n")
    payload.extend(b"truth\n"); payload.extend(b"\n".join(_canonical(row) for row in truth))
    return hashlib.sha256(bytes(payload)).hexdigest()


def _records(path: Path) -> list[dict[str, Any]]:
    files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
    rows: list[dict[str, Any]] = []
    for file_path in files:
        for line in file_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict) and isinstance(value.get("incident_packet"), str):
                    rows.append(value)
    return rows


def _audit(inputs: Mapping[str, list[Mapping[str, Any]]], sources: tuple[str, ...]) -> dict[str, Any]:
    records = [record for split in ("standard", "hard", "transfer") for record in inputs[split]]
    own_exact = {record["incident_packet"] for record in records}
    own_normalized = {normalize_incident_text(record["incident_packet"]) for record in records}
    own_structural = {_structural_signature(record["incident_packet"]) for record in records}
    source_counts: dict[str, dict[str, int]] = {}
    overlaps = {"exact": [], "normalized": [], "structural": [], "incident_id": []}
    own_ids = {record["incident_id"] for record in records}
    for source in sources:
        counts = {key: 0 for key in source_counts.get("template", {"exact": 0, "normalized": 0, "structural": 0, "incident_id": 0})}
        for row in _records(Path(source)) if Path(source).exists() else []:
            packet = row["incident_packet"]
            source_id = str(row.get("incident_id", "unknown"))
            if packet in own_exact:
                counts["exact"] += 1; overlaps["exact"].append(f"{source}:{source_id}")
            if normalize_incident_text(packet) in own_normalized:
                counts["normalized"] += 1; overlaps["normalized"].append(f"{source}:{source_id}")
            if _structural_signature(packet) in own_structural:
                counts["structural"] += 1; overlaps["structural"].append(f"{source}:{source_id}")
            if source_id in own_ids:
                counts["incident_id"] += 1; overlaps["incident_id"].append(f"{source}:{source_id}")
        source_counts[source] = counts
    exact_normalized_clean = not overlaps["exact"] and not overlaps["normalized"] and not overlaps["incident_id"]
    return {
        "status": "pass" if exact_normalized_clean else "fail",
        "structural_status": "reported_only_taxonomy_shape_reuse",
        "exact_packet_overlap_count": len(overlaps["exact"]),
        "normalized_packet_overlap_count": len(overlaps["normalized"]),
        "canonical_structural_overlap_count": len(overlaps["structural"]),
        "incident_id_overlap_count": len(overlaps["incident_id"]),
        "overlap_ids": {key: sorted(value) for key, value in overlaps.items()},
        "source_counts": source_counts,
        "structural_signature_interpretation": "taxonomy-aligned topology shape reuse is disclosed and is not treated as packet contamination",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/incident_diagnosis_e06_final")
    parser.add_argument("--protocol-output", default="results/experiment_06/final_protocol.json")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite frozen E06 final benchmark: {output}")
    inputs, truth = generate_blind_benchmark(seed=SEED)
    # The shared blind generator uses a historical ``blind-v2`` identifier
    # namespace. Give this separately frozen E06 challenge its own identity
    # namespace before auditing or persisting it; the packet content and truth
    # values remain exactly those generated by the new E06 seed.
    id_map = {}
    for record in [row for split in ("standard", "hard", "transfer") for row in inputs[split]]:
        old_id = record["incident_id"]
        new_id = old_id.replace("blind-v2-", "e06-final-")
        id_map[old_id] = new_id
        record["incident_id"] = new_id
        record["incident_packet"] = record["incident_packet"].replace(f"id: {old_id}", f"id: {new_id}", 1)
    for target in truth:
        target["incident_id"] = id_map[target["incident_id"]]
    audit = _audit(inputs, SOURCES)
    if audit["status"] != "pass":
        raise ValueError(f"E06 final benchmark contamination audit failed: {audit}")
    fingerprint = _fingerprint(inputs, truth)
    output.mkdir(parents=True, exist_ok=False)
    for split in ("standard", "hard", "transfer"):
        (output / f"{split}.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inputs[split]), encoding="utf-8"
        )
    (output / "ground_truth.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in truth), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "experiment": "E06",
        "benchmark_version": "incident-e06-final-blind-v1",
        "generator_version": "independent-blind-namespace-v1",
        "generation_seed": SEED,
        "fingerprint": fingerprint,
        "counts": {split: len(inputs[split]) for split in ("standard", "hard", "transfer")},
        "total_cases": len(truth),
        "taxonomy_reused": True,
        "final_evidence_separate_from_capability_screen": True,
        "scorer_version": "incident-scorer-v1",
        "scorer_fingerprint": scorer_fingerprint(),
        "contamination_sources": list(SOURCES),
        "contamination_audit": audit,
        "frozen": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "experiment": "E06",
        "protocol_version": "e06-one-shot-final-blind-v1",
        "benchmark_dir": str(output),
        "benchmark_fingerprint": fingerprint,
        "benchmark_manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "scorer_version": "incident-scorer-v1",
        "scorer_fingerprint": scorer_fingerprint(),
        "prompt_path": "configs/incident_diagnosis_eval.json",
        "prompt_sha256": hashlib.sha256(Path("configs/incident_diagnosis_eval.json").read_bytes()).hexdigest(),
        "decoding": {"max_new_tokens": 96, "batch_size": 4, "do_sample": False},
        "inputs_frozen": True,
        "truths_frozen": True,
        "one_shot_per_system": True,
        "fresh_reload_per_system": True,
        "raw_predictions_required": True,
        "offline_reproduction_required": True,
        "capability_screen_used_as_final_evidence": False,
        "systems": [
            {"name": "base", "model_id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083", "adapter": None, "trust_remote_code": False},
            {"name": "tuned", "model_id": "microsoft/Phi-4-mini-instruct", "revision": "cfbefacb99257ffa30c83adab238a50856ac3083", "adapter": "runs/experiment_06/training/runner/checkpoint-step-000125", "trust_remote_code": False},
        ],
    }
    protocol_path = Path(args.protocol_output)
    protocol_path.parent.mkdir(parents=True, exist_ok=True)
    protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    contamination_path = protocol_path.parent / "final_contamination_report.json"
    contamination_path.write_text(json.dumps({"schema_version": 1, "benchmark_fingerprint": fingerprint, **audit}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "benchmark_fingerprint": fingerprint, "total_cases": len(truth), "contamination": audit}, sort_keys=True))


if __name__ == "__main__":
    main()
