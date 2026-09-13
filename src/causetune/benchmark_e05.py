"""Fresh held-out Experiment 05 challenge generation and contamination audit."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .incident_benchmark import normalize_incident_text, packet_evidence_ids
from .incident_taxonomy import ACTION_SET, FAILURE_FAMILIES, FAILURE_FAMILY_SET, FAILURE_SPECS, SLICES


BENCHMARK_VERSION = "incident-e05-blind-v1"
GENERATOR_VERSION = "e05-independent-generator-namespace-v1"
STATIC_VERSION = "e05-static-authored-fixtures-v1"
SCORER_VERSION = "incident-scorer-v1"
DEFAULT_SEED = 20260913
_REQUIRED_SECTIONS = ("INCIDENT", "TOPOLOGY", "RECENT CHANGES", "METRICS", "LOGS / EVENTS", "ALERTS / DEPENDENCY HEALTH")
_FAMILY_ACTIONS = frozenset((*FAILURE_FAMILY_SET, *ACTION_SET))


def _topology(prefix: str, names: tuple[str, ...], edges: tuple[tuple[int, int], ...]) -> dict[str, Any]:
    roles = ("gateway", "app", "auth", "worker", "consumer", "db", "cache", "external", "queue", "dns", "cert")
    components = dict(zip(roles, names, strict=True))
    return {"components": components, "edges": tuple(f"{names[left]} -> {names[right]}" for left, right in edges), "namespace": prefix}


_TOPOLOGIES: dict[str, dict[str, Any]] = {
    "ledger_lattice": _topology("ledger", ("ledger-edge", "settlement-orchestrator", "risk-oracle", "posting-worker", "ledger-consumer", "ledger-store", "ledger-hotset", "clearing-house", "posting-bus", "ledger-resolver", "clearing-house"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9))),
    "research_grid": _topology("research", ("research-front", "experiment-api", "policy-oracle", "batch-runner", "result-consumer", "result-store", "result-cache", "model-registry", "job-stream", "research-dns", "model-registry"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (2, 7))),
    "fleet_control": _topology("fleet", ("fleet-edge", "dispatch-service", "route-authority", "vehicle-worker", "telemetry-consumer", "fleet-store", "fleet-cache", "map-provider", "dispatch-bus", "fleet-resolver", "map-provider"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 5))),
    "care_exchange": _topology("care", ("care-front", "case-service", "policy-checker", "care-worker", "audit-consumer", "case-store", "case-cache", "records-partner", "case-events", "care-resolver", "records-partner"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 5), (3, 5))),
    "media_fabric": _topology("media", ("studio-edge", "render-api", "rights-service", "render-worker", "asset-consumer", "asset-store", "preview-cache", "transcode-partner", "render-bus", "studio-resolver", "transcode-partner"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 7))),
    "supply_chain": _topology("supply", ("supply-edge", "allocation-service", "vendor-policy", "warehouse-worker", "stock-consumer", "inventory-store", "inventory-cache", "carrier-network", "stock-bus", "supply-resolver", "carrier-network"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 5), (3, 8), (5, 6))),
    "energy_mesh": _topology("energy", ("grid-edge", "dispatch-core", "safety-policy", "meter-worker", "meter-consumer", "grid-store", "grid-cache", "market-link", "meter-bus", "grid-resolver", "market-link"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 5), (3, 5), (5, 6), (6, 1))),
    "learning_exchange": _topology("learning", ("classroom-edge", "lesson-service", "roster-policy", "grading-worker", "progress-consumer", "lesson-store", "lesson-cache", "content-partner", "progress-bus", "classroom-resolver", "content-partner"), ((0, 1), (1, 2), (1, 5), (1, 6), (1, 7), (3, 8), (4, 8), (4, 5), (0, 9), (7, 9), (2, 5), (3, 5), (5, 6), (6, 1), (4, 7))),
}
_TOPOLOGY_NAMES = tuple(_TOPOLOGIES)

_GENERATED_NARRATIVES = (
    "The incident-room handoff lists several plausible symptoms; the first independent signal is the deciding observation.",
    "The operator recorded the packet in a different order from the alert feed, with the dependency evidence preserved verbatim.",
    "A routine change overlaps the outage window, but the surrounding signals distinguish coincidence from the failing path.",
    "The service names are unfamiliar and the timeline is compressed; compare the local and remote health observations.",
    "The customer report is broad while one internal signal is narrow and specific; the narrow signal should carry more weight.",
    "The packet includes a recovered warning, a partial metric, and a persistent dependency signal.",
)
_STATIC_NARRATIVES = (
    "Static fixture: an incident coordinator copied this short evidence bundle from a separate notation style.",
    "Static fixture: the attached observations are ordered by an operator after the event and include an explicit negative check.",
)
_CHRONOLOGIES = ("causal_first", "causal_after", "coincident_change", "recovered_distractor", "negative_distractor")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def benchmark_fingerprint(inputs: Mapping[str, list[Mapping[str, Any]]], truth: Iterable[Mapping[str, Any]]) -> str:
    payload = bytearray()
    for split in SLICES:
        payload.extend(split.encode("utf-8"))
        payload.extend(b"\n")
        payload.extend(b"\n".join(_canonical(row) for row in inputs[split]))
        payload.extend(b"\n")
    payload.extend(b"truth\n")
    payload.extend(b"\n".join(_canonical(row) for row in truth))
    return hashlib.sha256(bytes(payload)).hexdigest()


def scorer_fingerprint() -> str:
    return hashlib.sha256(f"{SCORER_VERSION}:strict incident JSON: no repair".encode("utf-8")).hexdigest()


def _structural_signature(packet: str) -> str:
    topology = packet.split("TOPOLOGY\n", 1)[1].split("\nRECENT CHANGES\n", 1)[0]
    edges = re.findall(r"^([^\n]+?) -> ([^\n]+)$", topology, re.MULTILINE)
    nodes = {node for edge in edges for node in edge}
    degrees = Counter(node for edge in edges for node in edge)
    shape = {
        "edge_count": len(edges),
        "node_count": len(nodes),
        "degree_sequence": sorted(degrees.values()),
        "metric_slots": len(re.findall(r"^M\d+ ", packet, re.MULTILINE)),
        "event_slots": len(re.findall(r"^E\d+ ", packet, re.MULTILINE)),
        "alert_slots": len(re.findall(r"^A\d+ ", packet, re.MULTILINE)),
    }
    return hashlib.sha256(_canonical(shape)).hexdigest()


def _validate_input(record: Mapping[str, Any]) -> None:
    if set(record) != {"incident_id", "slice", "incident_packet", "metadata"}:
        raise ValueError("E05 input schema mismatch")
    if record["slice"] not in SLICES:
        raise ValueError("E05 input slice mismatch")
    packet = record["incident_packet"]
    if not isinstance(packet, str) or any(section not in packet for section in _REQUIRED_SECTIONS):
        raise ValueError(f"{record['incident_id']}: E05 packet sections incomplete")
    metadata = record["metadata"]
    if set(metadata) != {"difficulty", "topology_family", "red_herring", "present_components", "evidence_ids"}:
        raise ValueError("E05 metadata schema mismatch")
    if metadata["difficulty"] != record["slice"] or metadata["topology_family"] not in _TOPOLOGIES:
        raise ValueError(f"{record['incident_id']}: E05 metadata mismatch")
    if not metadata["present_components"] or not isinstance(metadata["red_herring"], bool):
        raise ValueError(f"{record['incident_id']}: E05 metadata values invalid")
    evidence = metadata["evidence_ids"]
    packet_ids = packet_evidence_ids(packet)
    if not isinstance(evidence, list) or len(evidence) != 3 or len(set(evidence)) != 3 or not set(evidence).issubset(packet_ids):
        raise ValueError(f"{record['incident_id']}: E05 evidence IDs invalid")
    lowered = packet.lower()
    leaked = [label for label in _FAMILY_ACTIONS if label in lowered or label.replace("_", " ") in lowered]
    if leaked:
        raise ValueError(f"{record['incident_id']}: E05 literal taxonomy/action leakage: {leaked}")


def _validate_truth(target: Mapping[str, Any], record: Mapping[str, Any], origin: str) -> None:
    if set(target) != {"incident_id", "culprit_service", "failure_mode", "recommended_action", "evidence_ids", "metadata"}:
        raise ValueError("E05 truth schema mismatch")
    family = target["failure_mode"]
    if family not in FAILURE_FAMILY_SET or target["recommended_action"] not in ACTION_SET or FAILURE_SPECS[family].action != target["recommended_action"]:
        raise ValueError("E05 truth taxonomy mismatch")
    if target["incident_id"] != record["incident_id"] or target["culprit_service"] not in record["metadata"]["present_components"]:
        raise ValueError("E05 truth identity mismatch")
    if target["evidence_ids"] != record["metadata"]["evidence_ids"]:
        raise ValueError("E05 truth evidence mismatch")
    metadata = target["metadata"]
    if metadata != {
        "difficulty": record["slice"],
        "failure_family": family,
        "topology_family": record["metadata"]["topology_family"],
        "red_herring": record["metadata"]["red_herring"],
        "generator_version": GENERATOR_VERSION if origin == "generated" else STATIC_VERSION,
    }:
        raise ValueError("E05 truth metadata mismatch")


def _times(family_index: int, case_index: int, seed: int) -> tuple[str, str, str, str]:
    minute = 5 + ((seed + family_index * 11 + case_index * 13) % 48)
    return f"16:{minute:02d}", f"16:{(minute + 2) % 60:02d}", f"16:{(minute + 4) % 60:02d}", f"16:{(minute + 9) % 60:02d}"


def _render_packet(case_id: str, split: str, family_index: int, case_index: int, topology_name: str, seed: int, origin: str) -> tuple[str, str, list[str], str]:
    topology = _TOPOLOGIES[topology_name]
    components = topology["components"]
    family = FAILURE_FAMILIES[family_index]
    spec = FAILURE_SPECS[family]
    culprit = components[spec.culprit_role]
    distractors = [components[role] for role in ("gateway", "app", "db", "external") if role != spec.culprit_role]
    start, signal, change, end = _times(family_index, case_index, seed)
    chronology = _CHRONOLOGIES[(family_index + case_index + seed) % len(_CHRONOLOGIES)]
    intro_pool = _GENERATED_NARRATIVES if origin == "generated" else _STATIC_NARRATIVES
    intro = intro_pool[(family_index + case_index) % len(intro_pool)]
    signal_slot = (family_index * 3 + case_index + seed) % 4 + 1
    evidence_ids = [f"M{signal_slot}", f"E{signal_slot}", f"A{signal_slot}"]
    if chronology == "causal_after":
        change_line = f"{change} {culprit} configuration became active; the distinctive signal followed at {signal}."
    elif chronology == "coincident_change":
        change_line = f"{change} {distractors[0]} routine revision and the first signal were recorded in the same window; independent checks separate them."
    elif chronology == "recovered_distractor":
        change_line = f"{change} {distractors[0]} warning recovered before the customer symptom ended; the {culprit} signal persisted at {signal}."
    elif chronology == "negative_distractor":
        change_line = f"{change} {distractors[0]} emitted a correlated alert, but its dependency check was normal before the primary signal at {signal}."
    else:
        change_line = f"{change} {distractors[0]} routine revision was recorded after the distinctive {culprit} signal at {signal}."
    metric_lines = [f"M{signal_slot} {spec.metric_label}: {spec.metric_change}"]
    for slot, distractor in enumerate(distractors, 1):
        if slot != signal_slot:
            metric_lines.append(f"M{slot} {distractor} request or queue signal: stable baseline with no matching error increase")
    event_lines = [f"E{signal_slot} {culprit}: {spec.primary_log}"]
    alert_lines = [f"A{signal_slot} {culprit}: {spec.alert_signal}"]
    for slot, distractor in enumerate(distractors, 1):
        if slot == signal_slot:
            continue
        event_lines.append(f"E{slot} {distractor}: a correlated observation without the distinctive failure signature")
        alert_lines.append(f"A{slot} {distractor}: dependency checks remain within the normal range")
    packet = "\n".join([
        "INCIDENT", f"id: {case_id}", f"{intro} The reported window is {start} to {end}.",
        "", "TOPOLOGY", *topology["edges"], "COMPONENTS", *[f"{role}: {name}" for role, name in sorted(components.items())],
        "", "RECENT CHANGES", change_line, f"{end} incident declared after the evidence interval.",
        "", "METRICS", *metric_lines,
        "", "LOGS / EVENTS", *event_lines,
        "", "ALERTS / DEPENDENCY HEALTH", *alert_lines,
    ])
    return packet, culprit, evidence_ids, chronology


def generate_e05_benchmark(seed: int = DEFAULT_SEED) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    inputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SLICES}
    truth: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    ordered = [(family_index, case_index) for case_index in range(10) for family_index in range(len(FAILURE_FAMILIES))]
    for family_index, case_index in ordered:
        origin = "generated" if case_index < 8 else "static"
        split = ("standard", "standard", "standard", "standard", "hard", "hard", "transfer", "transfer", "standard", "hard")[case_index]
        topology_name = _TOPOLOGY_NAMES[(seed + family_index * 5 + case_index * 7) % len(_TOPOLOGY_NAMES)]
        case_number = len(provenance) + 1
        case_id = f"e05-{origin}-{case_number:03d}"
        packet, culprit, evidence_ids, chronology = _render_packet(case_id, split, family_index, case_index, topology_name, seed, origin)
        topology = _TOPOLOGIES[topology_name]
        components = topology["components"]
        record = {
            "incident_id": case_id,
            "slice": split,
            "incident_packet": packet,
            "metadata": {
                "difficulty": split,
                "topology_family": topology_name,
                "red_herring": split == "hard" or chronology in {"coincident_change", "recovered_distractor", "negative_distractor"},
                "present_components": sorted(set(components.values())),
                "evidence_ids": evidence_ids,
            },
        }
        family = FAILURE_FAMILIES[family_index]
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
                "generator_version": GENERATOR_VERSION if origin == "generated" else STATIC_VERSION,
            },
        }
        _validate_input(record)
        _validate_truth(target, record, origin)
        inputs[split].append(record)
        truth.append(target)
        provenance.append({"incident_id": case_id, "origin": origin, "generator_version": target["metadata"]["generator_version"], "topology_family": topology_name, "chronology": chronology, "structural_signature": _structural_signature(packet)})
    return inputs, truth, provenance


def contamination_audit(inputs: Mapping[str, list[Mapping[str, Any]]], *source_paths: str | Path) -> dict[str, Any]:
    records = [record for split in SLICES for record in inputs[split]]
    own_exact = {record["incident_packet"] for record in records}
    own_normalized = {normalize_incident_text(record["incident_packet"]) for record in records}
    own_structural = {_structural_signature(record["incident_packet"]) for record in records}
    source_counts: dict[str, dict[str, int]] = {}
    exact: list[str] = []
    normalized: list[str] = []
    structural: list[str] = []
    id_overlap: list[str] = []
    for source_path in source_paths:
        path = Path(source_path)
        counts = {"exact": 0, "normalized": 0, "structural": 0, "incident_id": 0}
        if path.exists():
            files = sorted(path.rglob("*.jsonl")) if path.is_dir() else [path]
            for file_path in files:
                for line in file_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    packet = value.get("incident_packet") if isinstance(value, Mapping) else None
                    if not isinstance(packet, str):
                        continue
                    source_id = str(value.get("incident_id", "unknown"))
                    if packet in own_exact:
                        counts["exact"] += 1; exact.append(f"{path}:{source_id}")
                    if normalize_incident_text(packet) in own_normalized:
                        counts["normalized"] += 1; normalized.append(f"{path}:{source_id}")
                    if _structural_signature(packet) in own_structural:
                        counts["structural"] += 1; structural.append(f"{path}:{source_id}")
                    if source_id in {record["incident_id"] for record in records}:
                        counts["incident_id"] += 1; id_overlap.append(f"{path}:{source_id}")
        source_counts[str(path)] = counts
    return {
        "status": "pass" if not any((exact, normalized, structural, id_overlap)) else "fail",
        "exact_packet_overlap_count": len(exact),
        "normalized_packet_overlap_count": len(normalized),
        "canonical_structural_overlap_count": len(structural),
        "incident_id_overlap_count": len(id_overlap),
        "overlap_ids": {"exact": sorted(exact), "normalized": sorted(normalized), "structural": sorted(structural), "incident_id": sorted(id_overlap)},
        "source_counts": source_counts,
    }


def write_e05_benchmark(output_dir: str | Path, *, seed: int = DEFAULT_SEED, contamination_sources: tuple[str | Path, ...] = ()) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    inputs, truth, provenance = generate_e05_benchmark(seed)
    audit = contamination_audit(inputs, *contamination_sources)
    if audit["status"] != "pass":
        raise ValueError(f"E05 contamination audit failed: {audit}")
    fingerprint = benchmark_fingerprint(inputs, truth)
    for split in SLICES:
        (directory / f"{split}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in inputs[split]), encoding="utf-8")
    (directory / "ground_truth.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in truth), encoding="utf-8")
    (directory / "case_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "static_fixture_version": STATIC_VERSION,
        "scorer_version": SCORER_VERSION,
        "scorer_fingerprint": scorer_fingerprint(),
        "generation_seed": seed,
        "fingerprint": fingerprint,
        "counts": {split: len(inputs[split]) for split in SLICES},
        "total_cases": len(truth),
        "origin_counts": dict(Counter(row["origin"] for row in provenance)),
        "taxonomy_reused": True,
        "topology_namespace": "e05-custom-compositions-v1",
        "contamination_sources": [str(path) for path in contamination_sources],
        "contamination_audit": audit,
        "frozen": True,
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return manifest
