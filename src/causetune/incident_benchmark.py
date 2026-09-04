"""Deterministic incident-packet generation, validation, and fingerprints."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from .incident_taxonomy import (
    ACTION_SET,
    ACTIONS,
    FAILURE_FAMILIES,
    FAILURE_FAMILY_SET,
    FAILURE_SPECS,
    SLICES,
    SLICE_COUNTS,
    TOPOLOGIES,
    validate_taxonomy,
)


BENCHMARK_VERSION = "incident-diagnosis-v1"
GENERATOR_VERSION = "incident-diagnosis-local-v1"
DEFAULT_SEED = 20260904
REQUIRED_PACKET_SECTIONS = ("INCIDENT", "TOPOLOGY", "RECENT CHANGES", "METRICS", "LOGS / EVENTS")
INPUT_KEYS = {"incident_id", "slice", "incident_packet", "metadata"}
GROUND_TRUTH_KEYS = {
    "incident_id",
    "culprit_service",
    "failure_mode",
    "recommended_action",
    "evidence_ids",
    "metadata",
}

_EVIDENCE_ID_RE = re.compile(r"\b(?:M|E|A|D)\d+\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|ms|s|GiB|k|K)?\b", re.IGNORECASE)

_INTRODUCTIONS = (
    "At {start}, an operator reports a broad customer-facing degradation.",
    "The on-call note says requests began failing around {start}; the packet captures the next few minutes.",
    "Users describe a slow, then failing flow. Telemetry was sampled between {start} and {end}.",
    "Incident channel summary: several alerts fired close together, so timestamps matter more than alert severity.",
    "The first symptom reached the edge before the underlying component was isolated.",
    "A routine change and a noisy alert overlap in time in this incident.",
    "The operator has already retried once; compare the dependency signals before choosing a culprit.",
    "An unrelated component is mentioned to test whether correlation is mistaken for cause.",
    "The report is terse, but the timeline and health checks contain the diagnosis.",
    "This packet compresses a support-style incident exchange into one timeline.",
    "The incident commander asks for the primary component, not every symptom.",
    "Several services look unhealthy; identify the earliest causal signal.",
)

_TRANSFER_INTROS = (
    "The transfer packet is a short incident-room transcript compressed into telemetry.",
    "An operator first noticed customer symptoms, then attached the following platform evidence.",
    "The service owner reports a narrow time window with unrelated noise nearby.",
    "Read this packet as a handoff between the alerting and platform teams.",
    "The incident summary lists effects before causes, as often happens during an outage.",
    "A terse escalation includes one healthy dependency that should eliminate a tempting guess.",
)


def _canonical_json_lines(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "\n".join(
        json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for row in rows
    ).encode("utf-8")


def benchmark_fingerprint(
    inputs: Mapping[str, list[Mapping[str, Any]]],
    ground_truth: Iterable[Mapping[str, Any]],
) -> str:
    """Hash canonical input and ground-truth content, excluding the manifest."""

    payload = bytearray()
    for split in SLICES:
        payload.extend(split.encode("utf-8"))
        payload.extend(b"\n")
        payload.extend(_canonical_json_lines(inputs[split]))
        payload.extend(b"\n")
    payload.extend(b"ground_truth\n")
    payload.extend(_canonical_json_lines(ground_truth))
    return hashlib.sha256(bytes(payload)).hexdigest()


def normalize_incident_text(text: str) -> str:
    """Normalize variable IDs, measurements, punctuation, and whitespace."""

    normalized = text.lower()
    normalized = re.sub(r"\bincident[-_ ]?\d+\b", "<incident>", normalized)
    normalized = re.sub(r"\b(?:v|build|release)[-_]?\d+\b", "<version>", normalized)
    normalized = _NUMBER_RE.sub("<number>", normalized)
    normalized = re.sub(r"[^a-z0-9<>]+", " ", normalized)
    return " ".join(normalized.split())


def packet_evidence_ids(packet: str) -> set[str]:
    return set(_EVIDENCE_ID_RE.findall(packet))


def _literal_label_leakage(packet: str) -> list[str]:
    lowered = packet.lower()
    leaked: list[str] = []
    for label in (*FAILURE_FAMILIES, *ACTIONS):
        variants = (label, label.replace("_", " "))
        if any(variant in lowered for variant in variants):
            leaked.append(label)
    return leaked


def validate_input_record(record: Mapping[str, Any]) -> dict[str, Any]:
    if set(record) != INPUT_KEYS:
        raise ValueError(f"incident input keys must be {sorted(INPUT_KEYS)}")
    incident_id = record.get("incident_id")
    if not isinstance(incident_id, str) or not incident_id.strip():
        raise ValueError("incident_id must be a non-empty string")
    split = record.get("slice")
    if split not in SLICES:
        raise ValueError(f"{incident_id}: unknown benchmark slice {split!r}")
    packet = record.get("incident_packet")
    if not isinstance(packet, str) or not packet.strip():
        raise ValueError(f"{incident_id}: incident_packet must be non-empty")
    missing = [section for section in REQUIRED_PACKET_SECTIONS if section not in packet]
    if missing:
        raise ValueError(f"{incident_id}: packet missing sections {missing}")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{incident_id}: metadata must be an object")
    required_metadata = {"difficulty", "topology_family", "red_herring", "present_components", "evidence_ids"}
    if set(metadata) != required_metadata:
        raise ValueError(f"{incident_id}: input metadata keys mismatch")
    if metadata["difficulty"] != split:
        raise ValueError(f"{incident_id}: difficulty/slice mismatch")
    if not isinstance(metadata["topology_family"], str) or metadata["topology_family"] not in TOPOLOGIES:
        raise ValueError(f"{incident_id}: unknown topology family")
    components = metadata["present_components"]
    if not isinstance(components, list) or not components or not all(isinstance(item, str) for item in components):
        raise ValueError(f"{incident_id}: invalid present_components")
    packet_ids = packet_evidence_ids(packet)
    evidence_ids = metadata["evidence_ids"]
    if not isinstance(evidence_ids, list) or not 2 <= len(evidence_ids) <= 4:
        raise ValueError(f"{incident_id}: evidence_ids must contain 2-4 IDs")
    if len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids).issubset(packet_ids):
        raise ValueError(f"{incident_id}: invalid input evidence IDs")
    if not isinstance(metadata["red_herring"], bool):
        raise ValueError(f"{incident_id}: red_herring must be boolean")
    leaked = _literal_label_leakage(packet)
    if leaked:
        raise ValueError(f"{incident_id}: literal taxonomy/action leakage: {leaked}")
    return dict(record)


def validate_ground_truth_record(
    record: Mapping[str, Any],
    input_record: Mapping[str, Any],
) -> dict[str, Any]:
    if set(record) != GROUND_TRUTH_KEYS:
        raise ValueError(f"ground truth keys must be {sorted(GROUND_TRUTH_KEYS)}")
    incident_id = record.get("incident_id")
    if incident_id != input_record.get("incident_id"):
        raise ValueError(f"ground truth incident ID mismatch: {incident_id!r}")
    family = record.get("failure_mode")
    if family not in FAILURE_FAMILY_SET:
        raise ValueError(f"{incident_id}: unknown failure family {family!r}")
    action = record.get("recommended_action")
    if action not in ACTION_SET:
        raise ValueError(f"{incident_id}: unknown action {action!r}")
    if action != FAILURE_SPECS[family].action:
        raise ValueError(f"{incident_id}: action does not match frozen family mapping")
    components = input_record["metadata"]["present_components"]
    culprit = record.get("culprit_service")
    if not isinstance(culprit, str) or culprit not in components:
        raise ValueError(f"{incident_id}: culprit is not an explicitly present component")
    evidence_ids = record.get("evidence_ids")
    packet_ids = packet_evidence_ids(input_record["incident_packet"])
    if not isinstance(evidence_ids, list) or not 2 <= len(evidence_ids) <= 4:
        raise ValueError(f"{incident_id}: ground-truth evidence_ids must contain 2-4 IDs")
    if len(evidence_ids) != len(set(evidence_ids)) or not set(evidence_ids).issubset(packet_ids):
        raise ValueError(f"{incident_id}: ground-truth evidence IDs are invalid")
    metadata = record.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{incident_id}: ground-truth metadata must be an object")
    required = {"difficulty", "failure_family", "topology_family", "red_herring", "generator_version"}
    if set(metadata) != required:
        raise ValueError(f"{incident_id}: ground-truth metadata keys mismatch")
    if metadata["difficulty"] != input_record["slice"]:
        raise ValueError(f"{incident_id}: ground-truth difficulty mismatch")
    if metadata["failure_family"] != family:
        raise ValueError(f"{incident_id}: ground-truth family metadata mismatch")
    if metadata["topology_family"] != input_record["metadata"]["topology_family"]:
        raise ValueError(f"{incident_id}: topology metadata mismatch")
    if not isinstance(metadata["red_herring"], bool):
        raise ValueError(f"{incident_id}: ground-truth red_herring must be boolean")
    if metadata["generator_version"] != GENERATOR_VERSION:
        raise ValueError(f"{incident_id}: generator version mismatch")
    return dict(record)


def validate_benchmark(
    inputs: Mapping[str, list[Mapping[str, Any]]],
    ground_truth: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate all benchmark invariants and return a manifest-ready report."""

    validate_taxonomy()
    if set(inputs) != set(SLICES):
        raise ValueError(f"benchmark slices must be {list(SLICES)}")
    flat_inputs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for split in SLICES:
        for item in inputs[split]:
            validated = validate_input_record(item)
            if validated["incident_id"] in seen_ids:
                raise ValueError(f"duplicate incident ID: {validated['incident_id']}")
            seen_ids.add(validated["incident_id"])
            flat_inputs.append(validated)
    if len(ground_truth) != len(flat_inputs):
        raise ValueError("input and ground-truth counts differ")
    truth_by_id: dict[str, dict[str, Any]] = {}
    for truth in ground_truth:
        incident_id = truth.get("incident_id")
        if incident_id in truth_by_id:
            raise ValueError(f"duplicate ground-truth incident ID: {incident_id}")
        if incident_id not in seen_ids:
            raise ValueError(f"ground truth references unknown incident ID: {incident_id}")
        truth_by_id[incident_id] = validate_ground_truth_record(truth, next(item for item in flat_inputs if item["incident_id"] == incident_id))
    if set(truth_by_id) != seen_ids:
        raise ValueError("ground truth does not cover exactly the input incidents")
    normalized_values = [normalize_incident_text(item["incident_packet"]) for item in flat_inputs]
    normalized_duplicates = {
        value: count for value, count in Counter(normalized_values).items() if count > 1
    }
    if normalized_duplicates:
        raise ValueError(f"normalized duplicate incident packets: {len(normalized_duplicates)}")
    family_counts = Counter(truth["failure_mode"] for truth in ground_truth)
    slice_counts = {split: len(inputs[split]) for split in SLICES}
    if slice_counts != SLICE_COUNTS:
        raise ValueError(f"unexpected slice counts: {slice_counts}")
    if set(family_counts) != FAILURE_FAMILY_SET or set(family_counts.values()) != {12}:
        raise ValueError(f"each failure family must occur exactly 12 times: {dict(family_counts)}")
    hard_red_herring = sum(
        truth["metadata"]["red_herring"] for truth in ground_truth if truth["metadata"]["difficulty"] == "hard"
    )
    if hard_red_herring < 36:
        raise ValueError(f"HARD red-herring coverage is only {hard_red_herring}/48")
    return {
        "status": "pass",
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split_counts": slice_counts,
        "failure_family_counts": dict(sorted(family_counts.items())),
        "hard_red_herring_count": hard_red_herring,
        "hard_red_herring_rate": hard_red_herring / SLICE_COUNTS["hard"],
        "duplicate_count": 0,
        "normalized_duplicate_count": 0,
        "literal_label_leakage_count": 0,
        "invalid_reference_count": 0,
        "benchmark_fingerprint": benchmark_fingerprint(inputs, ground_truth),
    }


def _format_topology(topology: Mapping[str, object]) -> str:
    components = topology["components"]
    edges = topology["edges"]
    return "\n".join([*edges, "COMPONENTS", *[f"{role}: {name}" for role, name in sorted(components.items())]])


def _render_packet(
    incident_id: str,
    split: str,
    family_index: int,
    case_index: int,
    topology_name: str,
    topology: Mapping[str, object],
    red_herring: bool,
    temporal_cause: bool,
) -> tuple[str, list[str]]:
    spec = FAILURE_SPECS[FAILURE_FAMILIES[family_index]]
    components = topology["components"]
    culprit = components[spec.culprit_role]
    roles = ["gateway", "app", "worker", "consumer", "db", "cache", "external", "dns", "cert"]
    distractor_role = next(role for role in roles if role != spec.culprit_role)
    distractor = components[distractor_role]
    minute = 40 + ((family_index * 7 + case_index * 3) % 15)
    start = f"09:{minute:02d}"
    end = f"10:{(minute + 8) % 60:02d}"
    first_signal = f"09:{(minute + 2) % 60:02d}"
    change_time = f"09:{(minute + 1) % 60:02d}"
    if temporal_cause:
        change = f"{change_time} {culprit} deployment/configuration revision r{180 + family_index * 3 + case_index} became active"
        relation = "the first distinctive signal follows this change"
    else:
        change = f"09:{(minute + 10) % 60:02d} {distractor} deployment/configuration revision r{70 + case_index} became active"
        relation = "the incident signal predates this latest change"
    intro_pool = _TRANSFER_INTROS if split == "transfer" else _INTRODUCTIONS
    intro = intro_pool[(family_index + case_index) % len(intro_pool)].format(start=start, end=end)
    if split == "transfer":
        summary = (
            f"The handoff says: 'we saw customer impact before the platform alarm; "
            f"{culprit} deserves attention only if the timeline supports it.'"
        )
    else:
        summary = f"The visible symptom is on {components['gateway']}; that service may be downstream of the cause."
    red_herring_metric = (
        f"M4 {distractor} CPU: 46% -> 91%, then returned to 49% without a matching error increase"
        if red_herring
        else f"M4 {distractor} health probe: stable at 200 ms"
    )
    red_herring_log = (
        f"E3 {distractor}: one noisy timeout was observed, but its dependency checks stayed healthy"
        if red_herring
        else f"E3 {distractor}: no corresponding error spike"
    )
    packet = "\n".join(
        [
            "INCIDENT",
            f"id: {incident_id}",
            intro,
            summary,
            "",
            "TOPOLOGY",
            _format_topology(topology),
            "",
            "RECENT CHANGES",
            change,
            f"{first_signal} {relation}; incident declared at {end}",
            "",
            "METRICS",
            f"M1 {spec.metric_label}: {spec.metric_change}",
            f"M2 {components['gateway']} error/latency signal: baseline -> elevated",
            f"M3 {culprit} dependency health: {spec.secondary_signal}",
            red_herring_metric,
            "",
            "LOGS / EVENTS",
            f"E1 {culprit}: {spec.primary_log}",
            f"E2 {components['gateway']}: upstream request deadline exceeded after the first symptom",
            red_herring_log,
            "",
            "ALERTS / DEPENDENCY HEALTH",
            f"A1 {culprit}: {spec.alert_signal}",
            f"A2 {components['cache']} and {components['external']}: health checks normal",
        ]
    )
    return packet, ["M1", "E1", "A1"]


def generate_benchmark(seed: int = DEFAULT_SEED) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Generate exactly 144 deterministic incidents without an external model."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    validate_taxonomy()
    inputs: dict[str, list[dict[str, Any]]] = {split: [] for split in SLICES}
    ground_truth: list[dict[str, Any]] = []
    core_topologies = [name for name, data in TOPOLOGIES.items() if data["slice_group"] == "core"]
    transfer_topologies = [name for name, data in TOPOLOGIES.items() if data["slice_group"] == "transfer"]
    for family_index, family in enumerate(FAILURE_FAMILIES):
        spec = FAILURE_SPECS[family]
        for case_index in range(12):
            if case_index < 6:
                split = "standard"
                topology_name = core_topologies[(family_index + case_index + seed) % len(core_topologies)]
                red_herring = False
            elif case_index < 10:
                split = "hard"
                topology_name = core_topologies[(family_index + case_index + seed) % len(core_topologies)]
                red_herring = True
            else:
                split = "transfer"
                topology_name = transfer_topologies[(family_index + case_index + seed) % len(transfer_topologies)]
                red_herring = False
            topology = TOPOLOGIES[topology_name]
            temporal_cause = (family_index + case_index + seed) % 4 != 0
            incident_id = f"incident-02a-{split}-{family_index * 12 + case_index + 1:03d}"
            packet, evidence_ids = _render_packet(
                incident_id,
                split,
                family_index,
                case_index,
                topology_name,
                topology,
                red_herring,
                temporal_cause,
            )
            components = topology["components"]
            input_record = {
                "incident_id": incident_id,
                "slice": split,
                "incident_packet": packet,
                "metadata": {
                    "difficulty": split,
                    "topology_family": topology_name,
                    "red_herring": red_herring,
                    "present_components": sorted(set(components.values())),
                    "evidence_ids": evidence_ids,
                },
            }
            truth_record = {
                "incident_id": incident_id,
                "culprit_service": components[spec.culprit_role],
                "failure_mode": family,
                "recommended_action": spec.action,
                "evidence_ids": evidence_ids,
                "metadata": {
                    "difficulty": split,
                    "failure_family": family,
                    "topology_family": topology_name,
                    "red_herring": red_herring,
                    "generator_version": GENERATOR_VERSION,
                },
            }
            inputs[split].append(input_record)
            ground_truth.append(truth_record)
    validate_benchmark(inputs, ground_truth)
    return inputs, ground_truth


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.write_text(
        "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_benchmark(output_dir: str | Path, seed: int = DEFAULT_SEED) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    inputs, ground_truth = generate_benchmark(seed)
    report = validate_benchmark(inputs, ground_truth)
    for split in SLICES:
        _write_jsonl(directory / f"{split}.jsonl", inputs[split])
    _write_jsonl(directory / "ground_truth.jsonl", ground_truth)
    manifest = {
        **report,
        "generation_seed": seed,
        "incident_count": sum(report["split_counts"].values()),
        "slice_counts": report["split_counts"],
        "failure_families": list(FAILURE_FAMILIES),
        "actions": list(ACTIONS),
        "ground_truth_file": "ground_truth.jsonl",
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (directory / "validation_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def read_benchmark(directory: str | Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], dict[str, Any]]:
    path = Path(directory)
    inputs: dict[str, list[dict[str, Any]]] = {}
    for split in SLICES:
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate((path / f"{split}.jsonl").read_text(encoding="utf-8").splitlines(), 1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{split}.jsonl:{line_number}: malformed JSON: {exc.msg}") from exc
        inputs[split] = rows
    ground_truth = [
        json.loads(line)
        for line in (path / "ground_truth.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = validate_benchmark(inputs, ground_truth)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("benchmark_fingerprint") != report["benchmark_fingerprint"]:
        raise ValueError("manifest benchmark fingerprint mismatch")
    return inputs, ground_truth, manifest
