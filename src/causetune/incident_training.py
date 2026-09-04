"""Independent data and training-foundation utilities for Experiment 02B.1.

The generator in this module deliberately does not call the frozen 02A
benchmark generator.  It uses a different namespace, topology vocabulary, and
surface-realization templates so the future specialization run has an
independent train/validation distribution.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .data.preprocess import PreprocessedExample, build_preprocessed_messages, decode_trainable_assistant
from .incident_benchmark import normalize_incident_text, packet_evidence_ids, read_benchmark
from .incident_taxonomy import ACTION_SET, FAILURE_FAMILIES, FAILURE_FAMILY_SET, FAILURE_SPECS


TRAINING_VERSION = "incident-diagnosis-02b1-v1"
TRAINING_GENERATOR_VERSION = "incident-diagnosis-training-local-v1"
TRAIN_SEED = 20260941
VALIDATION_SEED = 20260942
TRAIN_COUNT = 2400
VALIDATION_COUNT = 288
MAX_EARLY_STOP_STEPS = 50
REQUIRED_PACKET_SECTIONS = ("INCIDENT", "TOPOLOGY", "RECENT CHANGES", "METRICS", "LOGS / EVENTS")
TRAIN_DIFFICULTIES = ("standard", "hard", "transfer")

_SURFACE_VARIANTS = (
    "The handoff starts with customer reports.", "The first note came from the service owner.",
    "The alert was escalated after retries failed.", "The timeline was assembled from two operator notes.",
    "The incident room first recorded an edge symptom.", "A routine health review uncovered the pattern.",
    "The team compared peer instances before escalating.", "The report arrived after a second failed attempt.",
    "The owner describes a narrow but repeated impact.", "The packet combines an alert with a follow-up check.",
    "The initial report names an effect, not its source.", "The on-call engineer attached dependency checks.",
    "The incident was first noticed by a batch consumer.", "A customer-facing alarm preceded the platform alarm.",
    "The handoff includes a cleared warning for comparison.", "The operator asks for the primary causal component.",
    "The evidence was collected while the error rate was rising.", "A neighboring service also produced a transient warning.",
    "The summary was written while recovery was still pending.", "The service owner included normal peer telemetry.",
    "The report mixes direct evidence with downstream effects.", "The first page of the incident channel is condensed here.",
    "The responder checked the dependency before changing capacity.", "The packet contains one tempting but non-causal signal.",
    "The team is separating trigger, symptom, and noise.", "The alert stream contains both primary and secondary effects.",
    "The incident note preserves the order of the observations.", "The responder compared healthy and unhealthy paths.",
    "The packet was handed from application support to platform support.", "The outage summary includes a misleading timing coincidence.",
    "The operator marked one component healthy despite the broad alarm.", "The first diagnosis was withheld until the timeline was checked.",
)
_CONTEXT_A = (
    "checkout attempts were retried twice", "a scheduled batch was still running", "only one region reported impact", "the first alert came from a canary",
    "the incident began during ordinary traffic", "a support handoff added customer timestamps", "peer instances were compared", "the warning was absent on the standby path",
    "the operator saw impact on both read and write paths", "the request volume stayed near its baseline", "recovery had not started when this packet was captured", "the alarm covered three availability zones",
    "one replica recovered briefly", "the first page included a partial trace", "the operator checked a healthy dependency", "the service owner reported no planned traffic surge",
)
_CONTEXT_B = (
    "the affected route is used by internal jobs", "the deployment window was otherwise quiet", "the incident crossed a reporting boundary", "the alert threshold had been stable for weeks",
    "a neighboring component emitted a single warning", "the team had already ruled out a traffic spike", "a previous support attempt had not changed the symptom", "the packet was assembled from application and platform notes",
    "the visible error was observed at the edge", "the first graph showed a delayed secondary effect", "normal health checks are included as negative evidence", "the responder recorded both failure and recovery signals",
    "the issue appeared on a newly added route", "the on-call note uses local service terminology", "the owner compared this incident with a quiet peer", "the evidence was captured before mitigation",
)

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?(?:%|ms|s|GiB|k|K)?\b", re.IGNORECASE)
_EVIDENCE_RE = re.compile(r"\b(?:M|E|A|D)\d+\b")

# These names are intentionally disjoint from the 02A topology-family names.
_TOPOLOGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("retail_ledger", ("retail-edge", "basket-service", "settlement-store", "retail-cache", "merchant-rail", "retail-resolver")),
    ("search_platform", ("search-frontdoor", "query-router", "index-store", "query-cache", "ranking-engine", "search-dns")),
    ("workflow_platform", ("workflow-entry", "run-coordinator", "run-database", "run-cache", "task-broker", "workflow-dns")),
    ("data_plane", ("data-ingress", "transform-api", "warehouse-store", "schema-cache", "object-archive", "data-resolver")),
    ("notification_mesh", ("message-front", "message-router", "delivery-store", "message-cache", "carrier-link", "message-dns")),
    ("identity_fabric", ("identity-front", "session-engine", "identity-store", "token-cache", "trust-provider", "identity-dns")),
    ("media_orchestrator", ("media-front", "render-controller", "media-store", "render-cache", "codec-provider", "media-dns")),
    ("telemetry_plane", ("telemetry-gateway", "signal-normalizer", "signal-store", "signal-cache", "collector-link", "telemetry-dns")),
)


@dataclass(frozen=True)
class TrainingTruth:
    incident_id: str
    culprit_service: str
    failure_mode: str
    recommended_action: str
    evidence_ids: tuple[str, ...]
    difficulty: str
    topology_family: str
    red_herring: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "culprit_service": self.culprit_service,
            "failure_mode": self.failure_mode,
            "recommended_action": self.recommended_action,
            "evidence_ids": list(self.evidence_ids),
            "metadata": {
                "difficulty": self.difficulty,
                "failure_family": self.failure_mode,
                "topology_family": self.topology_family,
                "red_herring": self.red_herring,
                "generator_version": TRAINING_GENERATOR_VERSION,
            },
        }


def _stable_json(rows: Iterable[Mapping[str, Any]]) -> bytes:
    return "\n".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows).encode()


def training_fingerprint(inputs: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(_stable_json(inputs) + b"\nground_truth\n" + _stable_json(truths)).hexdigest()


def _normalized_packet(text: str) -> str:
    return normalize_incident_text(text)


def _canonical_packet_signature(record: Mapping[str, Any]) -> tuple[str, str, tuple[str, ...], str]:
    packet = record["incident_packet"]
    changes = packet.split("RECENT CHANGES", 1)[1].split("METRICS", 1)[0]
    return (
        str(record["metadata"]["topology_family"]),
        re.sub(r"\s+", " ", changes.lower()).strip(),
        tuple(sorted(record["metadata"]["evidence_ids"])),
        re.sub(r"\s+", " ", packet.split("TOPOLOGY", 1)[1].split("RECENT CHANGES", 1)[0].lower()).strip(),
    )


def _format_packet(incident_id: str, difficulty: str, family: str, topology_name: str, components: tuple[str, ...], case_index: int, red_herring: bool, rng_seed: int) -> tuple[str, str, tuple[str, ...]]:
    edge_a, edge_b, edge_c, edge_d, edge_e, dns = components
    # Role assignment is deterministic and independent of 02A's role map.
    culprit_by_family = {
        "db_connection_pool_exhaustion": edge_c,
        "db_query_regression": edge_c,
        "memory_leak": edge_b,
        "downstream_dependency_timeout": edge_e,
        "cache_stampede": edge_d,
        "kafka_consumer_lag": edge_e,
        "thread_pool_exhaustion": edge_b,
        "disk_io_saturation": edge_c,
        "dns_resolution_failure": dns,
        "tls_certificate_expiration": edge_e,
        "rate_limit_misconfiguration": edge_a,
        "configuration_regression": edge_b,
    }
    culprit = culprit_by_family[family]
    action = FAILURE_SPECS[family].action
    evidence_number = FAILURE_FAMILIES.index(family) + 1
    family_lines = {
        "db_connection_pool_exhaustion": ("connection acquisition wait", "18 ms -> 3.4 s", f"{culprit} workers wait before receiving a datastore session", "pool wait exceeded the request budget"),
        "db_query_regression": ("statement completion p95", "74 ms -> 5.2 s", f"{culprit} spends most of each request inside one slow read path", "statement latency exceeded the database objective"),
        "memory_leak": ("resident heap", "1.8 GiB -> 7.1 GiB", f"{culprit} retains objects across completed requests", "working set rose without a traffic increase"),
        "downstream_dependency_timeout": ("remote call p99", "140 ms -> 6.4 s", f"{culprit} responses arrive after the caller deadline", "remote health checks show delayed responses"),
        "cache_stampede": ("hot-key miss ratio", "6% -> 94%", f"{culprit} receives a burst of fills for the same expired keys", "misses cluster after a shared expiry boundary"),
        "kafka_consumer_lag": ("event age", "12 s -> 48 min", f"{culprit} commits offsets much more slowly than producers publish", "backlog crossed the delivery objective"),
        "thread_pool_exhaustion": ("busy worker fraction", "58% -> 100%", f"{culprit} has no free request workers for new arrivals", "worker availability reached zero"),
        "disk_io_saturation": ("storage await", "9 ms -> 210 ms", f"{culprit} writes queue behind a saturated volume", "device utilization exceeded its budget"),
        "dns_resolution_failure": ("name lookup p99", "4 ms -> 2.6 s", f"{culprit} cannot resolve new dependency addresses", "the resolver reports negative answers"),
        "tls_certificate_expiration": ("secure handshake errors", "0.2% -> 97%", f"{culprit} presents a credential outside its validity window", "validity checks reject the endpoint credential"),
        "rate_limit_misconfiguration": ("quota rejections", "0.3% -> 44%", f"{culprit} rejects normal callers under an unexpectedly low quota", "rejections begin after a policy edit"),
        "configuration_regression": ("application errors", "0.5% -> 21%", f"{culprit} returns invalid responses after a behavior flag changes", "the new error signature follows a setting change"),
    }
    metric, change, primary, alert = family_lines[family]
    minute = 11 + ((case_index * 7 + rng_seed) % 42)
    start = f"11:{minute:02d}"
    causal_change = case_index % 4 != 0
    change_service = culprit if causal_change else edge_a
    change_text = f"{start} {change_service} release/config revision {case_index + 31} completed"
    timing = (f"{start} distinctive telemetry followed the revision" if causal_change else f"{start} the first distinctive signal was already present before this revision")
    distractor = edge_a if culprit != edge_a else edge_b
    noise = (
        f"M4 {distractor} CPU briefly reached 93% and recovered without matching errors"
        if red_herring else f"M4 {distractor} health probe remained within its normal band"
    )
    intro = (
        f"{_SURFACE_VARIANTS[case_index % len(_SURFACE_VARIANTS)]} {_CONTEXT_B[case_index // len(_CONTEXT_A)]} Customer impact appeared during the {start} window; follow timestamps rather than the loudest alert."
        if difficulty == "transfer" else
        f"{_SURFACE_VARIANTS[case_index % len(_SURFACE_VARIANTS)]} {_CONTEXT_B[case_index // len(_CONTEXT_A)]} An on-call note covers a short {start} to 12:{(minute + 9) % 60:02d} incident window and includes unrelated telemetry."
    )
    packet = "\n".join([
        "INCIDENT", f"id: {incident_id}", intro,
        "", "TOPOLOGY",
        f"{edge_a} -> {edge_b}", f"{edge_b} -> {edge_c}", f"{edge_b} -> {edge_d}", f"{edge_b} -> {edge_e}", f"{edge_e} -> {dns}",
        f"components: {', '.join(components)}",
        "", "RECENT CHANGES", change_text, timing,
        "", "METRICS", f"M{evidence_number} {metric}: {change}", f"M{evidence_number + 12} {edge_a} customer-visible error/latency: baseline -> elevated", f"M{evidence_number + 24} {culprit} health signal: {primary}", noise,
        "", "LOGS / EVENTS", f"E{evidence_number} {culprit}: {primary}", f"E{evidence_number + 12} {edge_a}: downstream requests exceed their deadline after the first symptom", f"E{evidence_number + 24} {distractor}: a transient warning cleared while its dependency remained healthy",
        "", "ALERTS / DEPENDENCY HEALTH", f"A{evidence_number} {culprit}: {alert}", f"A{evidence_number + 12} {edge_d} and {edge_e}: no corresponding broad health failure",
    ])
    return packet, culprit, (f"M{evidence_number}", f"E{evidence_number}", f"A{evidence_number}")


def _difficulty_for(index: int, count: int) -> str:
    if index < round(count * 0.60):
        return "standard"
    if index < round(count * 0.90):
        return "hard"
    return "transfer"


def generate_training_split(split: str, count_per_family: int, seed: int, prefix: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if split not in {"train", "validation"} or count_per_family <= 0:
        raise ValueError("split must be train or validation with a positive family count")
    import random
    rng = random.Random(seed)
    jobs = [(family_index, index) for family_index in range(len(FAILURE_FAMILIES)) for index in range(count_per_family)]
    rng.shuffle(jobs)
    inputs: list[dict[str, Any]] = []
    truths: list[dict[str, Any]] = []
    for sequence, (family_index, case_index) in enumerate(jobs, 1):
        family = FAILURE_FAMILIES[family_index]
        difficulty = _difficulty_for(case_index, count_per_family)
        topology_name, components = _TOPOLOGIES[(sequence + family_index + seed) % len(_TOPOLOGIES)]
        red_herring = difficulty == "hard"
        incident_id = f"incident-02b1-{prefix}-{sequence:04d}"
        packet, culprit, evidence_ids = _format_packet(incident_id, difficulty, family, topology_name, components, case_index, red_herring, seed + sequence)
        present = list(components)
        inputs.append({"incident_id": incident_id, "split": split, "incident_packet": packet, "metadata": {"difficulty": difficulty, "topology_family": topology_name, "red_herring": red_herring, "present_components": present, "evidence_ids": list(evidence_ids)}})
        truths.append(TrainingTruth(incident_id, culprit, family, FAILURE_SPECS[family].action, evidence_ids, difficulty, topology_name, red_herring).to_dict())
    validate_training_split(inputs, truths, expected_count=count_per_family * len(FAILURE_FAMILIES), expected_per_family=count_per_family)
    return inputs, truths


def _literal_leakage(packet: str) -> list[str]:
    lower = packet.lower()
    leaked = []
    for label in (*FAILURE_FAMILIES, *ACTION_SET):
        if label in lower or label.replace("_", " ") in lower:
            leaked.append(label)
    return leaked


def validate_training_split(inputs: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]], *, expected_count: int, expected_per_family: int) -> dict[str, Any]:
    if len(inputs) != expected_count or len(truths) != expected_count:
        raise ValueError(f"expected {expected_count} inputs/truths")
    input_by_id = {}
    for record in inputs:
        keys = set(record)
        if keys != {"incident_id", "split", "incident_packet", "metadata"}:
            raise ValueError("invalid training input keys")
        incident_id = record["incident_id"]
        if incident_id in input_by_id:
            raise ValueError(f"duplicate incident ID: {incident_id}")
        input_by_id[incident_id] = record
        packet = record["incident_packet"]
        if not isinstance(packet, str) or not packet.strip() or any(section not in packet for section in REQUIRED_PACKET_SECTIONS):
            raise ValueError(f"{incident_id}: malformed incident packet")
        if _literal_leakage(packet):
            raise ValueError(f"{incident_id}: literal taxonomy/action leakage")
        metadata = record["metadata"]
        if set(metadata) != {"difficulty", "topology_family", "red_herring", "present_components", "evidence_ids"}:
            raise ValueError(f"{incident_id}: metadata keys mismatch")
        if metadata["difficulty"] not in TRAIN_DIFFICULTIES or not isinstance(metadata["red_herring"], bool):
            raise ValueError(f"{incident_id}: invalid metadata")
        ids = packet_evidence_ids(packet)
        evidence = metadata["evidence_ids"]
        if not isinstance(evidence, list) or len(evidence) != len(set(evidence)) or not 2 <= len(evidence) <= 4 or not set(evidence).issubset(ids):
            raise ValueError(f"{incident_id}: invalid evidence metadata")
        if not isinstance(metadata["present_components"], list) or not all(isinstance(component, str) and component in packet for component in metadata["present_components"]):
            raise ValueError(f"{incident_id}: invalid components metadata")
    truth_by_id = {}
    for truth in truths:
        incident_id = truth["incident_id"]
        if incident_id in truth_by_id or incident_id not in input_by_id:
            raise ValueError(f"invalid or duplicate ground truth ID: {incident_id}")
        truth_by_id[incident_id] = truth
        if set(truth) != {"incident_id", "culprit_service", "failure_mode", "recommended_action", "evidence_ids", "metadata"}:
            raise ValueError(f"{incident_id}: ground truth keys mismatch")
        if truth["failure_mode"] not in FAILURE_FAMILY_SET or truth["recommended_action"] not in ACTION_SET:
            raise ValueError(f"{incident_id}: invalid taxonomy value")
        if truth["recommended_action"] != FAILURE_SPECS[truth["failure_mode"]].action:
            raise ValueError(f"{incident_id}: action mapping mismatch")
        if truth["culprit_service"] not in input_by_id[incident_id]["metadata"]["present_components"]:
            raise ValueError(f"{incident_id}: unknown culprit")
        packet_ids = packet_evidence_ids(input_by_id[incident_id]["incident_packet"])
        if not set(truth["evidence_ids"]).issubset(packet_ids) or len(truth["evidence_ids"]) != len(set(truth["evidence_ids"])):
            raise ValueError(f"{incident_id}: invalid truth evidence")
        if set(truth["metadata"]) != {"difficulty", "failure_family", "topology_family", "red_herring", "generator_version"}:
            raise ValueError(f"{incident_id}: ground truth metadata keys mismatch")
        if truth["metadata"]["generator_version"] != TRAINING_GENERATOR_VERSION:
            raise ValueError(f"{incident_id}: wrong generator version")
    if set(truth_by_id) != set(input_by_id):
        raise ValueError("ground truth does not exactly cover inputs")
    family_counts = Counter(item["failure_mode"] for item in truths)
    if set(family_counts) != FAILURE_FAMILY_SET or set(family_counts.values()) != {expected_per_family}:
        raise ValueError(f"unexpected family counts: {dict(family_counts)}")
    normalized = [_normalized_packet(item["incident_packet"]) for item in inputs]
    if len(normalized) != len(set(normalized)):
        raise ValueError("normalized duplicate training packets")
    return {"count": len(inputs), "family_counts": dict(sorted(family_counts.items())), "difficulty_counts": dict(sorted(Counter(item["metadata"]["difficulty"] for item in inputs).items()))}


def contamination_report(train_inputs: Sequence[Mapping[str, Any]], validation_inputs: Sequence[Mapping[str, Any]], benchmark_dir: str | Path) -> dict[str, Any]:
    benchmark_inputs, _, manifest = read_benchmark(benchmark_dir)
    benchmark_flat = [item for split in benchmark_inputs.values() for item in split]
    groups = {"train": list(train_inputs), "validation": list(validation_inputs), "benchmark": benchmark_flat}
    normalized = {name: {_normalized_packet(item["incident_packet"]): item["incident_id"] for item in rows} for name, rows in groups.items()}
    exact = {name: {item["incident_packet"] for item in rows} for name, rows in groups.items()}
    signatures = {name: {_canonical_packet_signature(item) for item in rows} for name, rows in groups.items()}
    pairs = {}
    for left, right in (("train", "validation"), ("train", "benchmark"), ("validation", "benchmark")):
        pairs[f"{left}_vs_{right}"] = {
            "exact_text_overlap": len(exact[left] & exact[right]),
            "normalized_text_overlap": len(set(normalized[left]) & set(normalized[right])),
            "canonical_structure_overlap": len(signatures[left] & signatures[right]),
            "incident_id_overlap": len({item["incident_id"] for item in groups[left]} & {item["incident_id"] for item in groups[right]}),
        }
    return {"status": "pass" if all(value == 0 for pair in pairs.values() for value in pair.values()) else "fail", "benchmark_fingerprint": manifest["benchmark_fingerprint"], "pairs": pairs}


def build_diagnosis_chat_record(record: Mapping[str, Any], truth: Mapping[str, Any]) -> tuple[str, list[dict[str, str]]]:
    target = json.dumps({"culprit_service": truth["culprit_service"], "failure_mode": truth["failure_mode"], "recommended_action": truth["recommended_action"], "evidence_ids": truth["evidence_ids"]}, ensure_ascii=False, separators=(",", ":"))
    return record["incident_id"], [{"role": "user", "content": record["incident_packet"]}, {"role": "assistant", "content": target}]


def preprocess_incident_records(inputs: Sequence[Mapping[str, Any]], truths: Sequence[Mapping[str, Any]], tokenizer: Any, max_sequence_length: int, system_message: str) -> list[PreprocessedExample]:
    truth_by_id = {truth["incident_id"]: truth for truth in truths}
    return [build_preprocessed_messages(*build_diagnosis_chat_record(item, truth_by_id[item["incident_id"]]), tokenizer, max_sequence_length, system_message) for item in inputs]


def decoded_target_report(examples: Sequence[PreprocessedExample], truths: Sequence[Mapping[str, Any]], tokenizer: Any) -> dict[str, Any]:
    expected = {truth["incident_id"]: json.dumps({"culprit_service": truth["culprit_service"], "failure_mode": truth["failure_mode"], "recommended_action": truth["recommended_action"], "evidence_ids": truth["evidence_ids"]}, ensure_ascii=False, separators=(",", ":")) for truth in truths}
    decoded = {example.example_id: decode_trainable_assistant(example, tokenizer).replace("<|im_end|>", "").strip() for example in examples}
    mismatches = [example_id for example_id, target in expected.items() if decoded.get(example_id) != target]
    return {"count": len(examples), "decoded_target_mismatches": mismatches, "zero_supervised": sum(example.trainable_assistant_token_count == 0 for example in examples)}


def validate_training_manifest(manifest: Mapping[str, Any], *, expected_benchmark_fingerprint: str, expected_evaluation_fingerprint: str) -> None:
    required = {"experiment", "model_id", "quantization", "lora", "training", "train_fingerprint", "validation_fingerprint", "frozen_benchmark_fingerprint", "frozen_evaluation_fingerprint", "seeds", "checkpoint_policy", "early_stopping", "max_epochs", "training_sources"}
    if set(manifest) != required:
        raise ValueError(f"manifest keys must be {sorted(required)}")
    if manifest["experiment"] != "02B_production_incident_qlora" or manifest["model_id"] != "Qwen/Qwen3-4B":
        raise ValueError("unexpected experiment/model")
    if manifest["frozen_benchmark_fingerprint"] != expected_benchmark_fingerprint or manifest["frozen_evaluation_fingerprint"] != expected_evaluation_fingerprint:
        raise ValueError("frozen fingerprint invariant failed")
    if set(manifest["training_sources"]) != {"train", "validation"} or any(Path(str(value)).resolve() == Path("data/incident_diagnosis").resolve() for value in manifest["training_sources"].values()):
        raise ValueError("training sources must exclude frozen 02A benchmark")
    if manifest["checkpoint_policy"]["primary_metric"] != "diagnosis_exact_match":
        raise ValueError("checkpoint selection must use validation diagnosis exact match")
    if manifest["early_stopping"] != {"eval_interval_steps": 25, "patience": 3, "min_delta": 0.005, "warmup_floor_steps": 50}:
        raise ValueError("unexpected early-stopping policy")


class CheckpointSelectionPolicy:
    """Validation-only lexicographic checkpoint selection."""

    def __init__(self, min_delta: float = 0.005, patience: int = 3, warmup_floor_steps: int = 50) -> None:
        self.min_delta, self.patience, self.warmup_floor_steps = min_delta, patience, warmup_floor_steps
        self.best: tuple[float, float, float, float, int] | None = None
        self.best_metrics: dict[str, Any] | None = None
        self.best_step: int | None = None
        self.last_improvement_step: int | None = None
        self.patience_counter = 0
        self.stop_reason: str | None = None

    def observe(self, step: int, metrics: Mapping[str, Any]) -> bool:
        key = (float(metrics["diagnosis_exact_match"]), float(metrics["resolution_exact_match"]), float(metrics["failure_mode_macro_f1"]), -float(metrics["teacher_forced_loss"]), -step)
        meaningful = self.best is None or key[0] > self.best[0] + self.min_delta
        if self.best is None or key > self.best:
            self.best, self.best_metrics, self.best_step = key, dict(metrics), step
        if meaningful:
            self.last_improvement_step, self.patience_counter = step, 0
        elif step >= self.warmup_floor_steps:
            self.patience_counter += 1
        if step >= self.warmup_floor_steps and self.patience_counter >= self.patience:
            self.stop_reason = "validation_no_improvement"
            return True
        return False

    def state(self) -> dict[str, Any]:
        return {"best_step": self.best_step, "best_metric": None if self.best_metrics is None else self.best_metrics.get("diagnosis_exact_match"), "last_improvement_step": self.last_improvement_step, "patience_counter": self.patience_counter, "stop_reason": self.stop_reason}


def validation_schedule(max_optimizer_steps: int, interval_steps: int = 25) -> tuple[int, ...]:
    """Return the deterministic step-0-plus-interval validation schedule."""

    if max_optimizer_steps < 0 or interval_steps <= 0:
        raise ValueError("max_optimizer_steps must be non-negative and interval_steps positive")
    return (0, *range(interval_steps, max_optimizer_steps + 1, interval_steps))


def checkpoint_path(output_root: str | Path, step: int) -> Path:
    if step < 0:
        raise ValueError("checkpoint step must be non-negative")
    return Path(output_root) / f"checkpoint-step-{step:06d}"


def checkpoint_metadata(*, step: int, validation_metrics: Mapping[str, Any], resolved_manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Build metadata for an adapter-only checkpoint."""

    return {"checkpoint_type": "adapter_only", "step": step, "validation_metrics": dict(validation_metrics), "resolved_manifest": dict(resolved_manifest)}


def earliest_within_tolerance(history: Sequence[Mapping[str, Any]], tolerance: float = 0.01) -> dict[str, Any]:
    if not history:
        raise ValueError("validation history must not be empty")
    best = max(float(item["diagnosis_exact_match"]) for item in history)
    selected = next(item for item in history if float(item["diagnosis_exact_match"]) >= best - tolerance)
    return {"best_diagnosis_exact_match": best, "earliest_step": selected["step"], "tolerance": tolerance, "updates_potentially_avoided": max(0, int(history[-1]["step"]) - int(selected["step"]))}


def save_adapter_checkpoint(model: Any, tokenizer: Any, output_dir: str | Path, metadata: Mapping[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    (path / "checkpoint_metadata.json").write_text(json.dumps(dict(metadata), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
