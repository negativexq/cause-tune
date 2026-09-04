"""Bounded, auditable Experiment 03C rendering pilot orchestration."""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Mapping

from .archetypes import ARCHETYPE_BY_ID
from .audit import audit_dataset
from .fingerprints import canonical_json, scenario_fingerprint, split_manifest_fingerprint
from .generator import GENERATOR_VERSION, CounterfactualPair
from .models import (
    OUTPUT_CONTRACT_VERSION,
    SCENARIO_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
    CanonicalScenario,
    RenderedTelemetry,
)
from .ontology import ONTOLOGY_VERSION
from .providers import ProviderClient, ProviderError, ProviderRawResponse
from .reference import REFERENCE_SEED, build_reference_corpus
from .rendering import RenderContractError, RenderPlan, build_render_plan, build_render_prompt, render_provider_payload


PILOT_VERSION = "incident-telemetry-03c-pilot-v1"
DEFAULT_RENDER_SEED = 303
DEFAULT_PROVIDER_CONFIG_ID = "fake_renderer_v1"
DEFAULT_RENDERER_FAMILIES = ("concise_ops_v1", "verbose_enterprise_v1", "ecosystem_native_v1")


@dataclass(frozen=True)
class PilotManifestEntry:
    entry_id: str
    scenario_id: str
    canonical_scenario_fingerprint: str
    split: str
    renderer_family: str
    provider_config_id: str
    render_seed: int
    expected_evidence_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.entry_id or not self.scenario_id or not self.canonical_scenario_fingerprint:
            raise ValueError("pilot manifest entry identity is incomplete")
        if not isinstance(self.render_seed, int) or self.render_seed < 0:
            raise ValueError("pilot render seed must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "scenario_id": self.scenario_id,
            "canonical_scenario_fingerprint": self.canonical_scenario_fingerprint,
            "split": self.split,
            "renderer_family": self.renderer_family,
            "provider_config_id": self.provider_config_id,
            "render_seed": self.render_seed,
            "expected_evidence_ids": list(self.expected_evidence_ids),
        }


@dataclass(frozen=True)
class PilotManifest:
    pilot_version: str
    reference_seed: int
    render_seed: int
    source_manifest_fingerprint: str
    provider_config_id: str
    entries: tuple[PilotManifestEntry, ...]
    manifest_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.pilot_version != PILOT_VERSION or not self.entries:
            raise ValueError("invalid or empty pilot manifest")
        ids = [item.entry_id for item in self.entries]
        if len(ids) != len(set(ids)):
            raise ValueError("pilot manifest contains duplicate entries")
        scenario_ids = [item.scenario_id for item in self.entries]
        if any(not item for item in scenario_ids):
            raise ValueError("pilot manifest contains an empty scenario ID")
        if self.manifest_fingerprint is not None and not re.fullmatch(r"[0-9a-f]{64}", self.manifest_fingerprint):
            raise ValueError("pilot manifest fingerprint must be SHA-256")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "pilot_version": self.pilot_version,
            "reference_seed": self.reference_seed,
            "render_seed": self.render_seed,
            "source_manifest_fingerprint": self.source_manifest_fingerprint,
            "provider_config_id": self.provider_config_id,
            "entries": [item.to_dict() for item in self.entries],
        }
        if include_fingerprint:
            result["manifest_fingerprint"] = self.manifest_fingerprint
        return result


@dataclass(frozen=True)
class PilotProviderConfig:
    config_id: str = DEFAULT_PROVIDER_CONFIG_ID
    temperature: float = 0.0
    max_output_tokens: int = 2048
    max_attempts: int = 3

    def __post_init__(self) -> None:
        if not self.config_id.strip() or self.temperature < 0 or self.max_output_tokens <= 0 or self.max_attempts < 1:
            raise ValueError("invalid pilot provider configuration")

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "max_attempts": self.max_attempts,
        }


def pilot_manifest_fingerprint(manifest: PilotManifest) -> str:
    return sha256(canonical_json(manifest.to_dict(include_fingerprint=False))).hexdigest()


def build_pilot_manifest(
    *,
    reference_seed: int = REFERENCE_SEED,
    render_seed: int = DEFAULT_RENDER_SEED,
    provider_config_id: str = DEFAULT_PROVIDER_CONFIG_ID,
    renderer_families: Iterable[str] = DEFAULT_RENDERER_FAMILIES,
) -> tuple[PilotManifest, dict[str, CanonicalScenario], dict[str, Any], list[CounterfactualPair]]:
    """Select non-sealed reference scenarios before any provider call."""

    families = tuple(renderer_families)
    if not families or len(set(families)) != len(families):
        raise ValueError("pilot needs distinct renderer families")
    scenarios, source_manifest, pairs = build_reference_corpus(reference_seed)
    assignments = {item.scenario_id: item for item in source_manifest.assignments}
    selected = [item for item in scenarios if assignments[item.scenario_id].split != "SEALED_HOLDOUT"]
    scenario_by_id = {item.scenario_id: item for item in selected}
    entries: list[PilotManifestEntry] = []
    for index, scenario in enumerate(selected):
        for family_index, family in enumerate(families):
            plan = build_render_plan(scenario, renderer_family=family, render_seed=render_seed + index * 17 + family_index, split_manifest_hash=source_manifest.manifest_fingerprint)
            identity = f"{scenario.scenario_id}|{family}|{plan.render_seed}|{provider_config_id}"
            entry_id = "render_03c_" + sha256(identity.encode("utf-8")).hexdigest()[:20]
            entries.append(
                PilotManifestEntry(
                    entry_id,
                    scenario.scenario_id,
                    scenario_fingerprint(scenario),
                    assignments[scenario.scenario_id].split,
                    family,
                    provider_config_id,
                    plan.render_seed,
                    tuple(sorted(plan.expected_evidence_ids)),
                )
            )
    manifest = PilotManifest(
        PILOT_VERSION,
        reference_seed,
        render_seed,
        split_manifest_fingerprint(source_manifest),
        provider_config_id,
        tuple(entries),
    )
    manifest = PilotManifest(**{**manifest.__dict__, "manifest_fingerprint": pilot_manifest_fingerprint(manifest)})
    return manifest, scenario_by_id, {item.scenario_id: item for item in source_manifest.assignments}, pairs


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _normalised_message(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\b\d+(?:\.\d+)?\b", "<number>", text.lower())).strip()


def _new_audit_state() -> dict[str, Any]:
    return {
        "attempted_renders": 0,
        "accepted": 0,
        "rejected": 0,
        "retries": 0,
        "malformed_output_failures": 0,
        "semantic_consistency_failures": 0,
        "label_leakage_failures": 0,
        "missing_evidence_failures": 0,
        "unsupported_evidence_failures": 0,
        "provider_failures": 0,
        "render_failures_by_provider_model": {},
        "render_failures_by_renderer_family": {},
        "acceptance_by_root_cause": {},
        "acceptance_by_fault_domain": {},
        "acceptance_by_runtime": {},
        "acceptance_by_topology": {},
        "acceptance_by_difficulty": {},
        "exact_duplicate_rendered_surfaces": 0,
    }


def _increment_nested(counter: dict[str, dict[str, int]], value: str, accepted: bool) -> None:
    bucket = counter.setdefault(value, {"accepted": 0, "rejected": 0})
    bucket["accepted" if accepted else "rejected"] += 1


def _failure_category(state: dict[str, Any], category: str) -> None:
    mapping = {
        "malformed_output": "malformed_output_failures",
        "semantic_consistency": "semantic_consistency_failures",
        "label_leakage": "label_leakage_failures",
        "missing_evidence": "missing_evidence_failures",
        "unsupported_evidence": "unsupported_evidence_failures",
        "provider_error": "provider_failures",
    }
    key = mapping.get(category, "semantic_consistency_failures")
    state[key] += 1


def run_rendering_pilot(
    output_dir: str | Path,
    *,
    provider: ProviderClient,
    manifest: PilotManifest,
    scenarios: Mapping[str, CanonicalScenario],
    assignments: Mapping[str, Any],
    pairs: list[CounterfactualPair],
    provider_config: PilotProviderConfig = PilotProviderConfig(),
) -> dict[str, Any]:
    """Run a bounded pilot; every attempt and rejection is retained."""

    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"pilot output directory is non-empty: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "accepted", "rejected", "prompts"):
        (directory / name).mkdir()
    if manifest.manifest_fingerprint != pilot_manifest_fingerprint(manifest):
        raise ValueError("pilot manifest fingerprint is invalid")
    _write_json(directory / "pilot_manifest.json", manifest.to_dict())

    state = _new_audit_state()
    accepted: list[tuple[PilotManifestEntry, CanonicalScenario, RenderedTelemetry, RenderPlan, ProviderRawResponse, int]] = []
    provenance_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for entry in manifest.entries:
        scenario = scenarios[entry.scenario_id]
        assignment = assignments.get(entry.scenario_id)
        if assignment is None or assignment.split != entry.split:
            raise ValueError(f"pilot assignment does not match manifest entry: {entry.entry_id}")
        if entry.split == "SEALED_HOLDOUT":
            raise ValueError("SEALED_HOLDOUT cannot be rendered by the pilot")
        if entry.canonical_scenario_fingerprint != scenario_fingerprint(scenario):
            raise ValueError(f"pilot entry canonical fingerprint mismatch: {entry.entry_id}")
        plan = build_render_plan(
            scenario,
            renderer_family=entry.renderer_family,
            render_seed=entry.render_seed,
            split_manifest_hash=manifest.source_manifest_fingerprint,
        )
        if tuple(sorted(plan.expected_evidence_ids)) != entry.expected_evidence_ids:
            raise ValueError(f"pilot entry expected evidence mismatch: {entry.entry_id}")
        prompt = build_render_prompt(plan)
        (directory / "prompts" / f"{entry.entry_id}.txt").write_text(prompt + "\n", encoding="utf-8")
        state["attempted_renders"] += 1
        accepted_result: tuple[RenderedTelemetry, ProviderRawResponse, int] | None = None
        for attempt in range(1, provider_config.max_attempts + 1):
            if attempt > 1:
                state["retries"] += 1
            raw: ProviderRawResponse | None = None
            try:
                raw = provider.generate(prompt)
                raw_path = directory / "raw" / f"{entry.entry_id}__attempt_{attempt}.json"
                _write_json(raw_path, {"entry_id": entry.entry_id, "attempt": attempt, "content": raw.content, "usage": raw.usage.to_dict(), "raw_response_hash": raw.content_hash()})
                payload = json.loads(raw.content)
                rendered = render_provider_payload(
                    scenario,
                    plan,
                    payload,
                    provider=provider.provider_name,
                    model_identifier=provider.model_identifier,
                    split_manifest_hash=manifest.source_manifest_fingerprint,
                )
                accepted_result = (rendered, raw, attempt)
                break
            except ProviderError as exc:
                _failure_category(state, exc.category)
                reason = str(exc)
                failure = {"entry_id": entry.entry_id, "attempt": attempt, "category": exc.category, "reason": reason, "raw_response_hash": raw.content_hash() if raw else None}
            except json.JSONDecodeError as exc:
                _failure_category(state, "malformed_output")
                failure = {"entry_id": entry.entry_id, "attempt": attempt, "category": "malformed_output", "reason": str(exc), "raw_response_hash": raw.content_hash() if raw else None}
            except RenderContractError as exc:
                _failure_category(state, exc.category)
                failure = {"entry_id": entry.entry_id, "attempt": attempt, "category": exc.category, "reason": str(exc), "raw_response_hash": raw.content_hash() if raw else None}
            except (TypeError, ValueError, KeyError) as exc:
                _failure_category(state, "semantic_consistency")
                failure = {"entry_id": entry.entry_id, "attempt": attempt, "category": "semantic_consistency", "reason": str(exc), "raw_response_hash": raw.content_hash() if raw else None}
            else:
                continue
            _write_json(directory / "rejected" / f"{entry.entry_id}__attempt_{attempt}.json", failure)
            failures.append(failure)
        if accepted_result is None:
            state["rejected"] += 1
            key = f"{provider.provider_name}/{provider.model_identifier}"
            state["render_failures_by_provider_model"][key] = state["render_failures_by_provider_model"].get(key, 0) + 1
            state["render_failures_by_renderer_family"][entry.renderer_family] = state["render_failures_by_renderer_family"].get(entry.renderer_family, 0) + 1
            accepted_for_dimensions = False
        else:
            accepted_for_dimensions = True
        _increment_nested(state["acceptance_by_root_cause"], scenario.root_cause_id or "HEALTHY_CONTROL", accepted_for_dimensions)
        _increment_nested(state["acceptance_by_fault_domain"], scenario.fault_domain or "NONE", accepted_for_dimensions)
        _increment_nested(state["acceptance_by_runtime"], scenario.runtime.runtime_id, accepted_for_dimensions)
        _increment_nested(state["acceptance_by_topology"], scenario.topology.topology_family, accepted_for_dimensions)
        _increment_nested(state["acceptance_by_difficulty"], scenario.difficulty, accepted_for_dimensions)
        if accepted_result is None:
            continue
        rendered, raw, accepted_attempt = accepted_result
        state["accepted"] += 1
        _write_json(
            directory / "accepted" / f"{entry.entry_id}.json",
            {
                "entry_id": entry.entry_id,
                "experiment_version": "03C",
                "ontology_version": ONTOLOGY_VERSION,
                "scenario_engine_version": GENERATOR_VERSION,
                "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
                "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
                "output_contract_version": OUTPUT_CONTRACT_VERSION,
                "attempt": accepted_attempt,
                "scenario_id": scenario.scenario_id,
                "canonical_scenario_fingerprint": scenario_fingerprint(scenario),
                "renderer_family": entry.renderer_family,
                "provider": provider.provider_name,
                "model_identifier": provider.model_identifier,
                "provider_adapter_version": provider.adapter_version,
                "render_seed": entry.render_seed,
                "raw_response_hash": raw.content_hash(),
                "accepted_render_hash": rendered.rendered_telemetry_hash,
                "rendered_telemetry": rendered.to_dict(),
            },
        )
        accepted.append((entry, scenario, rendered, plan, raw, accepted_attempt))
        provenance_records.append({
            "entry_id": entry.entry_id,
            "experiment_version": "03C",
            "ontology_version": ONTOLOGY_VERSION,
            "scenario_engine_version": GENERATOR_VERSION,
            "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
            "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
            "scenario_id": scenario.scenario_id,
            "canonical_scenario_fingerprint": scenario_fingerprint(scenario),
            "renderer_family": entry.renderer_family,
            "prompt_template_version": plan.prompt_template_version,
            "provider_adapter_version": provider.adapter_version,
            "provider": provider.provider_name,
            "model_identifier": provider.model_identifier,
            "generation_parameters": provider_config.to_dict(),
            "render_seed": entry.render_seed,
            "attempt": accepted_attempt,
            "raw_response_hash": raw.content_hash(),
            "accepted_render_hash": rendered.rendered_telemetry_hash,
            "pilot_manifest_fingerprint": manifest.manifest_fingerprint,
            "usage": raw.usage.to_dict(),
        })

    accepted_hashes = [item[2].rendered_telemetry_hash for item in accepted]
    state["exact_duplicate_rendered_surfaces"] = len(accepted_hashes) - len(set(accepted_hashes))
    audit_groups = []
    for family in {item.renderer_family for item in manifest.entries}:
        family_rendered = [item[2] for item in accepted if item[0].renderer_family == family]
        if family_rendered:
            report = audit_dataset(
                tuple(scenarios.values()),
                split_manifest=None,
                rendered_telemetry=family_rendered,
                archetypes=ARCHETYPE_BY_ID,
                counterfactual_pairs=pairs,
            )
            audit_groups.append(report)
    audit_status = "pass" if all(item["status"] == "pass" for item in audit_groups) and state["rejected"] == 0 else "fail"
    _write_json(directory / "audit.json", {"pilot_version": PILOT_VERSION, "status": audit_status, **state, "failure_details": failures, "renderer_audits": audit_groups})

    diversity = _diversity_report(accepted)
    _write_json(directory / "distribution.json", {"pilot_version": PILOT_VERSION, "scenario_count": len(scenarios), "accepted": state["accepted"], "rejected": state["rejected"], "by_split": dict(Counter(item.split for item in manifest.entries)), "diversity": diversity})
    _write_json(directory / "provenance.json", {"pilot_version": PILOT_VERSION, "manifest_fingerprint": manifest.manifest_fingerprint, "records": provenance_records})
    _write_json(directory / "inspection_sample.json", {"pilot_version": PILOT_VERSION, "ground_truth_included": False, "samples": [_inspection_record(item) for item in accepted[:20]]})
    return {"pilot_version": PILOT_VERSION, "status": audit_status, "attempted_renders": state["attempted_renders"], "accepted": state["accepted"], "rejected": state["rejected"], "retries": state["retries"], "manifest_fingerprint": manifest.manifest_fingerprint}


def _inspection_record(item: tuple[PilotManifestEntry, CanonicalScenario, RenderedTelemetry, RenderPlan, ProviderRawResponse, int]) -> dict[str, Any]:
    entry, scenario, rendered, plan, _raw, _attempt = item
    return {
        "entry_id": entry.entry_id,
        "scenario_id": scenario.scenario_id,
        "runtime": scenario.runtime.to_dict(),
        "difficulty": scenario.difficulty,
        "renderer_family": entry.renderer_family,
        "semantic_evidence": [{"evidence_id": item.evidence_id, "kind": item.semantic_kind, "role": item.causal_role, "source_component": item.source_component} for item in plan.entries],
        "rendered_telemetry": rendered.to_dict(),
    }


def _diversity_report(accepted: list[tuple[PilotManifestEntry, CanonicalScenario, RenderedTelemetry, RenderPlan, ProviderRawResponse, int]]) -> dict[str, Any]:
    messages = [observation.text for _entry, _scenario, rendered, _plan, _raw, _attempt in accepted for observation in rendered.observations]
    normalised = [_normalised_message(text) for text in messages]
    message_counts = Counter(normalised)
    family_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for entry, _scenario, rendered, _plan, _raw, _attempt in accepted:
        family_counts[entry.renderer_family].update(_normalised_message(observation.text) for observation in rendered.observations)
    return {
        "observation_count": len(messages),
        "unique_message_count": len(set(messages)),
        "unique_normalized_message_count": len(set(normalised)),
        "exact_duplicate_rate": round((len(messages) - len(set(messages))) / len(messages), 6) if messages else 0,
        "normalized_duplicate_rate": round((len(messages) - len(set(normalised))) / len(messages), 6) if messages else 0,
        "vocabulary_size": len(set(word for text in messages for word in re.findall(r"[a-z0-9_/-]+", text.lower()))),
        "most_repeated_normalized_messages": [{"message": message, "count": count} for message, count in message_counts.most_common(10)],
        "by_renderer_family": {family: {"observation_count": sum(counts.values()), "unique_normalized_messages": len(counts)} for family, counts in sorted(family_counts.items())},
        "interpretation": "Lexical statistics only; semantic diversity and near-duplicate detection are deferred.",
    }
