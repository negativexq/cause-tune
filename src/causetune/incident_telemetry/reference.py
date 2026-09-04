"""Small deterministic reference corpus for the 03B contract foundation."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from .archetypes import ARCHETYPE_BY_ID, ARCHETYPES
from .audit import audit_dataset
from .fingerprints import scenario_fingerprint, split_manifest_fingerprint
from .generator import (
    GENERATOR_VERSION,
    CounterfactualPair,
    GenerationContext,
    ScenarioRequest,
    ScenarioGenerator,
)
from .models import (
    CanonicalScenario,
    OUTPUT_CONTRACT_VERSION,
    SCENARIO_SCHEMA_VERSION,
    TELEMETRY_SCHEMA_VERSION,
)
from .ontology import ONTOLOGY_VERSION
from .runtimes import RUNTIME_FAMILIES
from .splits import SplitManifest, build_split_manifest, validate_split_manifest
from .topologies import TOPOLOGY_FAMILIES


REFERENCE_VERSION = "incident-telemetry-03b-reference-v1"
REFERENCE_SEED = 302
_SINGLE_SPLITS = (
    ("TRAIN", 12, "STANDARD"),
    ("VALIDATION", 8, "STANDARD"),
    ("ID_TEST", 5, "STANDARD"),
    ("HARD_TEST", 5, "HARD"),
    ("TEMPLATE_OOD", 4, "HARD"),
    ("TOPOLOGY_OOD", 4, "HARD"),
    ("GENERATOR_OOD", 4, "HARD"),
    ("ABSTENTION_TEST", 4, "INCOMPLETE"),
    ("SEALED_HOLDOUT", 4, "STANDARD"),
)


def _context(split: str, index: int, topology: str) -> GenerationContext:
    return GenerationContext(
        split_group_id=f"03b:reference:group:{split.lower()}:{index:03d}",
        topology_group_id=f"03b:reference:topology:{topology.lower()}:{split.lower()}",
        generator_family="deterministic-engine-ood-v1" if split == "GENERATOR_OOD" else GENERATOR_VERSION,
        template_family="template-ood-03b-v1" if split == "TEMPLATE_OOD" else "not-rendered-03b-v1",
    )


def build_reference_corpus(seed: int = REFERENCE_SEED) -> tuple[list[CanonicalScenario], SplitManifest, list[CounterfactualPair]]:
    """Build a fixture-sized corpus without rendering telemetry."""

    if not isinstance(seed, int) or seed < 0:
        raise ValueError("reference seed must be a non-negative integer")
    generator = ScenarioGenerator()
    scenario_records: list[tuple[CanonicalScenario, str]] = []
    pairs: list[CounterfactualPair] = []
    index = 0
    for split, count, difficulty in _SINGLE_SPLITS:
        for offset in range(count):
            archetype = ARCHETYPES[index % len(ARCHETYPES)]
            topology = archetype.compatible_topology_families[(index + seed) % len(archetype.compatible_topology_families)]
            runtime = archetype.allowed_runtime_families[(index + seed) % len(archetype.allowed_runtime_families)]
            scenario_records.append(
                (
                    generator.generate(
                        ScenarioRequest(
                            archetype.archetype_id,
                            topology,
                            runtime,
                            difficulty,
                            seed + index * 17,
                            _context(split, index, topology),
                        )
                    ),
                    split,
                )
            )
            index += 1

    pair_context = GenerationContext(
        split_group_id="03b:reference:group:counterfactual:001",
        topology_group_id="03b:reference:topology:checkout:counterfactual",
        generator_family=GENERATOR_VERSION,
        template_family="not-rendered-03b-v1",
        counterfactual_pair_id="pair_03b_reference_001",
    )
    first, second, pair = generator.generate_counterfactual_pair(
        "container_memory_exhaustion",
        "cpu_throttling",
        topology_family="CHECKOUT",
        runtime_family="python_fastapi",
        pair_seed=seed + 10000,
        pair_id="pair_03b_reference_001",
        context=pair_context,
    )
    scenario_records.extend(((first, "COUNTERFACTUAL_TEST"), (second, "COUNTERFACTUAL_TEST")))
    pairs.append(pair)

    scenarios = [scenario for scenario, _split in scenario_records]
    split_by_id = {scenario.scenario_id: split for scenario, split in scenario_records}
    manifest = build_split_manifest(scenarios, split_by_id)
    manifest_hash = split_manifest_fingerprint(manifest)
    scenarios = [
        replace(
            scenario,
            provenance=replace(scenario.provenance, split_manifest_hash=manifest_hash),
        )
        for scenario in scenarios
    ]
    manifest = replace(manifest, manifest_fingerprint=manifest_hash)
    validate_split_manifest(manifest, scenarios)
    return scenarios, manifest, pairs


def distribution_summary(scenarios: list[CanonicalScenario], manifest: SplitManifest, pairs: list[CounterfactualPair]) -> dict[str, Any]:
    assignments = {item.scenario_id: item for item in manifest.assignments}
    return {
        "reference_version": REFERENCE_VERSION,
        "scenario_count": len(scenarios),
        "by_root_cause": dict(sorted(Counter(item.root_cause_id for item in scenarios).items())),
        "by_fault_domain": dict(sorted(Counter(item.fault_domain for item in scenarios).items())),
        "by_topology_family": dict(sorted(Counter(item.topology.topology_family for item in scenarios).items())),
        "by_runtime": dict(sorted(Counter(item.runtime.runtime_id for item in scenarios).items())),
        "by_difficulty": dict(sorted(Counter(item.difficulty for item in scenarios).items())),
        "by_answerability": dict(sorted(Counter(item.answerability for item in scenarios).items())),
        "distractor_count": sum(len(item.distractors) for item in scenarios),
        "counterfactual_pair_count": len(pairs),
        "by_split": dict(sorted(Counter(item.split for item in assignments.values()).items())),
        "by_split_group": dict(sorted(Counter(item.split_group_id for item in assignments.values()).items())),
        "ontology_ids_covered": sorted({item.root_cause_id for item in scenarios}),
        "runtime_catalog": [item.runtime_id for item in RUNTIME_FAMILIES],
        "topology_catalog": [item.family_id for item in TOPOLOGY_FAMILIES],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_reference_corpus(output_dir: str | Path, seed: int = REFERENCE_SEED) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    scenarios, manifest, pairs = build_reference_corpus(seed)
    audit = audit_dataset(
        scenarios,
        split_manifest=manifest,
        archetypes=ARCHETYPE_BY_ID,
        counterfactual_pairs=pairs,
    )
    scenario_fingerprints = [scenario_fingerprint(item) for item in scenarios]
    contract = {
        "experiment": "03B",
        "reference_version": REFERENCE_VERSION,
        "seed": seed,
        "ontology_version": ONTOLOGY_VERSION,
        "scenario_schema_version": SCENARIO_SCHEMA_VERSION,
        "telemetry_schema_version": TELEMETRY_SCHEMA_VERSION,
        "output_contract_version": OUTPUT_CONTRACT_VERSION,
        "scenario_engine_version": GENERATOR_VERSION,
        "telemetry_rendering_performed": False,
    }
    (directory / "reference_scenarios.jsonl").write_text(
        "\n".join(json.dumps(item.to_dict(), ensure_ascii=False, sort_keys=True) for item in scenarios) + "\n",
        encoding="utf-8",
    )
    _write_json(directory / "reference_manifest.json", {**contract, "manifest": manifest.to_dict(), "scenario_fingerprints": scenario_fingerprints})
    _write_json(directory / "audit.json", {**contract, "audit": audit})
    _write_json(directory / "distribution.json", {**contract, **distribution_summary(scenarios, manifest, pairs)})
    _write_json(directory / "counterfactual_pairs.json", {**contract, "pairs": [pair.to_dict() for pair in pairs]})
    return {
        **contract,
        "scenario_count": len(scenarios),
        "manifest_fingerprint": split_manifest_fingerprint(manifest),
        "audit_status": audit["status"],
        "output_dir": str(directory),
    }
