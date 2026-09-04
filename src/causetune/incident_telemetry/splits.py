"""Grouped, pre-render split protocol for Experiment 03A."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

from .models import CanonicalScenario, SPLIT_NAMES, SPLIT_SET


SPLIT_MANIFEST_VERSION = "incident-telemetry-split-manifest-v1"


class SplitIntegrityError(ValueError):
    """Raised when grouped split invariants are violated."""


@dataclass(frozen=True)
class SplitAssignment:
    scenario_id: str
    split: str
    split_group_id: str
    topology_group_id: str
    archetype_id: str
    generator_family: str
    template_family: str
    counterfactual_pair_id: str | None = None

    def __post_init__(self) -> None:
        values = {
            "scenario_id": self.scenario_id,
            "split_group_id": self.split_group_id,
            "topology_group_id": self.topology_group_id,
            "archetype_id": self.archetype_id,
            "generator_family": self.generator_family,
            "template_family": self.template_family,
        }
        for field, value in values.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field} must be a non-empty string")
        if self.split not in SPLIT_SET:
            raise ValueError(f"unknown evaluation split: {self.split!r}")
        if self.counterfactual_pair_id is not None and not self.counterfactual_pair_id.strip():
            raise ValueError("counterfactual_pair_id must be non-empty when present")

    @classmethod
    def from_scenario(cls, scenario: CanonicalScenario, split: str) -> "SplitAssignment":
        return cls(
            scenario_id=scenario.scenario_id,
            split=split,
            split_group_id=scenario.split_group_id,
            topology_group_id=scenario.topology_group_id,
            archetype_id=scenario.provenance.archetype_id,
            generator_family=scenario.provenance.generator_family,
            template_family=scenario.provenance.template_family,
            counterfactual_pair_id=scenario.counterfactual_pair_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "split": self.split,
            "split_group_id": self.split_group_id,
            "topology_group_id": self.topology_group_id,
            "archetype_id": self.archetype_id,
            "generator_family": self.generator_family,
            "template_family": self.template_family,
            "counterfactual_pair_id": self.counterfactual_pair_id,
        }


@dataclass(frozen=True)
class SplitManifest:
    manifest_version: str
    assignments: tuple[SplitAssignment, ...]
    sealed_holdout_consumed: bool = False
    manifest_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if self.manifest_version != SPLIT_MANIFEST_VERSION:
            raise ValueError(f"unsupported split manifest version: {self.manifest_version!r}")
        if not isinstance(self.sealed_holdout_consumed, bool):
            raise ValueError("sealed_holdout_consumed must be boolean")
        ids = [item.scenario_id for item in self.assignments]
        if len(ids) != len(set(ids)):
            raise ValueError("split manifest contains duplicate scenario IDs")
        if self.manifest_fingerprint is not None and not re.fullmatch(r"[0-9a-f]{64}", self.manifest_fingerprint):
            raise ValueError("manifest_fingerprint must be a SHA-256 hex digest")

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        result = {
            "manifest_version": self.manifest_version,
            "assignments": [item.to_dict() for item in self.assignments],
            "sealed_holdout_consumed": self.sealed_holdout_consumed,
        }
        if include_fingerprint:
            result["manifest_fingerprint"] = self.manifest_fingerprint
        return result

    def assignment_for(self, scenario_id: str) -> SplitAssignment:
        for assignment in self.assignments:
            if assignment.scenario_id == scenario_id:
                return assignment
        raise KeyError(scenario_id)

    def consume_sealed_holdout(self) -> "SplitManifest":
        if not any(item.split == "SEALED_HOLDOUT" for item in self.assignments):
            raise SplitIntegrityError("cannot consume a manifest without a SEALED_HOLDOUT")
        return replace(self, sealed_holdout_consumed=True, manifest_fingerprint=None)


def build_split_manifest(
    scenarios: Iterable[CanonicalScenario],
    split_by_scenario_id: Mapping[str, str],
) -> SplitManifest:
    """Build a manifest from explicit assignments; no random split is provided."""

    scenarios = tuple(scenarios)
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    if len(scenario_ids) != len(scenarios):
        raise SplitIntegrityError("cannot build a manifest from duplicate scenarios")
    if set(split_by_scenario_id) != scenario_ids:
        raise SplitIntegrityError("explicit split assignments must cover exactly every scenario")
    assignments = tuple(
        SplitAssignment.from_scenario(scenario, split_by_scenario_id[scenario.scenario_id])
        for scenario in scenarios
    )
    manifest = SplitManifest(SPLIT_MANIFEST_VERSION, assignments)
    validate_split_manifest(manifest, scenarios)
    return manifest


def validate_split_manifest(
    manifest: SplitManifest,
    scenarios: Iterable[CanonicalScenario],
) -> dict[str, Any]:
    """Validate all grouped split, OOD, pair, and holdout invariants."""

    if not isinstance(manifest, SplitManifest):
        raise SplitIntegrityError("manifest must be SplitManifest")
    scenarios = tuple(scenarios)
    by_id = {scenario.scenario_id: scenario for scenario in scenarios}
    if len(by_id) != len(scenarios):
        raise SplitIntegrityError("duplicate scenario IDs")
    if set(by_id) != {item.scenario_id for item in manifest.assignments}:
        raise SplitIntegrityError("manifest and scenarios do not cover the same IDs")

    for assignment in manifest.assignments:
        scenario = by_id[assignment.scenario_id]
        expected = SplitAssignment.from_scenario(scenario, assignment.split)
        if assignment != expected:
            raise SplitIntegrityError(f"assignment metadata disagrees with scenario: {assignment.scenario_id}")

    def no_cross_split(field: str) -> None:
        groups: dict[str, set[str]] = defaultdict(set)
        for item in manifest.assignments:
            groups[getattr(item, field)].add(item.split)
        leaked = {group: sorted(splits) for group, splits in groups.items() if len(splits) > 1}
        if leaked:
            raise SplitIntegrityError(f"{field} crosses protected splits: {leaked}")

    no_cross_split("split_group_id")
    no_cross_split("topology_group_id")

    pairs: dict[str, list[SplitAssignment]] = defaultdict(list)
    for item in manifest.assignments:
        if item.counterfactual_pair_id is not None:
            pairs[item.counterfactual_pair_id].append(item)
    for pair_id, members in pairs.items():
        if len(members) != 2:
            raise SplitIntegrityError(f"counterfactual pair {pair_id!r} must have exactly two members")
        if len({item.split for item in members}) != 1:
            raise SplitIntegrityError(f"counterfactual pair {pair_id!r} crosses splits")
        if any(by_id[item.scenario_id].case_type != "COUNTERFACTUAL" for item in members):
            raise SplitIntegrityError(f"counterfactual pair {pair_id!r} contains a non-counterfactual case")

    by_split = {split: [item for item in manifest.assignments if item.split == split] for split in SPLIT_NAMES}
    train = by_split["TRAIN"]
    train_values = {
        field: {getattr(item, field) for item in train}
        for field in ("template_family", "topology_group_id", "generator_family")
    }
    for split, field in (
        ("TEMPLATE_OOD", "template_family"),
        ("TOPOLOGY_OOD", "topology_group_id"),
        ("GENERATOR_OOD", "generator_family"),
    ):
        overlap = {getattr(item, field) for item in by_split[split]} & train_values[field]
        if overlap:
            raise SplitIntegrityError(f"{field} overlaps TRAIN in {split}: {sorted(overlap)}")

    counts = {split: len(by_split[split]) for split in SPLIT_NAMES}
    return {
        "status": "pass",
        "manifest_version": manifest.manifest_version,
        "scenario_count": len(scenarios),
        "split_counts": counts,
        "counterfactual_pair_count": len(pairs),
        "sealed_holdout_consumed": manifest.sealed_holdout_consumed,
        "protected_group_count": len({item.split_group_id for item in manifest.assignments}),
    }


validate_split_assignments = validate_split_manifest
