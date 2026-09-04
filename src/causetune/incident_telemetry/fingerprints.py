"""Canonical serialization and fingerprints for Experiment 03 artifacts."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import CanonicalScenario, RenderedTelemetry


def canonical_json(value: Any) -> bytes:
    """Serialize JSON-compatible values without incidental formatting variance."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def scenario_fingerprint(scenario: CanonicalScenario) -> str:
    return sha256_json(scenario.to_dict(include_hashes=False))


def scenario_content_fingerprint(scenario: CanonicalScenario) -> str:
    """Fingerprint scenario content independent of its assigned instance ID."""

    payload = scenario.to_dict(include_hashes=False)
    payload.pop("scenario_id", None)
    payload.pop("split_group_id", None)
    payload.pop("topology_group_id", None)
    payload.pop("counterfactual_pair_id", None)
    provenance = payload.get("provenance")
    if isinstance(provenance, dict):
        provenance.pop("archetype_id", None)
        provenance.pop("seed", None)
    return sha256_json(payload)


def rendered_telemetry_fingerprint(rendered: RenderedTelemetry) -> str:
    return sha256_json(rendered.to_dict(include_hashes=False))


def rendered_content_fingerprint(rendered: RenderedTelemetry) -> str:
    """Fingerprint rendered observations independent of scenario identity."""

    payload = rendered.to_dict(include_hashes=False)
    payload.pop("scenario_id", None)
    payload.pop("canonical_scenario_hash", None)
    return sha256_json(payload)


def split_manifest_fingerprint(manifest: Any) -> str:
    payload = manifest.to_dict(include_fingerprint=False)
    return sha256_json(payload)


# Short aliases make the contract convenient to use from audit scripts.
canonical_scenario_fingerprint = scenario_fingerprint
telemetry_fingerprint = rendered_telemetry_fingerprint
