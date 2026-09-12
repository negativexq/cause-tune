"""Run evidence manifests and deterministic artifact provenance."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .doctor import environment_snapshot
from .experiment_contract import ExperimentContract


MANIFEST_SCHEMA_VERSION = 1
_RUN_ID_SAFE = re.compile(r"[^A-Za-z0-9_.-]+")


class EvidenceError(RuntimeError):
    """Raised when an evidence bundle cannot be created or finalized safely."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    source = Path(path)
    if not source.is_file():
        raise EvidenceError(f"artifact is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    source = Path(path)
    if source.is_file():
        return sha256_file(source)
    if not source.is_dir():
        raise EvidenceError(f"data path does not exist: {source}")
    digest = hashlib.sha256()
    for child in sorted(item for item in source.rglob("*") if item.is_file()):
        relative = child.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _git_state(cwd: str | Path = ".") -> dict[str, Any]:
    root = Path(cwd)
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=all"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "error": str(exc)}
    return {"available": True, "commit": commit, "dirty": dirty}


def _data_manifest(contract: ExperimentContract) -> dict[str, Any]:
    roles: dict[str, Any] = {}
    for role in ("train", "validation", "benchmark"):
        path = Path(contract.data[role].path)
        try:
            fingerprint = sha256_path(path)
            size = sum(item.stat().st_size for item in ([path] if path.is_file() else path.rglob("*")) if item.is_file())
        except (OSError, EvidenceError) as exc:
            raise EvidenceError(f"cannot fingerprint {role}: {exc}") from exc
        roles[role] = {
            "path": str(path),
            "sha256": fingerprint,
            "bytes": size,
        }
    return {"schema_version": 1, "roles": roles}


def _identity_payload(contract: ExperimentContract, data_manifest: Mapping[str, Any]) -> dict[str, Any]:
    config = contract.to_dict()
    config.pop("output", None)
    config.pop("metadata", None)
    config["data"] = {
        role: {"sha256": data_manifest["roles"][role]["sha256"]}
        for role in ("train", "validation", "benchmark")
    }
    return config


def experiment_fingerprint(contract: ExperimentContract, data_manifest: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json(_identity_payload(contract, data_manifest)).encode("utf-8"))


def make_run_id(contract: ExperimentContract, data_manifest: Mapping[str, Any]) -> str:
    model_slug = _RUN_ID_SAFE.sub("-", contract.model.model_id.replace("/", "-")).strip("-").lower()
    return f"{contract.experiment_id}__{model_slug}__{experiment_fingerprint(contract, data_manifest)[:8]}"


def initialize_evidence(
    contract: ExperimentContract,
    run_dir: str | Path,
    *,
    git_cwd: str | Path = ".",
    hardware_snapshot: bool = False,
) -> dict[str, Any]:
    """Create the immutable-at-start portion of a run evidence bundle."""

    destination = Path(run_dir)
    if destination.exists() and any(destination.iterdir()):
        raise EvidenceError(f"run directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    data_manifest = _data_manifest(contract)
    run_id = make_run_id(contract, data_manifest)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "experiment_id": contract.experiment_id,
        "run_id": run_id,
        "experiment_fingerprint": experiment_fingerprint(contract, data_manifest),
        "git": _git_state(git_cwd),
        "config": {"sha256": contract.sha256(), "path": "resolved_config.json"},
        "model": contract.model.to_dict(),
        "data": {role: data_manifest["roles"][role]["sha256"] for role in ("train", "validation", "benchmark")},
        "training": {
            "seed": contract.training.seed,
            "configured_steps": contract.training.stopping_policy.max_steps,
            "actual_steps": None,
            "stop_reason": None,
        },
        "selection": {"checkpoint": None, "source": "validation"},
        "artifacts": {},
    }
    contract.write_resolved(destination / "resolved_config.json")
    _write_json(destination / "environment.json", environment_snapshot(hardware=hardware_snapshot))
    _write_json(destination / "data_manifest.json", data_manifest)
    _write_json(destination / "evaluation_contract.json", contract.evaluation.to_dict())
    _write_json(destination / "manifest.json", manifest)
    return manifest


def record_training_result(
    run_dir: str | Path,
    *,
    actual_steps: int,
    stop_reason: str,
) -> dict[str, Any]:
    if not isinstance(actual_steps, int) or isinstance(actual_steps, bool) or actual_steps < 0:
        raise EvidenceError("actual_steps must be a non-negative integer")
    if not isinstance(stop_reason, str) or not stop_reason.strip():
        raise EvidenceError("stop_reason must not be empty")
    path = Path(run_dir) / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["training"]["actual_steps"] = actual_steps
    manifest["training"]["stop_reason"] = stop_reason
    _write_json(path, manifest)
    return manifest


def record_checkpoint_selection(run_dir: str | Path, *, checkpoint: int, source: str = "validation") -> dict[str, Any]:
    if source != "validation":
        raise EvidenceError("checkpoint selection provenance must use validation")
    if not isinstance(checkpoint, int) or isinstance(checkpoint, bool) or checkpoint < 0:
        raise EvidenceError("checkpoint must be a non-negative integer")
    destination = Path(run_dir)
    selection = {"schema_version": 1, "checkpoint": checkpoint, "source": source}
    _write_json(destination / "checkpoint_selection.json", selection)
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selection"] = {"checkpoint": checkpoint, "source": source}
    _write_json(manifest_path, manifest)
    return selection


def finalize_evidence(run_dir: str | Path) -> dict[str, Any]:
    """Hash persisted artifacts and attach the hashes to an immutable manifest.

    A finalized bundle may be inspected repeatedly, but it cannot be repaired
    or re-finalized after an artifact changes. This prevents a later write from
    silently replacing the provenance record for an earlier run.
    """

    destination = Path(run_dir)
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise EvidenceError(f"manifest is missing: {manifest_path}")
    current_manifest = load_manifest(destination)
    index_path = destination / "artifact_hashes.json"
    already_finalized = bool(current_manifest.get("artifacts")) or index_path.is_file()
    if already_finalized:
        if not index_path.is_file() or not current_manifest.get("artifacts"):
            raise EvidenceError("evidence bundle finalization is incomplete and cannot be repaired")
        try:
            indexed = _read_json(index_path)
        except (OSError, json.JSONDecodeError) as exc:
            raise EvidenceError("finalized evidence bundle has an unreadable artifact hash index") from exc
        expected = indexed.get("artifacts") if isinstance(indexed, Mapping) else None
        actual = artifact_hashes(destination)
        if expected != actual or current_manifest.get("artifacts") != expected:
            raise EvidenceError("finalized evidence bundle is immutable; persisted artifacts changed")
        return current_manifest
    if current_manifest["training"].get("actual_steps") is not None and not (
        destination / "checkpoint_selection.json"
    ).is_file():
        raise EvidenceError("training evidence is missing checkpoint_selection.json")
    artifacts = artifact_hashes(destination)
    artifact_hash_manifest = {"schema_version": 1, "artifacts": artifacts}
    _write_json(destination / "artifact_hashes.json", artifact_hash_manifest)
    manifest = current_manifest
    manifest["artifacts"] = dict(artifacts)
    _write_json(manifest_path, manifest)
    return manifest


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read run manifest: {path}: {exc}") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise EvidenceError(f"unsupported manifest schema_version: {manifest.get('schema_version')!r}")
    return manifest


def artifact_hashes(run_dir: str | Path, *, excluded: set[str] | None = None) -> dict[str, str]:
    """Hash persisted files by relative artifact name in stable order."""

    destination = Path(run_dir)
    ignored = excluded or {"manifest.json", "artifact_hashes.json"}
    return {
        path.relative_to(destination).as_posix(): sha256_file(path)
        for path in sorted(
            item for item in destination.rglob("*") if item.is_file() and item.name not in ignored
        )
    }
