"""Offline verification of CauseTune evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import EvidenceError, artifact_hashes, experiment_fingerprint, load_manifest, make_run_id, sha256_path
from .experiment_contract import ExperimentContractError, resolve_experiment_config


PASS = "PASS"
FAIL = "FAIL"
NOT_APPLICABLE = "NOT_APPLICABLE"


class VerificationFailure(RuntimeError):
    """Raised by :func:`verify` when an evidence bundle is not verified."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["summary"]["message"])


def _check(name: str, status: str, message: str, details: Mapping[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"name": name, "status": status, "message": message}
    if details:
        result["details"] = {str(key): details[key] for key in sorted(details, key=str)}
    return result


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read JSON artifact {path}: {exc}") from exc


def score_predictions(path: str | Path) -> dict[str, Any]:
    """Score the small, dependency-free prediction format used by M11.

    Rows must provide ``prediction`` and ``expected`` (or their ``output`` /
    ``target`` aliases). Values are compared as parsed JSON when possible and
    as ordinary values otherwise. This scorer is intentionally deterministic
    and does not perform generation, repair, or model loading.
    """

    predictions_path = Path(path)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(predictions_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"invalid prediction JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise EvidenceError(f"prediction line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise EvidenceError("predictions.jsonl is empty")

    def value(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
        for name in names:
            if name in row:
                return row[name]
        raise EvidenceError(f"prediction row is missing one of: {', '.join(names)}")

    correct = 0
    def comparable(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    for row in rows:
        expected = value(row, ("expected", "target"))
        prediction = value(row, ("prediction", "output"))
        if comparable(expected) == comparable(prediction):
            correct += 1
    return {
        "scorer_version": "causetune-exact-match-v1",
        "count": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows),
    }


def score_incident_predictions(path: str | Path) -> dict[str, Any]:
    """Re-score persisted incident rows without loading a model.

    The evidence runner persists the raw output plus the exact input metadata
    required by the canonical incident scorer. Reconstructing the scorer input
    here keeps offline verification independent of model inference while still
    checking JSON/schema and field-level incident metrics.
    """

    from .incident_evaluation import evaluate_incidents

    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceError(f"invalid incident prediction JSON at line {line_number}: {exc}") from exc
        if not isinstance(row, Mapping):
            raise EvidenceError(f"incident prediction line {line_number} is not an object")
        rows.append(row)
    if not rows:
        raise EvidenceError("predictions.jsonl is empty")

    incidents = []
    truths: dict[str, dict[str, Any]] = {}
    raw_outputs: dict[str, str] = {}
    for row in rows:
        incident_id = str(row["incident_id"])
        metadata = row.get("input_metadata")
        expected = row.get("expected")
        if not isinstance(metadata, Mapping) or not isinstance(expected, Mapping):
            raise EvidenceError(f"incident prediction {incident_id} lacks input_metadata or expected object")
        available = metadata.get("available_evidence_ids")
        components = metadata.get("present_components")
        if not isinstance(available, list) or not isinstance(components, list):
            raise EvidenceError(f"incident prediction {incident_id} has invalid input metadata")
        incidents.append({
            "incident_id": incident_id,
            "slice": row["slice"],
            "incident_packet": " ".join(str(item) for item in available),
            "metadata": {
                "present_components": components,
                "difficulty": row["difficulty"],
            },
        })
        truths[incident_id] = {
            "incident_id": incident_id,
            "culprit_service": expected["culprit_service"],
            "failure_mode": expected["failure_mode"],
            "recommended_action": expected["recommended_action"],
            "evidence_ids": expected["evidence_ids"],
            "metadata": {
                "difficulty": row["difficulty"],
                "failure_family": row["failure_family"],
                "topology_family": row["topology_family"],
                "red_herring": row["red_herring"],
            },
        }
        raw_outputs[incident_id] = str(row["raw_output"])
    result = evaluate_incidents(incidents, truths, raw_outputs)
    enriched = []
    for source, scored in zip(rows, result["predictions"]):
        enriched.append({**scored, "input_metadata": source["input_metadata"]})
    result["predictions"] = enriched
    return result


def _verify_artifacts(destination: Path, manifest: Mapping[str, Any], checks: list[dict[str, Any]]) -> None:
    path = destination / "artifact_hashes.json"
    if not path.is_file():
        checks.append(_check("artifact_hashes", FAIL, "artifact hash index is missing"))
        return
    try:
        recorded = _read_json(path)
        actual = artifact_hashes(destination)
    except (EvidenceError, OSError) as exc:
        checks.append(_check("artifact_hashes", FAIL, str(exc)))
        return
    expected = recorded.get("artifacts") if isinstance(recorded, Mapping) else None
    if not isinstance(expected, Mapping):
        checks.append(_check("artifact_hashes", FAIL, "artifact hash index schema is invalid"))
        return
    expected = {str(key): str(value) for key, value in expected.items()}
    if expected != actual:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        changed = sorted(key for key in set(expected) & set(actual) if expected[key] != actual[key])
        checks.append(
            _check(
                "artifact_hashes",
                FAIL,
                "artifact hash mismatch",
                {"missing": missing, "extra": extra, "changed": changed},
            )
        )
    elif manifest.get("artifacts") != expected:
        checks.append(_check("artifact_hashes", FAIL, "manifest artifact map differs from artifact hash index"))
    else:
        checks.append(_check("artifact_hashes", PASS, "all persisted artifact hashes match"))


def _verify_predictions(destination: Path, checks: list[dict[str, Any]], *, scorer_version: str | None = None) -> None:
    predictions = destination / "predictions.jsonl"
    evaluation = destination / "evaluation.json"
    if not predictions.exists() and not evaluation.exists():
        checks.append(_check("evaluation_reproduction", NOT_APPLICABLE, "no persisted predictions/evaluation artifacts"))
        return
    if not predictions.is_file() or not evaluation.is_file():
        checks.append(_check("evaluation_reproduction", FAIL, "predictions and evaluation must be persisted together"))
        return
    try:
        recomputed = (
            score_incident_predictions(predictions)
            if scorer_version == "incident-scorer-v1"
            else score_predictions(predictions)
        )
        persisted = _read_json(evaluation)
        persisted_metrics = persisted.get("metrics", persisted) if isinstance(persisted, Mapping) else None
        if not isinstance(persisted_metrics, Mapping):
            raise EvidenceError("evaluation.json metrics object is missing")
        expected = {key: persisted_metrics.get(key) for key in recomputed}
        if expected != recomputed:
            checks.append(
                _check("evaluation_reproduction", FAIL, "metrics do not reproduce from persisted predictions", {"expected": expected, "recomputed": recomputed})
            )
        else:
            checks.append(_check("evaluation_reproduction", PASS, "metrics reproduce from persisted predictions", recomputed))
    except (EvidenceError, OSError, TypeError) as exc:
        checks.append(_check("evaluation_reproduction", FAIL, str(exc)))


def verification_report(run_dir: str | Path, *, offline: bool = True) -> dict[str, Any]:
    """Verify a run bundle without repair, inference, or model loading."""

    destination = Path(run_dir)
    checks: list[dict[str, Any]] = []
    required = (
        "manifest.json",
        "resolved_config.json",
        "environment.json",
        "data_manifest.json",
        "evaluation_contract.json",
        "artifact_hashes.json",
    )
    missing = [name for name in required if not (destination / name).is_file()]
    if missing:
        checks.append(_check("bundle", FAIL, "required evidence artifact is missing", {"missing": missing}))
        return _report(run_dir, offline, checks)
    try:
        manifest = load_manifest(destination)
    except EvidenceError as exc:
        checks.append(_check("manifest", FAIL, str(exc)))
        return _report(run_dir, offline, checks)
    checks.append(_check("manifest", PASS, "manifest schema is valid"))

    resolved_contract = None
    try:
        resolved = resolve_experiment_config(_read_json(destination / "resolved_config.json"))
        resolved_contract = resolved
        config_hash = resolved.sha256()
        if config_hash != manifest.get("config", {}).get("sha256"):
            checks.append(_check("config", FAIL, "resolved config hash mismatch"))
        else:
            checks.append(_check("config", PASS, "resolved config hash matches manifest"))
        evaluation = _read_json(destination / "evaluation_contract.json")
        if evaluation != resolved.evaluation.to_dict():
            checks.append(_check("evaluation_contract", FAIL, "evaluation contract differs from resolved config"))
        else:
            checks.append(_check("evaluation_contract", PASS, "evaluation contract matches resolved config"))
    except (ExperimentContractError, EvidenceError, OSError, TypeError) as exc:
        checks.append(_check("config", FAIL, str(exc)))

    try:
        data_manifest = _read_json(destination / "data_manifest.json")
        roles = data_manifest["roles"]
        for role in ("train", "validation", "benchmark"):
            entry = roles[role]
            actual = sha256_path(entry["path"])
            if actual != entry["sha256"] or actual != manifest.get("data", {}).get(role):
                checks.append(_check(f"data_{role}", FAIL, f"{role} fingerprint mismatch"))
            else:
                checks.append(_check(f"data_{role}", PASS, f"{role} fingerprint matches"))
        if resolved_contract is not None:
            expected_run_id = make_run_id(resolved_contract, data_manifest)
            expected_fingerprint = experiment_fingerprint(resolved_contract, data_manifest)
            if manifest.get("run_id") != expected_run_id or manifest.get("experiment_fingerprint") != expected_fingerprint:
                checks.append(_check("identity", FAIL, "manifest experiment identity mismatch"))
            else:
                checks.append(_check("identity", PASS, "manifest experiment identity matches"))
    except (KeyError, EvidenceError, OSError, TypeError) as exc:
        checks.append(_check("data", FAIL, str(exc)))

    selection_path = destination / "checkpoint_selection.json"
    if manifest.get("selection", {}).get("checkpoint") is None:
        checks.append(_check("checkpoint_provenance", NOT_APPLICABLE, "no checkpoint selection recorded"))
    elif not selection_path.is_file():
        checks.append(_check("checkpoint_provenance", FAIL, "checkpoint-selection record is missing"))
    else:
        try:
            selection = _read_json(selection_path)
            if selection.get("checkpoint") != manifest["selection"].get("checkpoint") or selection.get("source") != "validation":
                checks.append(_check("checkpoint_provenance", FAIL, "checkpoint-selection record mismatch"))
            else:
                checks.append(_check("checkpoint_provenance", PASS, "checkpoint provenance is validation-only"))
        except (OSError, TypeError) as exc:
            checks.append(_check("checkpoint_provenance", FAIL, str(exc)))

    _verify_artifacts(destination, manifest, checks)
    scorer_version = None
    if resolved_contract is not None:
        scorer_version = resolved_contract.evaluation.scorer_version
    _verify_predictions(destination, checks, scorer_version=scorer_version)
    return _report(run_dir, offline, checks)


def _report(run_dir: str | Path, offline: bool, checks: list[dict[str, Any]]) -> dict[str, Any]:
    checks = sorted(checks, key=lambda check: check["name"])
    failed = sum(check["status"] == FAIL for check in checks)
    return {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "offline": offline,
        "checks": checks,
        "summary": {
            "status": FAIL if failed else PASS,
            "verified": failed == 0,
            "message": "evidence bundle verified" if failed == 0 else "evidence verification failed",
            "failures": failed,
        },
    }


def verify(run_dir: str | Path, *, offline: bool = True) -> dict[str, Any]:
    report = verification_report(run_dir, offline=offline)
    if report["summary"]["status"] == FAIL:
        raise VerificationFailure(report)
    return report


def verification_exit_code(report: Mapping[str, Any]) -> int:
    return 4 if report.get("summary", {}).get("status") == FAIL else 0


def verification_json(report: Mapping[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def render_verification_report(report: Mapping[str, Any]) -> str:
    lines = ["CauseTune Verification", ""]
    for check in report.get("checks", []):
        lines.append(f"{check['status']} {check['message']}")
    lines.extend(["", report["summary"]["status"]])
    return "\n".join(lines) + "\n"


def write_verification_report(report: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(verification_json(report), encoding="utf-8")
