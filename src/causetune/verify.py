"""Offline verification of CauseTune evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence import EvidenceError, artifact_hashes, canonical_json, load_manifest, sha256_file, sha256_path
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
    for row in rows:
        expected = value(row, ("expected", "target"))
        prediction = value(row, ("prediction", "output"))
        if expected == prediction:
            correct += 1
    return {
        "scorer_version": "causetune-exact-match-v1",
        "count": len(rows),
        "correct": correct,
        "exact_match": correct / len(rows),
    }


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


def _verify_predictions(destination: Path, checks: list[dict[str, Any]]) -> None:
    predictions = destination / "predictions.jsonl"
    evaluation = destination / "evaluation.json"
    if not predictions.exists() and not evaluation.exists():
        checks.append(_check("evaluation_reproduction", NOT_APPLICABLE, "no persisted predictions/evaluation artifacts"))
        return
    if not predictions.is_file() or not evaluation.is_file():
        checks.append(_check("evaluation_reproduction", FAIL, "predictions and evaluation must be persisted together"))
        return
    try:
        recomputed = score_predictions(predictions)
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

    try:
        resolved = resolve_experiment_config(_read_json(destination / "resolved_config.json"))
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
    _verify_predictions(destination, checks)
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
