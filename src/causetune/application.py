"""Application-level workflows shared by the thin CLI and scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .doctor import DoctorFailure, doctor_report
from .evidence import EvidenceError, finalize_evidence, initialize_evidence
from .experiment_contract import ExperimentContractError, load_experiment_contract
from .verify import score_predictions, verification_report


def doctor_workflow(config_path: str | Path, *, hardware: bool = False) -> dict[str, Any]:
    return doctor_report(config_path, hardware=hardware)


def prepare_train_workflow(
    config_path: str | Path,
    *,
    run_dir: str | Path | None = None,
    hardware: bool = False,
) -> dict[str, Any]:
    """Validate and initialize an evidence-backed run before training.

    The current M13 surface deliberately prepares the run and does not start a
    scientific GPU execution. E03 supplies the first authorized runner.
    """

    report = doctor_report(config_path, hardware=hardware)
    if report["summary"]["status"] == "FAIL":
        raise DoctorFailure(report)
    contract = load_experiment_contract(config_path)
    destination = Path(run_dir or contract.output.output_dir)
    manifest = initialize_evidence(contract, destination, hardware_snapshot=hardware)
    return {
        "status": "prepared",
        "execution": "not_started",
        "run_dir": str(destination),
        "run_id": manifest["run_id"],
        "doctor": report,
    }


def evaluate_workflow(predictions_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Persist deterministic aggregate metrics from already persisted predictions."""

    metrics = score_predictions(predictions_path)
    result = {"schema_version": 1, "metrics": metrics, "predictions": str(predictions_path)}
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


def compare_workflow(base_path: str | Path, tuned_path: str | Path) -> dict[str, Any]:
    """Compare persisted evaluation files without regenerating model output."""

    def read(path: str | Path) -> Mapping[str, Any]:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value.get("metrics", value) if isinstance(value, Mapping) else {}

    base = read(base_path)
    tuned = read(tuned_path)
    delta: dict[str, float] = {}
    for key in sorted(set(base) & set(tuned)):
        if isinstance(base[key], (int, float)) and isinstance(tuned[key], (int, float)):
            delta[key] = float(tuned[key]) - float(base[key])
    return {"base": dict(base), "tuned": dict(tuned), "delta": delta}


def verify_workflow(run_dir: str | Path, *, offline: bool = True) -> dict[str, Any]:
    return verification_report(run_dir, offline=offline)
