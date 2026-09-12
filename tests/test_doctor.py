from __future__ import annotations

import json
from pathlib import Path

import pytest

import causetune.doctor as doctor_module
from causetune.doctor import (
    Check,
    DoctorFailure,
    doctor,
    doctor_exit_code,
    doctor_json,
    doctor_report,
    environment_snapshot,
    render_doctor_report,
    write_doctor_report,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _doctor_config(tmp_path: Path, *, same_benchmark: bool = False) -> Path:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    benchmark = train if same_benchmark else tmp_path / "benchmark.jsonl"
    row = {
        "messages": [
            {"role": "user", "content": "diagnose this"},
            {"role": "assistant", "content": "{\"label\":\"ok\"}"},
        ]
    }
    _write_jsonl(train, [row])
    _write_jsonl(validation, [row | {"id": "validation"}])
    if not same_benchmark:
        _write_jsonl(benchmark, [{"id": "benchmark", "input": "sealed"}])
    config = {
        "schema_version": 1,
        "experiment_id": "doctor-test",
        "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
        "data": {"train": str(train), "validation": str(validation), "benchmark": str(benchmark)},
        "training": {"seed": 42},
        "evaluation": {"contract_version": "test-v1", "scorer_version": "test-scorer-v1"},
        "output": {"output_dir": str(tmp_path / "run")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_ready_doctor_is_cpu_safe_and_deterministic(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    first = doctor_report(config)
    second = doctor_report(config)

    assert first == second
    assert first["summary"] == {
        "status": "WARN",
        "ready": True,
        "message": "ready",
        "counts": {"PASS": first["summary"]["counts"]["PASS"], "WARN": 1, "FAIL": 0},
    }
    assert first["hardware_requested"] is False
    assert all(check["status"] != "FAIL" for check in first["checks"])


def test_blocking_data_problem_is_nonzero_failure(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path, same_benchmark=True)
    report = doctor_report(config)
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["ready"] is False
    with pytest.raises(DoctorFailure) as error:
        doctor(config)
    assert error.value.report == report
    assert doctor_exit_code(report) == 3


def test_missing_supervision_is_blocking(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    train = tmp_path / "train.jsonl"
    _write_jsonl(train, [{"input": "no assistant target"}])
    report = doctor_report(config)
    assert any(
        check["name"] == "train_supervision" and check["status"] == "FAIL"
        for check in report["checks"]
    )


def test_report_json_and_human_rendering_are_stable(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    report = doctor_report(config)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_doctor_report(report, first)
    write_doctor_report(report, second)
    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")
    human = render_doctor_report(report)
    assert human.startswith("CauseTune Doctor\n")
    assert human.endswith("WARN\n")
    assert doctor_exit_code(report) == 0
    assert doctor_json(report).endswith("\n")


def test_hardware_is_explicitly_opt_in(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    report = doctor_report(config)
    assert not any(check["layer"] == "Hardware" for check in report["checks"])
    hardware_report = doctor_report(config, hardware=True)
    assert hardware_report["hardware_requested"] is True


def test_warning_is_not_reported_as_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    monkeypatch.setattr(
        doctor_module,
        "_check_hardware",
        lambda: [Check("Hardware", "cuda", "WARN", "CUDA is not visible")],
    )
    report = doctor_report(config, hardware=True)
    assert report["summary"]["status"] == "WARN"
    assert report["summary"]["ready"] is True
    assert doctor_exit_code(report) == 0


def test_invalid_contract_is_reported_without_duplicate_validation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["training"]["unknown"] = True
    config.write_text(json.dumps(raw), encoding="utf-8")
    report = doctor_report(config)
    assert report["summary"]["status"] == "FAIL"
    assert report["summary"]["counts"]["FAIL"] == 1
    assert report["checks"][0]["layer"] == "Contract"


def test_default_environment_snapshot_does_not_import_torch() -> None:
    snapshot = environment_snapshot()
    assert snapshot["torch"]["imported"] is False
    assert "python" in snapshot
    assert "packages" in snapshot


def test_invalid_model_identifier_is_blocking(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["model"]["model_id"] = "not a repository id"
    config.write_text(json.dumps(raw), encoding="utf-8")
    report = doctor_report(config)
    assert report["summary"]["status"] == "FAIL"
    assert any(
        check["layer"] == "Model" and check["name"] == "model_identifier" and check["status"] == "FAIL"
        for check in report["checks"]
    )


def test_incident_directory_roles_recognize_separate_ground_truth_supervision(tmp_path: Path) -> None:
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    benchmark = tmp_path / "benchmark.jsonl"
    train.mkdir()
    validation.mkdir()
    incident = {"incident_id": "i-1", "incident_packet": "packet", "metadata": {"present_components": ["svc"]}}
    truth = {"incident_id": "i-1", "culprit_service": "svc", "failure_mode": "memory_leak", "recommended_action": "restart_or_replace_instance"}
    _write_jsonl(train / "train.jsonl", [incident])
    _write_jsonl(train / "ground_truth_train.jsonl", [truth])
    _write_jsonl(validation / "validation.jsonl", [incident | {"incident_id": "i-2"}])
    _write_jsonl(validation / "ground_truth_validation.jsonl", [truth | {"incident_id": "i-2"}])
    _write_jsonl(benchmark, [{"input": "sealed"}])
    config = tmp_path / "incident.json"
    config.write_text(json.dumps({
        "experiment_id": "incident-doctor",
        "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
        "data": {"train": str(train), "validation": str(validation), "benchmark": str(benchmark)},
        "training": {"seed": 42},
        "output": {"output_dir": str(tmp_path / "run")},
    }), encoding="utf-8")
    report = doctor_report(config)
    assert not any(check["status"] == "FAIL" for check in report["checks"] if check["name"].endswith("supervision"))
