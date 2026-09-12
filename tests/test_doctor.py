from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.doctor import (
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
        "status": "PASS",
        "ready": True,
        "message": "ready",
        "counts": {"PASS": first["summary"]["counts"]["PASS"], "WARN": 0, "FAIL": 0},
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
    assert human.endswith("PASS\n")
    assert doctor_exit_code(report) == 0
    assert doctor_json(report).endswith("\n")


def test_hardware_is_explicitly_opt_in(tmp_path: Path) -> None:
    config = _doctor_config(tmp_path)
    report = doctor_report(config)
    assert not any(check["layer"] == "Hardware" for check in report["checks"])
    hardware_report = doctor_report(config, hardware=True)
    assert hardware_report["hardware_requested"] is True


def test_default_environment_snapshot_does_not_import_torch() -> None:
    snapshot = environment_snapshot()
    assert snapshot["torch"]["imported"] is False
    assert "python" in snapshot
    assert "packages" in snapshot
