from __future__ import annotations

import json
from pathlib import Path

from causetune.cli import EXIT_CONTRACT, EXIT_PREFLIGHT, EXIT_SUCCESS, EXIT_VERIFICATION, main


def _config(tmp_path: Path) -> Path:
    paths = {}
    for role, row in {
        "train": {"messages": [{"role": "assistant", "content": "train"}]},
        "validation": {"messages": [{"role": "assistant", "content": "validation"}]},
        "benchmark": {"input": "sealed"},
    }.items():
        path = tmp_path / f"{role}.jsonl"
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths[role] = str(path)
    config = {
        "schema_version": 1,
        "experiment_id": "cli-test",
        "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
        "data": paths,
        "training": {"seed": 42},
        "output": {"output_dir": str(tmp_path / "run")},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_doctor_json_routes_to_application_api(tmp_path: Path, capsys) -> None:
    code = main(["doctor", "--config", str(_config(tmp_path)), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert code == EXIT_SUCCESS
    assert output["summary"]["ready"] is True


def test_train_initializes_evidence_bundle_without_training(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    code = main(["train", "--config", str(config), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert code == EXIT_SUCCESS
    assert output["execution"] == "not_started"
    run_dir = Path(output["run_dir"])
    assert (run_dir / "manifest.json").is_file()
    assert (run_dir / "resolved_config.json").is_file()


def test_invalid_contract_has_contract_exit_code(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["unknown"] = True
    config.write_text(json.dumps(raw), encoding="utf-8")
    code = main(["doctor", "--config", str(config), "--json"])
    assert code == EXIT_CONTRACT


def test_missing_data_is_preflight_exit_code(tmp_path: Path, capsys) -> None:
    config = _config(tmp_path)
    raw = json.loads(config.read_text(encoding="utf-8"))
    raw["data"]["benchmark"] = str(tmp_path / "missing.jsonl")
    config.write_text(json.dumps(raw), encoding="utf-8")
    code = main(["doctor", "--config", str(config), "--json"])
    assert code == EXIT_PREFLIGHT


def test_evaluate_and_compare_use_persisted_files(tmp_path: Path, capsys) -> None:
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        '{"expected":"a","prediction":"a"}\n{"expected":"b","prediction":"a"}\n',
        encoding="utf-8",
    )
    base = tmp_path / "base.json"
    tuned = tmp_path / "tuned.json"
    assert main(["evaluate", "--predictions", str(predictions), "--output", str(base), "--json"]) == EXIT_SUCCESS
    capsys.readouterr()
    tuned.write_text(json.dumps({"metrics": {"exact_match": 1.0, "count": 2}}), encoding="utf-8")
    result = main(["compare", "--base", str(base), "--tuned", str(tuned), "--json"])
    output = json.loads(capsys.readouterr().out)
    assert result == EXIT_SUCCESS
    assert output["delta"]["exact_match"] == 0.5


def test_verify_failure_has_verification_exit_code(tmp_path: Path, capsys) -> None:
    code = main(["verify", str(tmp_path / "missing-run"), "--json"])
    assert code == EXIT_VERIFICATION
