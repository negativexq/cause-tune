from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.evidence import finalize_evidence, initialize_evidence, record_checkpoint_selection, record_training_result
from causetune.experiment_contract import resolve_experiment_config
from causetune.verify import VerificationFailure, verification_exit_code, verification_report, verify


def _contract(tmp_path: Path):
    paths = {}
    for role, content in {
        "train": '{"messages":[{"role":"assistant","content":"train"}]}\n',
        "validation": '{"messages":[{"role":"assistant","content":"validation"}]}\n',
        "benchmark": '{"input":"sealed"}\n',
    }.items():
        path = tmp_path / f"{role}.jsonl"
        path.write_text(content, encoding="utf-8")
        paths[role] = str(path)
    return resolve_experiment_config(
        {
            "experiment_id": "verify-test",
            "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
            "data": paths,
            "training": {"seed": 42},
            "output": {"output_dir": str(tmp_path / "run")},
        }
    )


def _bundle(tmp_path: Path, *, predictions: bool = True) -> Path:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    initialize_evidence(contract, run_dir)
    record_checkpoint_selection(run_dir, checkpoint=2)
    record_training_result(run_dir, actual_steps=2, stop_reason="completed")
    if predictions:
        rows = [{"id": "a", "expected": "ok", "prediction": "ok"}, {"id": "b", "expected": "bad", "prediction": "ok"}]
        (run_dir / "predictions.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
        )
        (run_dir / "evaluation.json").write_text(
            json.dumps(
                {
                    "metrics": {
                        "scorer_version": "causetune-exact-match-v1",
                        "count": 2,
                        "correct": 1,
                        "exact_match": 0.5,
                    }
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    finalize_evidence(run_dir)
    return run_dir


def test_clean_bundle_verifies_and_reproduces_metrics(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    report = verify(run_dir, offline=True)
    assert report["summary"]["verified"] is True
    assert any(check["name"] == "evaluation_reproduction" and check["status"] == "PASS" for check in report["checks"])
    assert verification_exit_code(report) == 0


@pytest.mark.parametrize("relative", ["predictions.jsonl", "resolved_config.json", "evaluation.json"])
def test_one_byte_artifact_tampering_fails(tmp_path: Path, relative: str) -> None:
    run_dir = _bundle(tmp_path)
    path = run_dir / relative
    data = bytearray(path.read_bytes())
    data[-1] = data[-1] ^ 1
    path.write_bytes(bytes(data))
    report = verification_report(run_dir)
    assert report["summary"]["status"] == "FAIL"
    with pytest.raises(VerificationFailure):
        verify(run_dir)


def test_benchmark_tampering_fails_without_repair(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    benchmark = tmp_path / "benchmark.jsonl"
    before = (run_dir / "manifest.json").read_bytes()
    benchmark.write_text('{"input":"changed"}\n', encoding="utf-8")
    report = verification_report(run_dir)
    assert any(check["name"] == "data_benchmark" and check["status"] == "FAIL" for check in report["checks"])
    assert (run_dir / "manifest.json").read_bytes() == before


def test_manifest_identity_tampering_fails(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["run_id"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    report = verification_report(run_dir)
    assert any(check["name"] == "identity" and check["status"] == "FAIL" for check in report["checks"])


def test_optional_predictions_are_not_reported_as_pass(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path, predictions=False)
    report = verify(run_dir)
    check = next(check for check in report["checks"] if check["name"] == "evaluation_reproduction")
    assert check["status"] == "NOT_APPLICABLE"


def test_missing_required_artifact_fails(tmp_path: Path) -> None:
    run_dir = _bundle(tmp_path)
    (run_dir / "environment.json").unlink()
    report = verification_report(run_dir)
    assert report["summary"]["status"] == "FAIL"
    assert verification_exit_code(report) == 4
