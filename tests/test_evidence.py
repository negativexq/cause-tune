from __future__ import annotations

import json
from pathlib import Path

import pytest

from causetune.evidence import (
    EvidenceError,
    artifact_hashes,
    finalize_evidence,
    initialize_evidence,
    load_manifest,
    record_checkpoint_selection,
    record_training_result,
    sha256_file,
)
from causetune.experiment_contract import resolve_experiment_config


def _contract(tmp_path: Path):
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    benchmark = tmp_path / "benchmark.jsonl"
    train.write_text('{"messages":[{"role":"assistant","content":"train"}]}\n', encoding="utf-8")
    validation.write_text('{"messages":[{"role":"assistant","content":"validation"}]}\n', encoding="utf-8")
    benchmark.write_text('{"input":"sealed"}\n', encoding="utf-8")
    return resolve_experiment_config(
        {
            "experiment_id": "evidence-test",
            "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
            "data": {"train": str(train), "validation": str(validation), "benchmark": str(benchmark)},
            "training": {"seed": 42},
            "output": {"output_dir": str(tmp_path / "run")},
        }
    )


def test_run_identity_is_stable_and_paths_do_not_change_it(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    first = initialize_evidence(contract, tmp_path / "one", git_cwd=".")
    second = initialize_evidence(contract, tmp_path / "two", git_cwd=".")
    assert first["run_id"] == second["run_id"]
    assert first["experiment_fingerprint"] == second["experiment_fingerprint"]


def test_run_identity_uses_data_content_not_dataset_paths(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    for role, content in {
        "train": '{"messages":[{"role":"assistant","content":"train"}]}\n',
        "validation": '{"messages":[{"role":"assistant","content":"validation"}]}\n',
        "benchmark": '{"input":"sealed"}\n',
    }.items():
        (first_root / f"{role}.jsonl").write_text(content, encoding="utf-8")
        (second_root / f"{role}.jsonl").write_text(content, encoding="utf-8")

    def make_contract(root: Path):
        return resolve_experiment_config(
            {
                "experiment_id": "path-independent",
                "model": {"model_id": "Qwen/Qwen3-4B", "revision": "commit-test"},
                "data": {
                    "train": str(root / "train.jsonl"),
                    "validation": str(root / "validation.jsonl"),
                    "benchmark": str(root / "benchmark.jsonl"),
                },
                "training": {"seed": 42},
                "output": {"output_dir": str(root / "run")},
            }
        )

    first = initialize_evidence(make_contract(first_root), first_root / "run")
    second = initialize_evidence(make_contract(second_root), second_root / "run")
    assert first["run_id"] == second["run_id"]
    assert first["experiment_fingerprint"] == second["experiment_fingerprint"]


def test_manifest_captures_provenance_and_finalize_hashes_artifacts(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    manifest = initialize_evidence(contract, run_dir, git_cwd=".")
    assert manifest["schema_version"] == 1
    assert set(manifest["data"]) == {"train", "validation", "benchmark"}
    assert "commit" in manifest["git"] or manifest["git"]["available"] is False
    assert (run_dir / "resolved_config.json").is_file()
    assert json.loads((run_dir / "environment.json").read_text())

    record_checkpoint_selection(run_dir, checkpoint=100)
    record_training_result(run_dir, actual_steps=100, stop_reason="validation_no_improvement")
    (run_dir / "predictions.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    finalized = finalize_evidence(run_dir)
    assert finalized["selection"] == {"checkpoint": 100, "source": "validation"}
    assert finalized["training"]["actual_steps"] == 100
    assert finalized["artifacts"]["predictions.jsonl"] == sha256_file(run_dir / "predictions.jsonl")
    assert "manifest.json" not in finalized["artifacts"]
    assert load_manifest(run_dir) == finalized
    assert finalized["config"]["sha256"] == contract.sha256()
    assert finalize_evidence(run_dir)["artifacts"] == finalized["artifacts"]
    before = artifact_hashes(run_dir)
    (run_dir / "predictions.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")
    after = artifact_hashes(run_dir)
    assert before["predictions.jsonl"] != after["predictions.jsonl"]


def test_finalized_bundle_is_immutable_and_repeatable(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    initialize_evidence(contract, run_dir)
    record_checkpoint_selection(run_dir, checkpoint=1)
    record_training_result(run_dir, actual_steps=1, stop_reason="completed")
    (run_dir / "predictions.jsonl").write_text('{"id":"a"}\n', encoding="utf-8")
    first = finalize_evidence(run_dir)
    assert finalize_evidence(run_dir) == first
    (run_dir / "predictions.jsonl").write_text('{"id":"changed"}\n', encoding="utf-8")
    with pytest.raises(EvidenceError, match="immutable"):
        finalize_evidence(run_dir)


def test_checkpoint_provenance_cannot_use_benchmark(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    initialize_evidence(contract, run_dir)
    with pytest.raises(EvidenceError):
        record_checkpoint_selection(run_dir, checkpoint=1, source="benchmark")


def test_completed_training_requires_checkpoint_provenance(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    initialize_evidence(contract, run_dir)
    record_training_result(run_dir, actual_steps=1, stop_reason="completed")
    with pytest.raises(EvidenceError, match="checkpoint_selection"):
        finalize_evidence(run_dir)


def test_non_empty_run_directory_is_not_overwritten(tmp_path: Path) -> None:
    contract = _contract(tmp_path)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "user-file").write_text("keep", encoding="utf-8")
    with pytest.raises(EvidenceError):
        initialize_evidence(contract, run_dir)
