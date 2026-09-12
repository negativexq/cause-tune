"""CPU-safe experiment preflight checks for CauseTune.

The default doctor validates a contract and inspects data without importing a
model runtime or allocating CUDA memory. Hardware diagnostics are an explicit,
opt-in mode because they are environment observations rather than experiment
contract validation.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from .experiment_contract import ExperimentContract, ExperimentContractError, load_experiment_contract


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
_MODEL_ID = re.compile(r"^[^\s/]+/[^\s/]+$")
_REQUIRED_PACKAGES = ("torch", "transformers", "peft")


class DoctorFailure(RuntimeError):
    """Raised when one or more blocking doctor checks fail."""

    def __init__(self, report: dict[str, Any]):
        self.report = report
        super().__init__(report["summary"]["message"])


@dataclass(frozen=True)
class Check:
    layer: str
    name: str
    status: str
    message: str
    details: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "layer": self.layer,
            "name": self.name,
            "status": self.status,
            "message": self.message,
        }
        if self.details:
            result["details"] = _sort_json(self.details)
        return result


def _sort_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sort_json(value[key]) for key in sorted(value, key=str)}
    if isinstance(value, (list, tuple)):
        return [_sort_json(item) for item in value]
    return value


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
    if path.is_dir():
        for child in sorted(item for item in path.rglob("*") if item.is_file()):
            relative = child.relative_to(path).as_posix().encode("utf-8")
            digest.update(len(relative).to_bytes(8, "big"))
            digest.update(relative)
            digest.update(bytes.fromhex(_sha256_path(child)))
        return digest.hexdigest()
    raise FileNotFoundError(path)


def _read_records(path: Path) -> tuple[list[dict[str, Any]], str | None]:
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file())
    records: list[dict[str, Any]] = []
    for file_path in files:
        if file_path.suffix.lower() not in {".jsonl", ".ndjson", ".json"}:
            continue
        try:
            if file_path.suffix.lower() == ".json":
                value = json.loads(file_path.read_text(encoding="utf-8"))
                if isinstance(value, list):
                    candidates = value
                elif isinstance(value, dict):
                    candidates = [value]
                else:
                    return [], f"{file_path}: root must be an object or array"
            else:
                candidates = []
                for line_number, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.strip():
                        continue
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        return [], f"{file_path}:{line_number}: record must be an object"
                    candidates.append(value)
            if any(not isinstance(item, dict) for item in candidates):
                return [], f"{file_path}: every record must be an object"
            records.extend(candidates)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return [], f"{file_path}: {exc}"
    return records, None


def _assistant_tokens(record: Mapping[str, Any]) -> int:
    messages = record.get("messages")
    if isinstance(messages, list):
        return sum(
            len(str(message.get("content", "")).strip())
            for message in messages
            if isinstance(message, Mapping) and message.get("role") == "assistant"
        )
    for key in ("assistant", "output", "response", "completion", "target"):
        value = record.get(key)
        if isinstance(value, str):
            return len(value.strip())
        if value is not None:
            return len(str(value).strip())
    # Incident diagnosis training keeps the model-visible packet and the
    # target in separate files. Ground-truth rows are still explicit
    # supervision evidence for a directory-backed training role.
    if all(key in record for key in ("failure_mode", "recommended_action", "culprit_service")):
        return len(json.dumps(record, ensure_ascii=False, sort_keys=True))
    return 0


def _check_contract(config_path: str | Path) -> tuple[ExperimentContract | None, list[Check]]:
    try:
        contract = load_experiment_contract(config_path)
    except (ExperimentContractError, OSError) as exc:
        return None, [Check("Contract", "contract", FAIL, str(exc))]
    try:
        output_path = Path(contract.output.output_dir)
        output_valid = "\x00" not in contract.output.output_dir and not output_path.exists() or output_path.is_dir()
    except (OSError, ValueError):
        output_valid = False
    checks = [
        Check("Contract", "schema", PASS, "experiment schema valid"),
        Check("Contract", "experiment_id", PASS, f"experiment id: {contract.experiment_id}"),
        Check("Contract", "model_revision", PASS, f"model revision policy: {contract.model.revision_policy}"),
        Check(
            "Contract",
            "output_path",
            PASS if output_valid else FAIL,
            "output path valid" if output_valid else "output path is invalid or points to a file",
        ),
    ]
    return contract, checks


def _check_data(contract: ExperimentContract) -> list[Check]:
    checks: list[Check] = []
    hashes: dict[str, str] = {}
    records_by_role: dict[str, list[dict[str, Any]]] = {}
    for role in ("train", "validation", "benchmark"):
        data_path = Path(contract.data[role].path)
        if not data_path.exists():
            checks.append(Check("Data", f"{role}_exists", FAIL, f"{role} does not exist: {data_path}"))
            continue
        try:
            fingerprint = _sha256_path(data_path)
        except OSError as exc:
            checks.append(Check("Data", f"{role}_fingerprint", FAIL, str(exc)))
            continue
        hashes[role] = fingerprint
        checks.append(
            Check(
                "Data",
                f"{role}_fingerprint",
                PASS,
                f"{role} fingerprint computable",
                {"sha256": fingerprint},
            )
        )
        records, error = _read_records(data_path)
        if error:
            checks.append(Check("Data", f"{role}_schema", FAIL, error))
            continue
        records_by_role[role] = records
        if not records:
            checks.append(Check("Data", f"{role}_non_empty", FAIL, f"{role} dataset is empty"))
        else:
            checks.append(Check("Data", f"{role}_non_empty", PASS, f"{role} dataset is non-empty"))

        if role in {"train", "validation"}:
            supervised = sum(_assistant_tokens(record) for record in records)
            if supervised <= 0:
                checks.append(Check("Data", f"{role}_supervision", FAIL, f"{role} has no supervised assistant tokens"))
            else:
                checks.append(
                    Check(
                        "Data",
                        f"{role}_supervision",
                        PASS,
                        f"{role} supervised assistant tokens present",
                        {"records": len(records), "non_empty_supervised_records": sum(_assistant_tokens(record) > 0 for record in records)},
                    )
                )
        else:
            checks.append(Check("Data", "benchmark_schema", PASS, "benchmark records readable"))

    if len(hashes) == 3:
        for left, right in (("train", "validation"), ("train", "benchmark"), ("validation", "benchmark")):
            if hashes[left] == hashes[right]:
                checks.append(Check("Data", f"roles_{left}_{right}", FAIL, f"{left} and {right} contents collide"))
            else:
                checks.append(Check("Data", f"roles_{left}_{right}", PASS, f"{left} and {right} roles are distinct"))

    return checks


def _check_training(contract: ExperimentContract) -> list[Check]:
    return [
        Check("Training", "seed", PASS, "deterministic seed", {"seed": contract.training.seed}),
        Check(
            "Training",
            "quantization",
            PASS,
            f"{contract.training.quantization.quant_type.upper()} / {contract.training.quantization.compute_dtype.upper()}",
        ),
        Check("Training", "lora_targets", PASS, "LoRA target policy valid", {"targets": list(contract.training.lora.target_modules)}),
        Check("Training", "checkpoint_selection", PASS, "validation-only checkpoint selection"),
        Check("Training", "preprocessing", PASS, "deterministic preprocessing configuration"),
    ]


def _check_model(contract: ExperimentContract) -> list[Check]:
    """Validate model identity without downloading weights or constructing a model."""

    model_id = contract.model.model_id
    local_path = Path(model_id)
    is_local = local_path.exists()
    identifier_valid = bool(_MODEL_ID.fullmatch(model_id)) or is_local
    checks = [
        Check(
            "Model",
            "model_identifier",
            PASS if identifier_valid else FAIL,
            "model identifier valid" if identifier_valid else "model identifier is not a valid repository id or local path",
            {"model_id": model_id, "local": is_local},
        )
    ]
    if not identifier_valid:
        checks.append(Check("Model", "tokenizer_configuration", FAIL, "tokenizer configuration cannot be resolved from an invalid model identifier"))
    elif is_local:
        tokenizer_markers = (
            "tokenizer.json",
            "tokenizer_config.json",
            "tokenizer.model",
            "spiece.model",
            "vocab.json",
        )
        present = [name for name in tokenizer_markers if (local_path / name).is_file()]
        checks.append(
            Check(
                "Model",
                "tokenizer_configuration",
                PASS if present else FAIL,
                "local tokenizer configuration found" if present else "local model directory has no recognizable tokenizer configuration",
                {"files": present},
            )
        )
    else:
        cached_files: list[str] = []
        try:
            from huggingface_hub import try_to_load_from_cache

            for filename in (
                "tokenizer_config.json",
                "tokenizer.json",
                "tokenizer.model",
                "spiece.model",
                "vocab.json",
                "sentencepiece.bpe.model",
            ):
                cached = try_to_load_from_cache(
                    model_id,
                    filename,
                    revision=contract.model.revision,
                )
                if isinstance(cached, str) and Path(cached).is_file():
                    cached_files.append(filename)
        except Exception:
            # Cache inspection is advisory. A missing optional cache helper
            # must not make the CPU-safe doctor import or load model assets.
            cached_files = []

        # Remote resolution is intentionally limited to cache/config metadata:
        # the default doctor must not download model or tokenizer assets.
        checks.append(
            Check(
                "Model",
                "tokenizer_configuration",
                PASS if cached_files else WARN,
                "cached remote tokenizer configuration found"
                if cached_files
                else "remote tokenizer configuration is not locally resolved; no model assets downloaded",
                {"model_id": model_id, "resolution": "cache", "files": cached_files},
            )
        )
    return checks


def _check_evaluation(contract: ExperimentContract) -> list[Check]:
    return [
        Check("Evaluation", "contract", PASS, "evaluation contract resolved", contract.evaluation.to_dict()),
        Check("Evaluation", "isolation", PASS, "frozen benchmark is outside checkpoint selection"),
    ]


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _check_environment() -> list[Check]:
    versions = {package: _package_version(package) for package in _REQUIRED_PACKAGES}
    missing = sorted(package for package, version in versions.items() if version is None)
    if missing:
        return [Check("Environment", "python_packages", FAIL, "required packages missing", {"missing": missing})]
    return [
        Check("Environment", "python", PASS, f"Python {sys.version_info.major}.{sys.version_info.minor}"),
        Check("Environment", "python_packages", PASS, "torch / transformers / peft available", versions),
    ]


def _check_hardware() -> list[Check]:
    try:
        import torch
    except Exception as exc:
        return [Check("Hardware", "torch_import", FAIL, f"torch import failed: {exc}")]
    checks = [Check("Hardware", "torch_import", PASS, "torch import succeeded")]
    if not torch.cuda.is_available():
        checks.append(Check("Hardware", "cuda", WARN, "CUDA is not visible"))
        return checks
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    checks.extend(
        [
            Check("Hardware", "cuda", PASS, "CUDA visible"),
            Check(
                "Hardware",
                "gpu",
                PASS,
                properties.name,
                {"index": device, "compute_capability": [properties.major, properties.minor], "vram_bytes": properties.total_memory},
            ),
            Check("Hardware", "bf16", PASS if torch.cuda.is_bf16_supported() else WARN, "BF16 support inspected"),
        ]
    )
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        checks.append(Check("Hardware", "bitsandbytes", FAIL, f"bitsandbytes import failed: {exc}"))
    else:
        checks.append(Check("Hardware", "bitsandbytes", PASS, "bitsandbytes import succeeded"))
    return checks


def environment_snapshot(*, hardware: bool = False) -> dict[str, Any]:
    """Return useful environment facts without touching CUDA by default."""

    snapshot: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {package: _package_version(package) for package in (*_REQUIRED_PACKAGES, "bitsandbytes")},
        "torch": {"imported": False, "cuda_version": None, "gpu": None, "vram_bytes": None},
    }
    if not hardware:
        return snapshot
    try:
        import torch
    except Exception as exc:
        snapshot["torch"] = {"imported": False, "error": str(exc)}
        return snapshot
    torch_info: dict[str, Any] = {"imported": True, "cuda_version": torch.version.cuda, "gpu": None, "vram_bytes": None}
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device)
        torch_info.update({"gpu": properties.name, "vram_bytes": properties.total_memory})
    snapshot["torch"] = torch_info
    return snapshot


def doctor_report(config_path: str | Path, *, hardware: bool = False) -> dict[str, Any]:
    """Return a deterministic machine-readable doctor report."""

    contract, checks = _check_contract(config_path)
    if contract is not None:
        checks.extend(_check_data(contract))
        checks.extend(_check_model(contract))
        checks.extend(_check_training(contract))
        checks.extend(_check_evaluation(contract))
        checks.extend(_check_environment())
        if hardware:
            checks.extend(_check_hardware())
    counts = {status: sum(check.status == status for check in checks) for status in (PASS, WARN, FAIL)}
    blocking = counts[FAIL] > 0
    checks = sorted(checks, key=lambda item: (item.layer, item.name))
    return {
        "schema_version": 1,
        "config": str(config_path),
        "hardware_requested": hardware,
        "environment": environment_snapshot(hardware=hardware),
        "checks": [check.to_dict() for check in checks],
        "summary": {
            "status": FAIL if blocking else (WARN if counts[WARN] else PASS),
            "ready": not blocking,
            "message": "blocking preflight failures detected" if blocking else "ready",
            "counts": counts,
        },
    }


def doctor(config_path: str | Path, *, hardware: bool = False) -> dict[str, Any]:
    report = doctor_report(config_path, hardware=hardware)
    if report["summary"]["status"] == FAIL:
        raise DoctorFailure(report)
    return report


def render_doctor_report(report: Mapping[str, Any]) -> str:
    """Render the same report for a human without changing check semantics."""

    lines = ["CauseTune Doctor", ""]
    last_layer = None
    for check in report["checks"]:
        layer = check["layer"]
        if layer != last_layer:
            if last_layer is not None:
                lines.append("")
            lines.append(layer)
            last_layer = layer
        lines.append(f"{check['status']} {check['message']}")
    lines.extend(["", report["summary"]["status"]])
    return "\n".join(lines) + "\n"


def write_doctor_report(report: Mapping[str, Any], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(doctor_json(report), encoding="utf-8")


def doctor_json(report: Mapping[str, Any]) -> str:
    """Serialize a doctor report for scripts with stable key ordering."""

    return json.dumps(_sort_json(report), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"


def doctor_exit_code(report: Mapping[str, Any]) -> int:
    """Return the M9 process result: zero for ready PASS/WARN, three for FAIL."""

    return 3 if report.get("summary", {}).get("status") == FAIL else 0
