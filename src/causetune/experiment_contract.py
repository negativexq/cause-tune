"""Versioned, strict experiment identity for CauseTune runs.

The older :mod:`causetune.config` module remains the compatibility layer for
the original SFT smoke scripts.  This module is the canonical contract for
new experiments: it describes the model, data roles, training policy and
evaluation boundary that make a run scientifically meaningful.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from dataclasses import dataclass, field
from pathlib import PurePath
from pathlib import Path
from typing import Any, Mapping
from types import MappingProxyType


SCHEMA_VERSION = 1
_EXPERIMENT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class ExperimentContractError(ValueError):
    """Raised when a contract is incomplete, ambiguous, or unsafe."""


@dataclass(frozen=True)
class ModelIdentity:
    model_id: str
    revision: str
    revision_policy: str = "pinned"

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ExperimentContractError("model.model_id must not be empty")
        if not isinstance(self.revision, str) or not self.revision.strip():
            raise ExperimentContractError(
                "model.revision is required; a model revision policy may not be implicit"
            )
        if self.revision_policy not in {"pinned", "legacy_unpinned"}:
            raise ExperimentContractError("model.revision_policy must be pinned or legacy_unpinned")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "revision_policy": self.revision_policy,
        }


@dataclass(frozen=True)
class DataRole:
    path: str
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path.strip():
            raise ExperimentContractError("data role path must not be empty")
        if self.fingerprint is not None and (
            not isinstance(self.fingerprint, str) or not self.fingerprint.strip()
        ):
            raise ExperimentContractError("data fingerprint must be a non-empty string when provided")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path}
        if self.fingerprint is not None:
            result["fingerprint"] = self.fingerprint
        return result


@dataclass(frozen=True)
class QuantizationPolicy:
    load_in_4bit: bool = True
    quant_type: str = "nf4"
    compute_dtype: str = "bfloat16"
    double_quant: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.load_in_4bit, bool) or not isinstance(self.double_quant, bool):
            raise ExperimentContractError("quantization boolean fields must be booleans")
        if self.quant_type not in {"nf4", "fp4", "none"}:
            raise ExperimentContractError("unsupported quantization type")
        if self.compute_dtype not in {"bfloat16", "float16", "float32"}:
            raise ExperimentContractError("unsupported quantization compute dtype")
        if self.load_in_4bit and self.quant_type == "none":
            raise ExperimentContractError("4-bit loading requires a quantization type")
        if not self.load_in_4bit and self.double_quant:
            raise ExperimentContractError("double quantization requires 4-bit loading")
        if not self.load_in_4bit and self.quant_type != "none":
            raise ExperimentContractError("quant_type must be none when 4-bit loading is disabled")

    def to_dict(self) -> dict[str, Any]:
        return {
            "load_in_4bit": self.load_in_4bit,
            "quant_type": self.quant_type,
            "compute_dtype": self.compute_dtype,
            "double_quant": self.double_quant,
        }


@dataclass(frozen=True)
class LoraPolicy:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.0
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )

    def __post_init__(self) -> None:
        if not isinstance(self.rank, int) or isinstance(self.rank, bool) or self.rank <= 0:
            raise ExperimentContractError("lora.rank must be a positive integer")
        if not isinstance(self.alpha, int) or isinstance(self.alpha, bool) or self.alpha <= 0:
            raise ExperimentContractError("lora.alpha must be a positive integer")
        if not isinstance(self.dropout, (float, int)) or not 0 <= self.dropout < 1:
            raise ExperimentContractError("lora.dropout must be in [0, 1)")
        if not self.target_modules or any(
            not isinstance(target, str) or not target.strip() for target in self.target_modules
        ):
            raise ExperimentContractError("lora.target_modules must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": list(self.target_modules),
        }


@dataclass(frozen=True)
class OptimizerPolicy:
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_epochs: int = 1
    max_sequence_length: int | None = 1024
    gradient_checkpointing: bool = True
    gradient_checkpointing_use_reentrant: bool = False

    def __post_init__(self) -> None:
        for name in ("micro_batch_size", "gradient_accumulation_steps", "max_epochs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ExperimentContractError(f"training.optimizer.{name} must be a positive integer")
        if self.max_sequence_length is not None and (
            not isinstance(self.max_sequence_length, int)
            or isinstance(self.max_sequence_length, bool)
            or self.max_sequence_length <= 0
        ):
            raise ExperimentContractError(
                "training.optimizer.max_sequence_length must be a positive integer or null"
            )
        if not isinstance(self.learning_rate, (float, int)) or self.learning_rate <= 0:
            raise ExperimentContractError("training.optimizer.learning_rate must be positive")
        if not isinstance(self.gradient_checkpointing, bool) or not isinstance(
            self.gradient_checkpointing_use_reentrant, bool
        ):
            raise ExperimentContractError("gradient checkpointing fields must be booleans")

    def to_dict(self) -> dict[str, Any]:
        return {
            "micro_batch_size": self.micro_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "max_epochs": self.max_epochs,
            "max_sequence_length": self.max_sequence_length,
            "gradient_checkpointing": self.gradient_checkpointing,
            "gradient_checkpointing_use_reentrant": self.gradient_checkpointing_use_reentrant,
        }


@dataclass(frozen=True)
class CheckpointPolicy:
    selection_split: str = "validation"
    validation_only: bool = True
    primary_metric: str = "validation_loss"
    interval_steps: int = 25

    def __post_init__(self) -> None:
        if self.selection_split != "validation":
            raise ExperimentContractError("checkpoint selection must use validation only")
        if self.validation_only is not True:
            raise ExperimentContractError("checkpoint_policy.validation_only must be true")
        if not isinstance(self.primary_metric, str) or not self.primary_metric.strip():
            raise ExperimentContractError("checkpoint_policy.primary_metric must not be empty")
        if not isinstance(self.interval_steps, int) or isinstance(self.interval_steps, bool) or self.interval_steps <= 0:
            raise ExperimentContractError("checkpoint_policy.interval_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_split": self.selection_split,
            "validation_only": self.validation_only,
            "primary_metric": self.primary_metric,
            "interval_steps": self.interval_steps,
        }


@dataclass(frozen=True)
class StoppingPolicy:
    mode: str = "fixed_steps"
    max_steps: int = 1
    patience: int | None = None
    min_delta: float = 0.0
    eval_interval_steps: int = 25

    def __post_init__(self) -> None:
        if self.mode not in {"fixed_steps", "early_stopping"}:
            raise ExperimentContractError("stopping_policy.mode is invalid")
        if not isinstance(self.max_steps, int) or isinstance(self.max_steps, bool) or self.max_steps <= 0:
            raise ExperimentContractError("stopping_policy.max_steps must be positive")
        if self.mode == "early_stopping":
            if self.patience is None or not isinstance(self.patience, int) or self.patience <= 0:
                raise ExperimentContractError("early stopping requires positive patience")
            if self.eval_interval_steps <= 0:
                raise ExperimentContractError("early stopping requires positive eval interval")
        elif self.patience is not None:
            raise ExperimentContractError("fixed-step stopping cannot define patience")
        if not isinstance(self.min_delta, (float, int)) or self.min_delta < 0:
            raise ExperimentContractError("stopping_policy.min_delta must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "max_steps": self.max_steps,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "eval_interval_steps": self.eval_interval_steps,
        }


@dataclass(frozen=True)
class PreprocessingPolicy:
    deterministic: bool = True
    shuffle: bool = True
    seed: int = 42
    version: str = "default-v1"

    def __post_init__(self) -> None:
        if self.deterministic is not True:
            raise ExperimentContractError("preprocessing must be deterministic")
        if not isinstance(self.shuffle, bool):
            raise ExperimentContractError("preprocessing.shuffle must be boolean")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ExperimentContractError("preprocessing.seed must be a non-negative integer")
        if not isinstance(self.version, str) or not self.version.strip():
            raise ExperimentContractError("preprocessing.version must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "deterministic": self.deterministic,
            "shuffle": self.shuffle,
            "seed": self.seed,
            "version": self.version,
        }


@dataclass(frozen=True)
class TrainingPolicy:
    seed: int
    quantization: QuantizationPolicy = field(default_factory=QuantizationPolicy)
    lora: LoraPolicy = field(default_factory=LoraPolicy)
    optimizer: OptimizerPolicy = field(default_factory=OptimizerPolicy)
    checkpoint_policy: CheckpointPolicy = field(default_factory=CheckpointPolicy)
    stopping_policy: StoppingPolicy = field(default_factory=StoppingPolicy)
    preprocessing: PreprocessingPolicy = field(default_factory=PreprocessingPolicy)

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ExperimentContractError("training.seed must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "quantization": self.quantization.to_dict(),
            "lora": self.lora.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "checkpoint_policy": self.checkpoint_policy.to_dict(),
            "stopping_policy": self.stopping_policy.to_dict(),
            "preprocessing": self.preprocessing.to_dict(),
        }


@dataclass(frozen=True)
class EvaluationBoundary:
    contract_version: str = "evaluation-v1"
    scorer_version: str = "scorer-v1"

    def __post_init__(self) -> None:
        if not self.contract_version.strip() or not self.scorer_version.strip():
            raise ExperimentContractError("evaluation contract and scorer versions are required")

    def to_dict(self) -> dict[str, str]:
        return {
            "contract_version": self.contract_version,
            "scorer_version": self.scorer_version,
        }


@dataclass(frozen=True)
class OutputPolicy:
    output_dir: str

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, str) or not self.output_dir.strip():
            raise ExperimentContractError("output.output_dir must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {"output_dir": self.output_dir}


@dataclass(frozen=True)
class ExperimentContract:
    schema_version: int
    experiment_id: str
    model: ModelIdentity
    data: Mapping[str, DataRole]
    training: TrainingPolicy
    evaluation: EvaluationBoundary
    output: OutputPolicy
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ExperimentContractError(f"unsupported contract schema_version: {self.schema_version!r}")
        if not isinstance(self.experiment_id, str) or not _EXPERIMENT_ID.fullmatch(self.experiment_id):
            raise ExperimentContractError("experiment_id must contain only letters, numbers, '.', '_' or '-'")
        if set(self.data) != {"train", "validation", "benchmark"}:
            raise ExperimentContractError("data must define train, validation, and benchmark roles")
        if any(not isinstance(value, DataRole) for value in self.data.values()):
            raise ExperimentContractError("data roles must be DataRole values")
        if any(not isinstance(key, str) or not isinstance(value, str) for key, value in self.metadata.items()):
            raise ExperimentContractError("metadata values must be strings")
        _validate_data_isolation(self.data)
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        result = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "model": self.model.to_dict(),
            "data": {role: self.data[role].to_dict() for role in ("train", "validation", "benchmark")},
            "training": self.training.to_dict(),
            "evaluation": self.evaluation.to_dict(),
            "output": self.output.to_dict(),
        }
        if self.metadata:
            result["metadata"] = dict(sorted(self.metadata.items()))
        return result

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    def resolved_json(self) -> str:
        return self.canonical_json() + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def field_classification(self) -> dict[str, str]:
        return field_classification()

    def write_resolved(self, path: str | Path) -> None:
        """Persist the stable, default-expanded contract representation."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(self.resolved_json(), encoding="utf-8")


def _validate_data_isolation(data: Mapping[str, DataRole]) -> None:
    roles = tuple(data)
    for index, left in enumerate(roles):
        for right in roles[index + 1 :]:
            left_data = data[left]
            right_data = data[right]
            same_path = _normal_path(left_data.path) == _normal_path(right_data.path)
            same_fingerprint = (
                left_data.fingerprint is not None
                and left_data.fingerprint == right_data.fingerprint
            )
            if same_path or same_fingerprint:
                raise ExperimentContractError(f"data role collision: {left} and {right} refer to the same dataset")


def _normal_path(path: str) -> str:
    return posixpath.normpath(str(PurePath(path)))


_ROOT_KEYS = {"schema_version", "experiment_id", "model", "data", "training", "evaluation", "output", "metadata"}
_MODEL_KEYS = {"model_id", "revision", "revision_policy"}
_DATA_KEYS = {"path", "fingerprint"}
_TRAINING_KEYS = {"seed", "quantization", "lora", "optimizer", "checkpoint_policy", "stopping_policy", "preprocessing"}
_QUANTIZATION_KEYS = {"load_in_4bit", "quant_type", "compute_dtype", "double_quant"}
_LORA_KEYS = {"rank", "alpha", "dropout", "target_modules"}
_OPTIMIZER_KEYS = {
    "micro_batch_size",
    "gradient_accumulation_steps",
    "learning_rate",
    "max_epochs",
    "max_sequence_length",
    "gradient_checkpointing",
    "gradient_checkpointing_use_reentrant",
}
_CHECKPOINT_KEYS = {"selection_split", "validation_only", "primary_metric", "interval_steps"}
_STOPPING_KEYS = {"mode", "max_steps", "patience", "min_delta", "eval_interval_steps"}
_PREPROCESSING_KEYS = {"deterministic", "shuffle", "seed", "version"}
_EVALUATION_KEYS = {"contract_version", "scorer_version"}
_OUTPUT_KEYS = {"output_dir"}


def _reject_unknown(raw: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(raw) - allowed)
    if unknown:
        joined = ", ".join(f"{path}.{key}" for key in unknown)
        raise ExperimentContractError(f"unknown config key(s): {joined}")


def _object(raw: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(raw, Mapping):
        raise ExperimentContractError(f"{path} must be an object")
    return raw


def _data_role(raw: Any, role: str) -> DataRole:
    if isinstance(raw, str):
        return DataRole(raw)
    obj = _object(raw, f"data.{role}")
    _reject_unknown(obj, _DATA_KEYS, f"data.{role}")
    return DataRole(path=obj["path"], fingerprint=obj.get("fingerprint"))


def _parse_mapping(raw: Any, path: str, allowed: set[str]) -> dict[str, Any]:
    obj = _object(raw, path)
    _reject_unknown(obj, allowed, path)
    return dict(obj)


def experiment_contract_from_dict(raw: Mapping[str, Any]) -> ExperimentContract:
    """Parse a complete canonical contract and fail closed on unknown fields."""

    root = _parse_mapping(raw, "config", _ROOT_KEYS)
    if root.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ExperimentContractError("unsupported or missing contract schema_version")
    if "experiment_id" not in root:
        raise ExperimentContractError("missing experiment_id")

    model = _parse_mapping(root.get("model"), "model", _MODEL_KEYS)
    if "model_id" not in model or "revision" not in model:
        raise ExperimentContractError("model_id and model revision policy are required")

    data = _parse_mapping(root.get("data"), "data", {"train", "validation", "benchmark"})
    if set(data) != {"train", "validation", "benchmark"}:
        raise ExperimentContractError("data must define train, validation, and benchmark roles")

    training = _parse_mapping(root.get("training"), "training", _TRAINING_KEYS)
    quantization = _parse_mapping(training.get("quantization", {}), "training.quantization", _QUANTIZATION_KEYS)
    lora = _parse_mapping(training.get("lora", {}), "training.lora", _LORA_KEYS)
    optimizer = _parse_mapping(training.get("optimizer", {}), "training.optimizer", _OPTIMIZER_KEYS)
    checkpoint = _parse_mapping(
        training.get("checkpoint_policy", {}), "training.checkpoint_policy", _CHECKPOINT_KEYS
    )
    stopping = _parse_mapping(training.get("stopping_policy", {}), "training.stopping_policy", _STOPPING_KEYS)
    preprocessing = _parse_mapping(
        training.get("preprocessing", {}), "training.preprocessing", _PREPROCESSING_KEYS
    )
    evaluation = _parse_mapping(root.get("evaluation", {}), "evaluation", _EVALUATION_KEYS)
    output_raw = root.get("output")
    if isinstance(output_raw, str):
        output = {"output_dir": output_raw}
    else:
        output = _parse_mapping(output_raw, "output", _OUTPUT_KEYS)
    metadata = _parse_mapping(root.get("metadata", {}), "metadata", {"description", "owner", "notes", "tags"})

    target_modules = lora.get("target_modules", list(LoraPolicy().target_modules))
    if not isinstance(target_modules, (list, tuple)):
        raise ExperimentContractError("training.lora.target_modules must be an array")

    return ExperimentContract(
        schema_version=root.get("schema_version", SCHEMA_VERSION),
        experiment_id=root["experiment_id"],
        model=ModelIdentity(**model),
        data={role: _data_role(data[role], role) for role in ("train", "validation", "benchmark")},
        training=TrainingPolicy(
            seed=training["seed"],
            quantization=QuantizationPolicy(**quantization),
            lora=LoraPolicy(**{**lora, "target_modules": tuple(target_modules)}),
            optimizer=OptimizerPolicy(**optimizer),
            checkpoint_policy=CheckpointPolicy(**checkpoint),
            stopping_policy=StoppingPolicy(**stopping),
            preprocessing=PreprocessingPolicy(**preprocessing),
        ),
        evaluation=EvaluationBoundary(**evaluation),
        output=OutputPolicy(**output),
        metadata={str(key): str(value) for key, value in metadata.items()},
    )


def resolve_experiment_config(raw: Mapping[str, Any]) -> ExperimentContract:
    """Expand defaults, then parse the resulting canonical contract.

    The resolver accepts the two ergonomic shorthands used in planning notes:
    top-level ``model_id`` and ``seed``.  They are normalized into the
    canonical ``model`` and ``training`` sections before strict parsing.  A
    resolved config is still required to identify all three data roles and a
    pinned (or explicitly legacy-unpinned) model revision.
    """

    if not isinstance(raw, Mapping):
        raise ExperimentContractError("configuration root must be an object")
    source = dict(raw)
    if "model_id" in source:
        if "model" in source:
            raise ExperimentContractError("model_id shorthand cannot be combined with model")
        source["model"] = {"model_id": source.pop("model_id")}
    if "seed" in source:
        if "training" in source and isinstance(source["training"], Mapping) and "seed" in source["training"]:
            raise ExperimentContractError("seed shorthand cannot be combined with training.seed")
        training = dict(source.get("training", {}))
        training["seed"] = source.pop("seed")
        source["training"] = training

    root = _parse_mapping(source, "config", _ROOT_KEYS)
    if "experiment_id" not in root:
        raise ExperimentContractError("missing experiment_id")
    model = dict(_parse_mapping(root.get("model"), "model", _MODEL_KEYS))
    if "model_id" not in model or "revision" not in model:
        raise ExperimentContractError("model_id and model revision policy are required")
    data = _parse_mapping(root.get("data"), "data", {"train", "validation", "benchmark"})
    if set(data) != {"train", "validation", "benchmark"}:
        raise ExperimentContractError("data must define train, validation, and benchmark roles")

    training = dict(_parse_mapping(root.get("training"), "training", _TRAINING_KEYS))
    training.setdefault("seed", None)
    if training["seed"] is None:
        raise ExperimentContractError("missing training.seed")
    for key, allowed, defaults in (
        ("quantization", _QUANTIZATION_KEYS, QuantizationPolicy().to_dict()),
        ("lora", _LORA_KEYS, LoraPolicy().to_dict()),
        ("optimizer", _OPTIMIZER_KEYS, OptimizerPolicy().to_dict()),
        ("checkpoint_policy", _CHECKPOINT_KEYS, CheckpointPolicy().to_dict()),
        ("stopping_policy", _STOPPING_KEYS, StoppingPolicy().to_dict()),
        ("preprocessing", _PREPROCESSING_KEYS, PreprocessingPolicy(seed=training["seed"]).to_dict()),
    ):
        value = dict(_parse_mapping(training.get(key, {}), f"training.{key}", allowed))
        merged = dict(defaults)
        merged.update(value)
        training[key] = merged
    source["schema_version"] = root.get("schema_version", SCHEMA_VERSION)
    source["model"] = model
    source["training"] = training
    source["evaluation"] = {
        **EvaluationBoundary().to_dict(),
        **_parse_mapping(root.get("evaluation", {}), "evaluation", _EVALUATION_KEYS),
    }
    output = root.get("output", {"output_dir": f"runs/{root['experiment_id']}"})
    source["output"] = output
    return experiment_contract_from_dict(source)


def load_experiment_contract(path: str | Path) -> ExperimentContract:
    """Load and resolve a JSON experiment contract."""

    contract_path = Path(path)
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ExperimentContractError(f"configuration does not exist: {contract_path}") from exc
    except json.JSONDecodeError as exc:
        raise ExperimentContractError(f"invalid JSON configuration: {contract_path}: {exc}") from exc
    return resolve_experiment_config(raw)


def field_classification() -> dict[str, str]:
    """Return the exhaustive training-affecting/metadata-only field map."""

    result: dict[str, str] = {
        "schema_version": "metadata-only",
        "experiment_id": "metadata-only",
        "model.model_id": "training-affecting",
        "model.revision": "training-affecting",
        "model.revision_policy": "metadata-only",
        "data.train": "training-affecting",
        "data.train.path": "training-affecting",
        "data.train.fingerprint": "training-affecting",
        "data.validation": "training-affecting",
        "data.validation.path": "training-affecting",
        "data.validation.fingerprint": "training-affecting",
        "data.benchmark": "metadata-only",
        "data.benchmark.path": "metadata-only",
        "data.benchmark.fingerprint": "metadata-only",
        "evaluation.contract_version": "metadata-only",
        "evaluation.scorer_version": "metadata-only",
        "output.output_dir": "metadata-only",
    }
    for section, values in (
        ("training.quantization", _QUANTIZATION_KEYS),
        ("training.lora", _LORA_KEYS),
        ("training.optimizer", _OPTIMIZER_KEYS),
        ("training.checkpoint_policy", _CHECKPOINT_KEYS),
        ("training.stopping_policy", _STOPPING_KEYS),
        ("training.preprocessing", _PREPROCESSING_KEYS),
    ):
        result.update({f"{section}.{key}": "training-affecting" for key in sorted(values)})
    result["training.seed"] = "training-affecting"
    result.update({f"metadata.{key}": "metadata-only" for key in ("description", "owner", "notes", "tags")})
    return dict(sorted(result.items()))


def training_affecting_fields() -> frozenset[str]:
    """Return fields whose change can alter training or checkpoint selection."""

    return frozenset(key for key, value in field_classification().items() if value == "training-affecting")


def metadata_only_fields() -> frozenset[str]:
    """Return fields that identify or describe a run without changing training."""

    return frozenset(key for key, value in field_classification().items() if value == "metadata-only")


def legacy_config_to_contract(raw: Mapping[str, Any], *, experiment_id: str) -> dict[str, Any]:
    """Represent an E01/E02-style config in the M8 schema without executing it.

    Legacy files did not pin a model revision and used several dataset field
    names.  The adapter makes that fact explicit with a legacy revision policy;
    it never silently claims that the old config had stronger provenance than
    it actually did.
    """

    model_id = raw.get("model_id")
    train_path = raw.get("dataset_path") or raw.get("dataset_dir")
    validation_path = raw.get("validation_path")
    benchmark_path = raw.get("benchmark_dir") or raw.get("benchmark_path")
    dataset_root = raw.get("dataset_dir") or (
        str(PurePath(str(raw["dataset_path"])).parent) if raw.get("dataset_path") else None
    )
    if validation_path is None and dataset_root:
        validation_path = str(PurePath(str(dataset_root)) / "validation.jsonl")
    if benchmark_path is None and dataset_root:
        benchmark_path = str(PurePath(str(dataset_root)) / "benchmark.jsonl")
    if not model_id or not train_path or not validation_path or not benchmark_path:
        raise ExperimentContractError("legacy config cannot be represented without model and three data roles")

    training = raw.get("training", raw)
    quantization_raw = raw.get("quantization", training.get("quantization", {}))
    lora_raw = raw.get("lora", training.get("lora", {}))
    quantization = {
        key: quantization_raw[key]
        for key in _QUANTIZATION_KEYS
        if key in quantization_raw
    }
    lora = {key: lora_raw[key] for key in _LORA_KEYS if key in lora_raw}
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "model": {
            "model_id": model_id,
            "revision": raw.get("model_revision", "legacy-unpinned"),
            "revision_policy": "legacy_unpinned",
        },
        "data": {
            "train": {"path": str(train_path)},
            "validation": {"path": str(validation_path)},
            "benchmark": {"path": str(benchmark_path)},
        },
        "training": {
            "seed": raw.get("seed", training.get("seed", 0)),
            "quantization": quantization,
            "lora": lora,
            "optimizer": {
                "micro_batch_size": training.get("micro_batch_size", raw.get("micro_batch_size", 1)),
                "gradient_accumulation_steps": training.get(
                    "gradient_accumulation_steps", raw.get("gradient_accumulation_steps", 1)
                ),
                "learning_rate": training.get("learning_rate", raw.get("learning_rate", 2e-4)),
                "max_epochs": training.get("max_epochs", training.get("num_epochs", raw.get("num_epochs", 1))),
                "max_sequence_length": raw.get("max_sequence_length", training.get("max_sequence_length", 1024)),
                "gradient_checkpointing": training.get(
                    "gradient_checkpointing", raw.get("gradient_checkpointing", True)
                ),
                "gradient_checkpointing_use_reentrant": training.get(
                    "gradient_checkpointing_use_reentrant", raw.get("gradient_checkpointing_use_reentrant", False)
                ),
            },
        },
        "evaluation": {
            "contract_version": raw.get("prompt_version", "legacy-evaluation-v1"),
            "scorer_version": raw.get("scorer_version", "legacy-scorer-v1"),
        },
        "output": {"output_dir": raw.get("output_dir", f"outputs/{experiment_id}")},
    }
