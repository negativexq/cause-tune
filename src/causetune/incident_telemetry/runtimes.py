"""Bounded runtime/ecosystem catalog for Experiment 03B."""

from __future__ import annotations

from dataclasses import dataclass

from .models import RuntimeEnvironment


@dataclass(frozen=True)
class RuntimeFamilySpec:
    runtime_id: str
    ecosystem: str
    version: str


RUNTIME_FAMILIES: tuple[RuntimeFamilySpec, ...] = (
    RuntimeFamilySpec("python_fastapi", "python", "fastapi"),
    RuntimeFamilySpec("python_django", "python", "django"),
    RuntimeFamilySpec("java_spring", "java", "spring"),
    RuntimeFamilySpec("go_http", "go", "net_http"),
    RuntimeFamilySpec("node_nestjs", "node", "nestjs"),
)
RUNTIME_FAMILY_BY_ID = {item.runtime_id: item for item in RUNTIME_FAMILIES}


def build_runtime(runtime_id: str) -> RuntimeEnvironment:
    try:
        runtime = RUNTIME_FAMILY_BY_ID[runtime_id]
    except KeyError as exc:
        raise ValueError(f"unknown runtime family: {runtime_id!r}") from exc
    return RuntimeEnvironment(runtime.runtime_id, runtime.ecosystem, runtime.version)


def validate_runtime_catalog() -> None:
    if len(RUNTIME_FAMILIES) != len(RUNTIME_FAMILY_BY_ID):
        raise ValueError("runtime catalog contains duplicate IDs")


validate_runtime_catalog()
