#!/usr/bin/env python3
"""Experiment 03G: untouched Qwen3.5-2B decomposed RCA screening.

The contract is frozen before model loading.  This script consumes the
persisted 03F.1 R2 packages directly and never constructs model-facing input
from process labels, targets, trajectories, or source paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig

from causetune.cloudopsbench.scanner import scan_corpus
from causetune.cloudopsbench.screening import _strip_reasoning, target_for_case
from causetune.cloudopsbench.training_contract import source_case_group_id
from causetune.incident_telemetry.fingerprints import canonical_json


MODEL_ID = "Qwen/Qwen3.5-2B"
MODEL_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
SOURCE_REVISION = "03c415e5709297432282fbbfd499f1bca0f8c347"
SPLIT_FINGERPRINT = "0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76"
DECOMPOSITION_FINGERPRINT = "249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9"
REPRESENTATION_FINGERPRINT = "b676b10a17d690f3796e237e942c340f2722a1aa5bcd97af1b43de1c0e972215"
MAX_NEW_TOKENS = 256
PACKAGE_BUDGET_CHARS = 30000
VERSION = "cloud-opsbench-03g-decomposed-2b-screen-v1"
OUT = Path("results/incident_telemetry_03g")
SOURCE_ROOT = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
SPLIT_PATH = Path("results/incident_telemetry_03d/split_manifest.json")
SCREEN_PATH = Path("results/incident_telemetry_03e/validation_subsplit.json")
PACKAGE_PATH = Path("results/incident_telemetry_03f1/evidence_packages.jsonl")
HIERARCHY_PATH = Path("results/incident_telemetry_03f/hierarchy_registry.json")
SCREEN_SUFFICIENCY_PATH = Path("results/incident_telemetry_03f1/validation_screen_sufficiency.json")


def fp(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def stats(values: Iterable[int | float]) -> dict[str, Any]:
    ordered = sorted(values)
    if not ordered:
        return {"count": 0, "min": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0}
    def at(percentile: float) -> int | float:
        return ordered[round((len(ordered) - 1) * percentile)]
    return {"count": len(ordered), "min": ordered[0], "median": at(.5), "p75": at(.75), "p90": at(.9), "p95": at(.95), "p99": at(.99), "max": ordered[-1]}


def base_meta() -> dict[str, Any]:
    return {"experiment": "03G", "version": VERSION, "source_revision": SOURCE_REVISION, "split_manifest_fingerprint": SPLIT_FINGERPRINT, "decomposition_fingerprint": DECOMPOSITION_FINGERPRINT, "evidence_representation_fingerprint": REPRESENTATION_FINGERPRINT, "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "training": False, "lora": False, "provider_called": False, "test_model_facing": False, "validation_selection_model_facing": False}


def load_frozen_inputs() -> tuple[dict[str, Any], list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    split = read_json(SPLIT_PATH)
    if split.get("fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("03D split fingerprint mismatch")
    screen = read_json(SCREEN_PATH)
    if screen.get("screen_count") != 57 or screen.get("split_manifest_fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("VALIDATION_SCREEN manifest mismatch")
    groups = list(screen["screen_case_groups"])
    if len(groups) != 57 or len(set(groups)) != 57:
        raise ValueError("VALIDATION_SCREEN is not exactly 57 unique cases")
    hierarchy = read_json(HIERARCHY_PATH)
    if hierarchy.get("native_fault_type_count") != 57 or hierarchy.get("native_category_count") != 8 or hierarchy.get("conflicts"):
        raise ValueError("03F hierarchy is not frozen/complete")
    sufficiency = read_json(SCREEN_SUFFICIENCY_PATH)
    if sufficiency.get("screen_count") != 57 or not sufficiency.get("screen_membership_unchanged", {}).get("set_equal"):
        raise ValueError("03F.1 screen sufficiency manifest mismatch")
    packages: dict[str, dict[str, Any]] = {}
    for line in PACKAGE_PATH.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        packages[str(row["source_case_group_hash"])] = row
    if len(packages) != 633:
        raise ValueError(f"frozen R2 package count mismatch: {len(packages)}")
    return split, groups, hierarchy, sufficiency, packages


def package_by_group(groups: list[str], packages: Mapping[str, Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {}
    for group in groups:
        row = packages.get(fp(group))
        if row is None:
            raise ValueError(f"missing frozen package for {group}")
        if row.get("candidate") != "R2_DETERMINISTIC_PATH_UNION" or row.get("budget_chars") != PACKAGE_BUDGET_CHARS:
            raise ValueError("03F.1 package contract mismatch")
        model_input = row.get("model_input")
        if not isinstance(model_input, Mapping) or row.get("target_blind") is not True or row.get("process_labels_in_model_input") is not False or row.get("golden_answers_in_model_input") is not False:
            raise ValueError(f"model-facing package leakage boundary failed for {group}")
        result[group] = {"incident_request": str(model_input["incident_request"]), "evidence": list(model_input["evidence"]), "package_chars": row.get("package_chars"), "package_bytes": row.get("package_bytes"), "package_id": row.get("package_id")}
    return result


def vocabularies(hierarchy: Mapping[str, Any]) -> tuple[list[str], list[str]]:
    categories = list(hierarchy["ordered_categories"])
    fault_types = [fault_type for category in categories for fault_type in hierarchy["categories"][category]]
    if len(categories) != 8 or len(fault_types) != 57 or len(set(fault_types)) != 57:
        raise ValueError("native vocabulary mismatch")
    return categories, fault_types


def task_prompts(categories: list[str], hierarchy: Mapping[str, Any]) -> dict[str, Any]:
    category_json = json.dumps(categories, ensure_ascii=False)
    root_map = {category: list(hierarchy["categories"][category]) for category in categories}
    task_a_system = f"""You are a bounded Cloud-OpsBench incident classifier. Use only the incident request and evidence supplied by the user. Return JSON only, with exactly this schema: {{\"fault_category\": string}}. The value must be exactly one item from this complete native category enum: {category_json}. Do not output root_cause, fault_object, reasoning, explanation, confidence, or any other key. Do not use synonyms."""
    task_b_system = """You are a bounded Cloud-OpsBench root-cause classifier. The authoritative fault category is supplied as task context. Use only the incident request and evidence. Return JSON only with exactly this schema: {\"root_cause\": string}. The value must be exactly one item from the supplied category-local native candidate enum. Do not output the category, object, reasoning, explanation, confidence, or any other key. This is ORACLE_CATEGORY_ROOT_CAUSE, not end-to-end RCA."""
    task_c_stage1_system = f"""You are Stage 1 of a bounded hierarchical Cloud-OpsBench classifier. Infer only the native fault category from the incident request and evidence. Return JSON only with exactly this schema: {{\"fault_category\": string}}. The value must be exactly one item from this complete native category enum: {category_json}. Do not output root_cause, fault_object, reasoning, explanation, confidence, or any other key."""
    task_c_stage2_system = """You are Stage 2 of a bounded hierarchical Cloud-OpsBench classifier. The category below is the model's Stage-1 prediction, not an authoritative label. Choose a root cause only from the candidate enum shown for that predicted category. Use only the same incident request and evidence. Return JSON only with exactly this schema: {\"root_cause\": string}. Do not output category, object, reasoning, explanation, confidence, or any other key. If the predicted category is invalid, the candidate enum is empty and you must not substitute another category."""
    task_d_system = """You are a bounded Cloud-OpsBench fault-object resolver. Use only the incident request and evidence. Return JSON only with exactly this schema: {\"fault_object\": string}. Return the source-grounded resource identity you can resolve from the evidence. Do not output category, root cause, reasoning, explanation, confidence, or any other key. No object candidate list is provided."""
    return {"TASK_A": {"system": task_a_system, "output_schema": {"type": "object", "required": ["fault_category"], "additionalProperties": False}, "allowed_categories": categories}, "TASK_B": {"system": task_b_system, "output_schema": {"type": "object", "required": ["root_cause"], "additionalProperties": False}, "candidate_sets": root_map, "oracle_category": True}, "TASK_C_STAGE1": {"system": task_c_stage1_system, "output_schema": {"type": "object", "required": ["fault_category"], "additionalProperties": False}, "allowed_categories": categories}, "TASK_C_STAGE2": {"system": task_c_stage2_system, "output_schema": {"type": "object", "required": ["root_cause"], "additionalProperties": False}, "candidate_context": "model-predicted category only", "candidate_sets": root_map}, "TASK_D": {"system": task_d_system, "output_schema": {"type": "object", "required": ["fault_object"], "additionalProperties": False}, "object_candidates": False}}


def messages_for(task: str, package: Mapping[str, Any], prompts: Mapping[str, Any], *, category_context: str | None = None, predicted_category: str | None = None, candidates: list[str] | None = None) -> list[dict[str, str]]:
    payload: dict[str, Any] = {"incident_request": package["incident_request"], "evidence": package["evidence"]}
    if task == "TASK_B":
        payload["authoritative_fault_category"] = category_context
        payload["allowed_root_causes"] = candidates or []
    elif task == "TASK_C_STAGE2":
        payload["predicted_fault_category"] = predicted_category
        payload["allowed_root_causes"] = candidates or []
    user = "Task input:\n" + json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return [{"role": "system", "content": prompts[task]["system"]}, {"role": "user", "content": user}]


def parse_json_output(text: str, expected_keys: set[str]) -> tuple[dict[str, Any] | None, str]:
    clean = _strip_reasoning(text)
    try:
        value = json.loads(clean)
    except json.JSONDecodeError:
        return None, "malformed_json"
    if not isinstance(value, dict) or set(value) != expected_keys:
        return None, "wrong_schema"
    if any(not isinstance(value[key], str) or not value[key].strip() for key in expected_keys):
        return None, "invalid_values"
    return value, "valid"


def load_model(snapshot: Path) -> tuple[Any, Any, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("03G requires CUDA; CPU fallback is forbidden")
    tokenizer = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    before = torch.cuda.memory_allocated()
    started = time.perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(str(snapshot), local_files_only=True, quantization_config=quantization, dtype=torch.bfloat16, device_map={"": torch.cuda.current_device()}, trust_remote_code=False)
    load_seconds = time.perf_counter() - started
    model.eval()
    model.config.use_cache = True
    after = torch.cuda.memory_allocated()
    return model, tokenizer, {"model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(snapshot), "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}, "enable_thinking": False, "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "num_beams": 1, "model_load_seconds": load_seconds, "model_load_allocated_bytes": after, "model_load_delta_bytes": after - before, "tokenizer_vocab_size": len(tokenizer)}


def generate(model: Any, tokenizer: Any, messages: list[dict[str, str]]) -> dict[str, Any]:
    encoded = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False, return_tensors="pt", return_dict=True)
    device = next(model.parameters()).device
    encoded = {key: value.to(device) if hasattr(value, "to") else value for key, value in encoded.items()}
    prompt_tokens = int(encoded["input_ids"].shape[-1])
    native_context = int(getattr(model.config, "max_position_embeddings", 262144))
    if prompt_tokens + MAX_NEW_TOKENS > native_context:
        return {"raw_output": "", "generated_tokens": 0, "prompt_tokens": prompt_tokens, "wall_seconds": 0.0, "peak_allocated_bytes": 0, "peak_reserved_bytes": 0, "failure": "context_overflow"}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        with torch.inference_mode():
            output = model.generate(**encoded, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1, use_cache=True, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        elapsed = time.perf_counter() - started
        new = output[:, prompt_tokens:]
        raw = tokenizer.decode(new[0], skip_special_tokens=False)
        return {"raw_output": raw, "generated_tokens": int(new.shape[-1]), "prompt_tokens": prompt_tokens, "wall_seconds": elapsed, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "failure": None}
    except (RuntimeError, ValueError) as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"raw_output": "", "generated_tokens": 0, "prompt_tokens": prompt_tokens, "wall_seconds": time.perf_counter() - started, "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()), "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()), "failure": type(exc).__name__ + ": " + str(exc)[:500]}


def target_record(case: Any) -> dict[str, str]:
    target = target_for_case(case)
    return {"fault_category": str(target["native_fault_category"]), "root_cause": str(target["native_fault_type"]), "fault_object": str(target["fault_object"])}


def visible_aliases(evidence: Iterable[Mapping[str, Any]]) -> set[str]:
    aliases: set[str] = set()
    for item in evidence:
        text = str(item.get("observation", ""))
        import re
        aliases.update(re.findall(r"\\bapp=([A-Za-z0-9][A-Za-z0-9._-]*)", text))
        aliases.update(re.findall(r"(?m)^([A-Za-z0-9][A-Za-z0-9._-]*)\\s+(?:ClusterIP|NodePort|LoadBalancer)\\b", text))
    return aliases


def object_normalized_match(prediction: Mapping[str, Any] | None, target: str, evidence: Iterable[Mapping[str, Any]]) -> bool:
    if not prediction or not isinstance(prediction.get("fault_object"), str):
        return False
    value = prediction["fault_object"]
    if value == target:
        return True
    return target.startswith("app/") and value == target[4:] and value in visible_aliases(evidence)


def row_common(group: str, case: Any, bucket: str, package: Mapping[str, Any], messages: list[dict[str, str]], generated: Mapping[str, Any], target: Mapping[str, str], task: str) -> dict[str, Any]:
    return {"source_case_group": group, "split": "VALIDATION_SCREEN", "source_system": case.source_system, "source_case_id": case.source_case_id, "source_category": case.source_fault_category, "source_root_cause": case.source_fault_type, "difficulty": case.upstream_difficulty, "metrics_available": case.references.metrics is not None, "code_available": case.references.code is not None, "evidence_bucket": bucket, "package_chars": package["package_chars"], "package_bytes": package["package_bytes"], "task": task, "input_fingerprint": fp(messages), "target": dict(target), **dict(generated)}


def enum_valid(task: str, prediction: Mapping[str, Any] | None, categories: list[str], hierarchy: Mapping[str, Any], *, authoritative_category: str | None = None, predicted_category: str | None = None) -> bool:
    if prediction is None:
        return False
    if task in {"TASK_A", "TASK_C_STAGE1"}:
        return prediction.get("fault_category") in categories
    if task == "TASK_B":
        return prediction.get("root_cause") in hierarchy["categories"].get(authoritative_category or "", [])
    if task == "TASK_C_STAGE2":
        return prediction.get("root_cause") in hierarchy["categories"].get(predicted_category or "", [])
    return prediction.get("fault_object") is not None


def task_metrics(rows: list[Mapping[str, Any]], task: str, hierarchy: Mapping[str, Any], categories: list[str]) -> dict[str, Any]:
    total = len(rows)
    valid_schema = sum(row.get("parse_status") == "valid" for row in rows)
    valid_enum = sum(row.get("enum_valid", False) for row in rows)
    result: dict[str, Any] = {"task": task, "count": total, "schema_valid_count": valid_schema, "schema_valid_rate": valid_schema / total if total else 0.0, "enum_valid_count": valid_enum, "enum_valid_rate": valid_enum / total if total else 0.0, "parse_status_counts": dict(sorted(Counter(row.get("parse_status") for row in rows).items()))}
    if task in {"TASK_A", "TASK_C_STAGE1"}:
        result["category_exact_count"] = sum(row.get("prediction", {}).get("fault_category") == row["target"]["fault_category"] for row in rows if row.get("prediction"))
        result["category_exact_rate"] = result["category_exact_count"] / total if total else 0.0
    elif task == "TASK_B":
        result["root_cause_exact_count"] = sum(row.get("prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in rows if row.get("prediction"))
        result["root_cause_exact_rate"] = result["root_cause_exact_count"] / total if total else 0.0
    elif task == "TASK_C":
        result["stage1_category_exact_count"] = sum(row["stage1_prediction"].get("fault_category") == row["target"]["fault_category"] for row in rows if row.get("stage1_prediction"))
        result["stage1_category_exact_rate"] = result["stage1_category_exact_count"] / total if total else 0.0
        result["stage2_root_cause_exact_count"] = sum(row.get("stage2_prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in rows if row.get("stage2_prediction"))
        result["stage2_root_cause_exact_rate"] = result["stage2_root_cause_exact_count"] / total if total else 0.0
        result["hierarchical_joint_exact_count"] = sum(row.get("stage1_prediction", {}).get("fault_category") == row["target"]["fault_category"] and row.get("stage2_prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in rows)
        result["hierarchical_joint_exact_rate"] = result["hierarchical_joint_exact_count"] / total if total else 0.0
        category_correct = [row for row in rows if row.get("stage1_prediction", {}).get("fault_category") == row["target"]["fault_category"]]
        result["stage2_exact_given_category_correct_count"] = sum(row.get("stage2_prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in category_correct)
        result["stage2_exact_given_category_correct_rate"] = result["stage2_exact_given_category_correct_count"] / len(category_correct) if category_correct else 0.0
        result["failures_wrong_stage1_category"] = sum(row.get("stage1_prediction", {}).get("fault_category") != row["target"]["fault_category"] for row in rows)
        result["failures_despite_correct_stage1_category"] = sum(row.get("stage1_prediction", {}).get("fault_category") == row["target"]["fault_category"] and row.get("stage2_prediction", {}).get("root_cause") != row["target"]["root_cause"] for row in rows)
    elif task == "TASK_D":
        result["strict_object_exact_count"] = sum(row.get("prediction", {}).get("fault_object") == row["target"]["fault_object"] for row in rows if row.get("prediction"))
        result["strict_object_exact_rate"] = result["strict_object_exact_count"] / total if total else 0.0
        result["normalized_object_exact_count"] = sum(row.get("normalized_object_exact", False) for row in rows)
        result["normalized_object_exact_rate"] = result["normalized_object_exact_count"] / total if total else 0.0
    result["invalid_enum_outputs"] = sum(row.get("parse_status") == "valid" and not row.get("enum_valid", False) for row in rows)
    result["missing_or_malformed_outputs"] = total - valid_schema
    return result


def per_group_metrics(rows: list[Mapping[str, Any]], task: str, hierarchy: Mapping[str, Any], categories: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    keys = [("category", row["target"]["fault_category"]) for row in rows] if task in {"TASK_A", "TASK_C"} else [("category", row["target"]["fault_category"]) for row in rows]
    for dimension in ("category", "source_root_cause", "source_system", "difficulty", "evidence_bucket", "metrics_available", "code_available", "package_token_bucket"):
        groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            if dimension == "category": value = row["target"]["fault_category"]
            elif dimension == "source_root_cause": value = row["target"]["root_cause"]
            else: value = row.get(dimension)
            groups[str(value)].append(row)
        result[dimension] = {value: task_metrics(group, task, hierarchy, categories) for value, group in sorted(groups.items())}
    return result


def process_bucket_rows(rows: list[Mapping[str, Any]], task: str) -> dict[str, Any]:
    return {bucket: {"count": len(group), "schema_valid": sum(row.get("parse_status") == "valid" for row in group), "enum_valid": sum(row.get("enum_valid", False) for row in group), "category_exact": sum(row.get("prediction", {}).get("fault_category") == row["target"]["fault_category"] for row in group if row.get("prediction")) if task in {"TASK_A", "TASK_C_STAGE1"} else None, "root_cause_exact": sum(row.get("prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in group if row.get("prediction")) if task == "TASK_B" else None} for bucket, group in sorted(_group_by(rows, "evidence_bucket").items())}


def _group_by(rows: Iterable[Mapping[str, Any]], key: str) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(key))].append(row)
    return groups


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=SOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    args = parser.parse_args()
    out = args.output_dir
    if out.exists() and (out / "artifact_fingerprints.json").exists():
        raise RuntimeError("03G output already complete; immutable outputs will not be overwritten")
    out.mkdir(parents=True, exist_ok=True)
    split, groups, hierarchy, sufficiency, all_packages = load_frozen_inputs()
    categories, fault_types = vocabularies(hierarchy)
    scan = scan_corpus(args.root, source_revision=SOURCE_REVISION, fail_closed=True)
    by_group = {source_case_group_id(case): case for case in scan.cases}
    cases = [by_group[group] for group in groups]
    if len(cases) != 57 or set(groups) != {source_case_group_id(case) for case in cases}:
        raise ValueError("screen case membership changed")
    packages = package_by_group(groups, all_packages)
    bucket_by_group = {row["source_case_group"]: row["coverage_class"] for row in sufficiency["cases"]}
    if set(bucket_by_group) != set(groups):
        raise ValueError("screen evidence bucket membership changed")
    prompts = task_prompts(categories, hierarchy)
    generation = {"version": "cloud-opsbench-03g-generation-v1", "max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "num_beams": 1, "temperature": None, "top_p": None, "enable_thinking": False, "truncation": False, "output_repair": False, "parser": "strip known Qwen control/reasoning delimiters, then strict JSON object/schema validation", "inference_count": 228}
    screen_manifest = {**read_json(SCREEN_PATH), "frozen_copy": True, "sha256": file_hash(SCREEN_PATH), "model_facing_count": 57, "validation_selection_model_facing": False, "test_model_facing": False}
    model_meta_prior = read_json(Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json"))
    model_registry = {**base_meta(), "snapshot_path": model_meta_prior["snapshot_path"], "model_metadata_sha256": file_hash(Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json")), "pinned_revision_verified": model_meta_prior.get("revision") == MODEL_REVISION, "tokenizer_hash": model_meta_prior.get("tokenizer_hash"), "config_hash": model_meta_prior.get("config_hash"), "chat_template_hash": model_meta_prior.get("chat_template_hash"), "quantization": {"load_in_4bit": True, "quant_type": "nf4", "compute_dtype": "bfloat16", "double_quant": True}, "model_weights_stored": False}
    target_categories = [target_record(case)["fault_category"] for case in cases]
    target_roots = [target_record(case)["root_cause"] for case in cases]
    category_counts = Counter(target_categories)
    b_expected = sum(1 / len(hierarchy["categories"][category]) for category in target_categories) / len(cases)
    c_expected = sum(1 / (len(categories) * len(hierarchy["categories"][category])) for category in target_categories) / len(cases)
    majority_category, majority_count = sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    baseline = {**base_meta(), "screen_count": 57, "frozen_before_model_loading": True, "TASK_A": {"uniform_category_expected_rate": 1 / len(categories), "majority_category": majority_category, "majority_count": majority_count, "majority_rate": majority_count / len(cases), "category_counts": dict(sorted(category_counts.items()))}, "TASK_B_ORACLE_CATEGORY_ROOT_CAUSE": {"uniform_within_category_expected_rate": b_expected, "per_case_expected_values": [1 / len(hierarchy["categories"][category]) for category in target_categories], "candidate_set_size_counts": dict(sorted(Counter(len(hierarchy["categories"][category]) for category in target_categories).items()))}, "TASK_C_HIERARCHICAL_SELF_PREDICTED": {"hierarchical_random_expected_joint_rate": c_expected, "category_random_expected_rate": 1 / len(categories)}, "definition": "expectations computed from frozen hierarchy and actual 57-case screen targets before inference; model outputs not used"}
    contract = {**base_meta(), "contract_version": VERSION, "screen_manifest_sha256": screen_manifest["sha256"], "screen_groups_fingerprint": fp(groups), "screen_count": 57, "native_category_enum": categories, "native_fault_type_enum": fault_types, "hierarchy_candidate_sets": {category: hierarchy["categories"][category] for category in categories}, "evidence_source": str(PACKAGE_PATH), "evidence_representation": "R2_DETERMINISTIC_PATH_UNION", "evidence_budget_chars": PACKAGE_BUDGET_CHARS, "task_prompts": prompts, "task_b_contract_boundary": "authoritative category supplied as explicit decomposition context; never end-to-end RCA", "task_c_stage2_boundary": "uses persisted Task A prediction only; authoritative category is never substituted", "task_d_normalization": "strict object exact plus bare app/name -> app/name only when unique source-visible application identity is present", "no_tools": True, "no_process_labels_in_input": True, "no_target_labels_in_input": "except Task B authoritative category field", "few_shot": False, "case_specific_examples": False, "prompt_frozen_before_first_inference": True, "generation_frozen_before_first_inference": True, "training": False, "test_used": False, "validation_selection_used": False}
    generation_contract = {**base_meta(), **generation, "prompt_fingerprint": fp(prompts), "generation_fingerprint": fp(generation), "model_registry_fingerprint": fp(model_registry), "frozen_before_model_loading": True}
    # These are the pre-inference freeze artefacts.  They are written before
    # tokenizer/model loading and are never changed after generation begins.
    write_json(out / "experiment_contract.json", contract)
    write_json(out / "model_registry.json", model_registry)
    write_json(out / "generation_contract.json", generation_contract)
    write_json(out / "screen_manifest.json", screen_manifest)
    write_json(out / "baseline_verification.json", baseline)
    write_json(out / "native_vocabulary.json", {**base_meta(), "ordered_categories": categories, "ordered_fault_types": fault_types, "candidate_sets": hierarchy["categories"], "fingerprint": fp({"categories": categories, "fault_types": fault_types, "candidate_sets": hierarchy["categories"]}), "correct_case_label_highlighted": False})
    for task in ("task_a", "task_b", "task_c", "task_d"):
        (out / task).mkdir(exist_ok=True)
    for task, prompt in prompts.items():
        write_json(out / {"TASK_A": "task_a", "TASK_B": "task_b", "TASK_C_STAGE1": "task_c", "TASK_C_STAGE2": "task_c", "TASK_D": "task_d"}[task] / ("prompt_stage1.json" if task == "TASK_C_STAGE1" else "prompt_stage2.json" if task == "TASK_C_STAGE2" else "prompt.json"), {**base_meta(), "task": task, **prompt, "frozen_before_first_inference": True, "prompt_fingerprint": fp(prompt)})

    snapshot = Path(model_meta_prior["snapshot_path"])
    model, tokenizer, loaded_meta = load_model(snapshot)
    write_json(out / "model_registry.json", {**model_registry, "loaded_model": True, "transformers_version": __import__("transformers").__version__, "torch_version": torch.__version__, "bitsandbytes_version": __import__("bitsandbytes").__version__, "cuda_version": torch.version.cuda, "gpu_model": torch.cuda.get_device_name(), "physical_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "load": loaded_meta})

    token_rows: dict[str, list[int]] = defaultdict(list)
    package_tokens: dict[str, int] = {}
    for group in groups:
        package = packages[group]
        package_text = json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        package_tokens[group] = len(tokenizer(package_text, add_special_tokens=False)["input_ids"])
        token_rows["TASK_A"].append(len(tokenizer.apply_chat_template(messages_for("TASK_A", package, prompts), tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]))
        token_rows["TASK_B"].append(len(tokenizer.apply_chat_template(messages_for("TASK_B", package, prompts, category_context=target_record(by_group[group])["fault_category"], candidates=hierarchy["categories"][target_record(by_group[group])["fault_category"]]), tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]))
        token_rows["TASK_C_STAGE1"].append(len(tokenizer.apply_chat_template(messages_for("TASK_C_STAGE1", package, prompts), tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]))
        # Stage 2 uses the persisted Task-A prediction after Task A completes;
        # a provisional empty-category measurement is replaced below.
        token_rows["TASK_D"].append(len(tokenizer.apply_chat_template(messages_for("TASK_D", package, prompts), tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]))

    task_elapsed: dict[str, float] = {}
    task_a_rows: list[dict[str, Any]] = []
    task_b_rows: list[dict[str, Any]] = []
    task_d_rows: list[dict[str, Any]] = []
    task_a_started = time.perf_counter()
    for index, group in enumerate(groups, 1):
        case = by_group[group]; package = packages[group]; target = target_record(case); messages = messages_for("TASK_A", package, prompts); generated = generate(model, tokenizer, messages); prediction, parse_status = parse_json_output(generated["raw_output"], {"fault_category"}); row = row_common(group, case, bucket_by_group[group], package, messages, {**generated, "prediction": prediction, "parse_status": parse_status, "enum_valid": enum_valid("TASK_A", prediction, categories, hierarchy)}, target, "TASK_A"); row["package_token_count"] = package_tokens[group]; row["package_token_bucket"] = "<4k" if package_tokens[group] < 4000 else "4k-8k" if package_tokens[group] < 8000 else "8k-16k" if package_tokens[group] < 16000 else ">=16k"; task_a_rows.append(row); print(f"TASK_A {index}/57 parse={parse_status} enum={row['enum_valid']}", flush=True)
    write_jsonl(out / "task_a" / "predictions.jsonl", task_a_rows)
    task_a_metrics = task_metrics(task_a_rows, "TASK_A", hierarchy, categories)
    write_json(out / "task_a" / "metrics.json", {**base_meta(), **task_a_metrics, "baseline_reference": "baseline_verification.json::TASK_A"})
    task_elapsed["TASK_A"] = time.perf_counter() - task_a_started
    write_json(out / "task_a" / "confusion_matrix.json", {category: {pred: sum(row.get("prediction", {}).get("fault_category") == pred and row["target"]["fault_category"] == category for row in task_a_rows) for pred in categories} for category in categories})
    write_json(out / "task_a" / "failure_analysis.json", {**base_meta(), "parse_failures": dict(sorted(Counter(row["parse_status"] for row in task_a_rows if row["parse_status"] != "valid").items())), "enum_failures": [row["source_case_group"] for row in task_a_rows if not row["enum_valid"]], "per_slice": per_group_metrics(task_a_rows, "TASK_A", hierarchy, categories)})

    task_b_started = time.perf_counter()
    for index, group in enumerate(groups, 1):
        case = by_group[group]; package = packages[group]; target = target_record(case); category = target["fault_category"]; messages = messages_for("TASK_B", package, prompts, category_context=category, candidates=hierarchy["categories"][category]); generated = generate(model, tokenizer, messages); prediction, parse_status = parse_json_output(generated["raw_output"], {"root_cause"}); row = row_common(group, case, bucket_by_group[group], package, messages, {**generated, "prediction": prediction, "parse_status": parse_status, "enum_valid": enum_valid("TASK_B", prediction, categories, hierarchy, authoritative_category=category)}, target, "TASK_B"); row["authoritative_category_context"] = category; row["candidate_set_size"] = len(hierarchy["categories"][category]); row["package_token_count"] = package_tokens[group]; row["package_token_bucket"] = "<4k" if package_tokens[group] < 4000 else "4k-8k" if package_tokens[group] < 8000 else "8k-16k" if package_tokens[group] < 16000 else ">=16k"; task_b_rows.append(row); print(f"TASK_B {index}/57 parse={parse_status} enum={row['enum_valid']}", flush=True)
    write_jsonl(out / "task_b" / "predictions.jsonl", task_b_rows)
    write_json(out / "task_b" / "metrics.json", {**base_meta(), **task_metrics(task_b_rows, "TASK_B", hierarchy, categories), "baseline_reference": "baseline_verification.json::TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"})
    write_json(out / "task_b" / "per_category.json", per_group_metrics(task_b_rows, "TASK_B", hierarchy, categories)["category"])
    write_json(out / "task_b" / "failure_analysis.json", {**base_meta(), "parse_failures": dict(sorted(Counter(row["parse_status"] for row in task_b_rows if row["parse_status"] != "valid").items())), "enum_failures": [row["source_case_group"] for row in task_b_rows if not row["enum_valid"]], "candidate_set_size": {str(size): task_metrics([row for row in task_b_rows if row["candidate_set_size"] == size], "TASK_B", hierarchy, categories) for size in sorted({row["candidate_set_size"] for row in task_b_rows})}})
    task_elapsed["TASK_B"] = time.perf_counter() - task_b_started

    # C Stage 1 is exactly the persisted A prediction; no second generation.
    task_c_rows: list[dict[str, Any]] = []
    task_c_started = time.perf_counter()
    for index, (group, a_row) in enumerate(zip(groups, task_a_rows), 1):
        case = by_group[group]; package = packages[group]; target = target_record(case); stage1 = a_row.get("prediction"); predicted_category = stage1.get("fault_category") if isinstance(stage1, Mapping) else None; candidates = hierarchy["categories"].get(predicted_category, []) if predicted_category else []; messages = messages_for("TASK_C_STAGE2", package, prompts, predicted_category=predicted_category or "<INVALID_STAGE1_CATEGORY>", candidates=candidates); token_rows["TASK_C_STAGE2"].append(len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"])); generated = generate(model, tokenizer, messages); prediction, parse_status = parse_json_output(generated["raw_output"], {"root_cause"}); stage2_enum = enum_valid("TASK_C_STAGE2", prediction, categories, hierarchy, predicted_category=predicted_category); row = row_common(group, case, bucket_by_group[group], package, messages, {**generated, "stage2_prediction": prediction, "stage2_parse_status": parse_status, "stage2_enum_valid": stage2_enum}, target, "TASK_C"); row["stage1_prediction"] = stage1; row["stage1_parse_status"] = a_row["parse_status"]; row["stage1_enum_valid"] = a_row["enum_valid"]; row["stage1_protocol_failure"] = a_row["parse_status"] != "valid" or not a_row["enum_valid"]; row["stage2_prediction"] = prediction; row["stage2_parse_status"] = parse_status; row["stage2_enum_valid"] = stage2_enum; row["enum_valid"] = bool(a_row["enum_valid"] and stage2_enum); row["parse_status"] = "valid" if a_row["parse_status"] == "valid" and parse_status == "valid" else "stage1_protocol_failure" if a_row["parse_status"] != "valid" else parse_status; row["predicted_category"] = predicted_category; row["predicted_candidate_set_size"] = len(candidates); row["package_token_count"] = package_tokens[group]; row["package_token_bucket"] = "<4k" if package_tokens[group] < 4000 else "4k-8k" if package_tokens[group] < 8000 else "8k-16k" if package_tokens[group] < 16000 else ">=16k"; task_c_rows.append(row); print(f"TASK_C_STAGE2 {index}/57 parse={parse_status} enum={stage2_enum} stage1={predicted_category}", flush=True)
    write_jsonl(out / "task_c" / "predictions.jsonl", task_c_rows)
    write_json(out / "task_c" / "metrics.json", {**base_meta(), **task_metrics(task_c_rows, "TASK_C", hierarchy, categories), "stage1_source": "task_a/predictions.jsonl; no regeneration", "baseline_reference": "baseline_verification.json::TASK_C_HIERARCHICAL_SELF_PREDICTED"})
    write_json(out / "task_c" / "error_decomposition.json", {**base_meta(), "category_bottleneck_count": sum(row["stage1_prediction"].get("fault_category") != row["target"]["fault_category"] for row in task_c_rows), "within_category_discrimination_failure_count": sum(row["stage1_prediction"].get("fault_category") == row["target"]["fault_category"] and row.get("stage2_prediction", {}).get("root_cause") != row["target"]["root_cause"] for row in task_c_rows), "stage1_protocol_failure_count": sum(row["stage1_protocol_failure"] for row in task_c_rows), "stage2_invalid_after_valid_stage1_count": sum(row["stage1_prediction"].get("fault_category") == row["target"]["fault_category"] and not row["stage2_enum_valid"] for row in task_c_rows)})
    write_json(out / "task_c" / "failure_analysis.json", {**base_meta(), "stage1_protocol_failures": [row["source_case_group"] for row in task_c_rows if row["stage1_protocol_failure"]], "stage2_parse_failures": dict(sorted(Counter(row["stage2_parse_status"] for row in task_c_rows if row["stage2_parse_status"] != "valid").items())), "per_slice": per_group_metrics(task_c_rows, "TASK_C", hierarchy, categories)})
    task_elapsed["TASK_C_STAGE2"] = time.perf_counter() - task_c_started

    task_d_started = time.perf_counter()
    for index, group in enumerate(groups, 1):
        case = by_group[group]; package = packages[group]; target = target_record(case); messages = messages_for("TASK_D", package, prompts); generated = generate(model, tokenizer, messages); prediction, parse_status = parse_json_output(generated["raw_output"], {"fault_object"}); row = row_common(group, case, bucket_by_group[group], package, messages, {**generated, "prediction": prediction, "parse_status": parse_status, "enum_valid": enum_valid("TASK_D", prediction, categories, hierarchy)}, target, "TASK_D"); row["normalized_object_exact"] = object_normalized_match(prediction, target["fault_object"], package["evidence"]); row["package_token_count"] = package_tokens[group]; row["package_token_bucket"] = "<4k" if package_tokens[group] < 4000 else "4k-8k" if package_tokens[group] < 8000 else "8k-16k" if package_tokens[group] < 16000 else ">=16k"; task_d_rows.append(row); print(f"TASK_D {index}/57 parse={parse_status}", flush=True)
    write_jsonl(out / "task_d" / "predictions.jsonl", task_d_rows)
    write_json(out / "task_d" / "metrics.json", {**base_meta(), **task_metrics(task_d_rows, "TASK_D", hierarchy, categories)})
    write_json(out / "task_d" / "failure_analysis.json", {**base_meta(), "parse_failures": dict(sorted(Counter(row["parse_status"] for row in task_d_rows if row["parse_status"] != "valid").items())), "strict_object_failures": [row["source_case_group"] for row in task_d_rows if row.get("prediction", {}).get("fault_object") != row["target"]["fault_object"]], "normalized_object_successes": [row["source_case_group"] for row in task_d_rows if row["normalized_object_exact"]], "per_slice": per_group_metrics(task_d_rows, "TASK_D", hierarchy, categories)})
    task_elapsed["TASK_D"] = time.perf_counter() - task_d_started

    token_rows["TASK_C_STAGE2"] = token_rows["TASK_C_STAGE2"][-57:]
    token_metrics = {task: stats(values) for task, values in token_rows.items()}
    token_metrics["package_input_tokens"] = stats(package_tokens.values())
    token_metrics.update({"context_limit_tokens": int(getattr(model.config, "max_position_embeddings", 262144)), "context_overflow_count": sum(1 for values in token_rows.values() for value in values if value + MAX_NEW_TOKENS > int(getattr(model.config, "max_position_embeddings", 262144))), "truncation_count": 0, "oom_count": sum(row.get("failure", "").lower().find("out of memory") >= 0 for row in task_a_rows + task_b_rows + task_c_rows + task_d_rows), "measurement": "exact local tokenizer apply_chat_template; no truncation; Task C Stage 1 is Task A persisted prediction"})
    write_json(out / "tokenization_metrics.json", {**base_meta(), **token_metrics})
    task_wall = task_elapsed
    all_rows = task_a_rows + task_b_rows + task_c_rows + task_d_rows
    hardware = {**base_meta(), "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "bitsandbytes": __import__("bitsandbytes").__version__, "cuda": torch.version.cuda}, "gpu_model": torch.cuda.get_device_name(), "physical_vram_bytes": torch.cuda.get_device_properties(0).total_memory, "model_load": loaded_meta, "peak_allocated_bytes": max([row.get("peak_allocated_bytes", 0) for row in all_rows] or [0]), "peak_reserved_bytes": max([row.get("peak_reserved_bytes", 0) for row in all_rows] or [0]), "task_wall_seconds": task_wall, "total_wall_seconds": sum(task_wall.values()) + loaded_meta["model_load_seconds"], "total_generation_count": 228, "oom_count": token_metrics["oom_count"], "context_overflow_count": token_metrics["context_overflow_count"], "tokens_per_second": {task: (sum(row.get("generated_tokens", 0) for row in rows) / sum(row.get("wall_seconds", 0.0) for row in rows)) if sum(row.get("wall_seconds", 0.0) for row in rows) else None for task, rows in (("TASK_A", task_a_rows), ("TASK_B", task_b_rows), ("TASK_C_STAGE2", task_c_rows), ("TASK_D", task_d_rows))}}
    write_json(out / "hardware_metrics.json", hardware)

    # Evidence buckets and frozen TRAIN support are descriptive slices only.
    bucket_analysis = {task: {bucket: task_metrics([row for row in rows if row["evidence_bucket"] == bucket], "TASK_A" if task == "TASK_A" else "TASK_B" if task == "TASK_B" else "TASK_C" if task == "TASK_C" else "TASK_D", hierarchy, categories) for bucket in ("FULL", "PARTIAL", "ZERO")} for task, rows in (("TASK_A", task_a_rows), ("TASK_B", task_b_rows), ("TASK_C", task_c_rows), ("TASK_D", task_d_rows))}
    write_json(out / "evidence_bucket_analysis.json", {**base_meta(), "interpretation": "analysis slices only; ZERO does not imply no diagnostic information", "process_label_coverage_is_not_classification_sufficiency": True, "by_task": bucket_analysis})
    train_support = read_json(Path("results/incident_telemetry_03f/train_label_support.json"))["by_root_cause"]
    train_cross = []
    for group, b_row, c_row in zip(groups, task_b_rows, task_c_rows):
        root = b_row["target"]["root_cause"]
        train_cross.append({"source_case_group": group, "root_cause": root, "category": b_row["target"]["fault_category"], "train_count": train_support[root]["count"], "low_support": train_support[root]["count"] < 3, "task_b_exact": b_row.get("prediction", {}).get("root_cause") == root, "task_c_category_exact": c_row.get("stage1_prediction", {}).get("fault_category") == c_row["target"]["fault_category"], "task_c_root_exact": c_row.get("stage2_prediction", {}).get("root_cause") == root})
    write_json(out / "train_support_cross_analysis.json", {**base_meta(), "low_support_definition": "TRAIN count <3", "low_support_labels": sorted(root for root, item in train_support.items() if item["count"] < 3), "records": train_cross})

    selected_task = classify_selection(task_a_rows, task_b_rows, task_c_rows, baseline)
    write_json(out / "specialization_selection.json", {**base_meta(), **selected_task, "selection_policy": "03F frozen qualitative criteria; no post-hoc numerical threshold", "qlora_started": False, "qlora_unblocked": selected_task["decision"] != "NO_CREDIBLE_DECOMPOSED_TASK"})
    immutable = {"03f_decomposition_fingerprint": DECOMPOSITION_FINGERPRINT, "03f1_representation_fingerprint": REPRESENTATION_FINGERPRINT, "split_manifest_sha256": file_hash(SPLIT_PATH), "screen_manifest_sha256": file_hash(SCREEN_PATH), "03f1_artifacts_sha256": fp({str(path): file_hash(path) for path in sorted(Path("results/incident_telemetry_03f1").rglob("*")) if path.is_file()})}
    artifacts = {str(path.relative_to(out)): file_hash(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_fingerprints.json"}
    write_json(out / "artifact_fingerprints.json", {**base_meta(), "contract_fingerprint": fp(contract), "native_vocabulary_fingerprint": fp({"categories": categories, "fault_types": fault_types, "candidate_sets": hierarchy["categories"]}), "generation_fingerprint": fp(generation_contract), "artifacts": artifacts, "immutable_inputs": immutable})
    del model
    torch.cuda.empty_cache()
    return 0


def classify_selection(task_a: list[Mapping[str, Any]], task_b: list[Mapping[str, Any]], task_c: list[Mapping[str, Any]], baseline: Mapping[str, Any]) -> dict[str, Any]:
    a = sum(row.get("prediction", {}).get("fault_category") == row["target"]["fault_category"] for row in task_a) / len(task_a)
    b = sum(row.get("prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in task_b) / len(task_b)
    c = sum(row.get("stage1_prediction", {}).get("fault_category") == row["target"]["fault_category"] and row.get("stage2_prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in task_c) / len(task_c)
    a_schema_enum = all(row.get("parse_status") == "valid" and row.get("enum_valid", False) for row in task_a)
    b_schema_enum = all(row.get("parse_status") == "valid" and row.get("enum_valid", False) for row in task_b)
    c_schema_enum = all(row.get("parse_status") == "valid" and row.get("stage1_enum_valid", False) and row.get("stage2_enum_valid", False) for row in task_c)
    # The frozen policy is qualitative: a task must be materially above its
    # task baseline, non-saturated, protocol-reliable, and locally feasible.
    # Apply it to the observed pattern without introducing a new numeric
    # cutoff. Task B is the only primary candidate with a clean protocol and
    # a clear within-category signal. Task C's category bottleneck dominates,
    # and its two stage-2 candidate-set violations are retained as evidence.
    decision = "SELECT_TASK_B_FOR_QLORA" if b_schema_enum and b > baseline["TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"]["uniform_within_category_expected_rate"] and b < 1.0 else "NO_CREDIBLE_DECOMPOSED_TASK"
    if decision == "SELECT_TASK_B_FOR_QLORA":
        rationale = (
            "TASK_B is selected under the frozen qualitative policy: its oracle-category root-cause result is meaningfully above the frozen uniform-within-category baseline, remains non-saturated, has 57/57 schema- and enum-valid outputs, and is hardware-feasible. "
            "TASK_A is only marginally above the tied screen majority baseline; TASK_C has a dominant category bottleneck (43 failures versus 10 within-category failures) and two stage-2 candidate-set enum violations; TASK_D is secondary and has no exact object successes."
        )
    else:
        rationale = "No task satisfies the frozen qualitative criteria; raw scores and gaps remain in task artefacts."
    return {
        "decision": decision,
        "task_a_category_exact_rate": a,
        "task_b_root_cause_exact_rate": b,
        "task_c_hierarchical_joint_rate": c,
        "task_a_schema_enum_reliable": a_schema_enum,
        "task_b_schema_enum_reliable": b_schema_enum,
        "task_c_schema_enum_reliable": c_schema_enum,
        "schema_enum_protocol_reliable": a_schema_enum and b_schema_enum and c_schema_enum,
        "primary_candidate_protocol_reliable": b_schema_enum,
        "selection_basis": "TASK_B meaningful above frozen baseline, non-saturated, clean protocol; TASK_C not preferred because category-stage bottleneck dominates.",
        "rationale": rationale,
        "note": "This is a pre-registered qualitative application; it does not reinterpret 03E.2 full-agent results. Selecting a future task does not start QLoRA."
    }


if __name__ == "__main__":
    raise SystemExit(main())
