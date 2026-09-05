"""Finalize 03G metrics from already persisted raw generations.

No language model is loaded and no generation is performed here. This is a
deterministic post-inference accounting step; raw prediction JSONL files are
read but never rewritten.
"""

from __future__ import annotations

import json
from pathlib import Path

from transformers import AutoTokenizer

import run_incident_03g as runner
from causetune.cloudopsbench.scanner import scan_corpus
from causetune.cloudopsbench.training_contract import source_case_group_id


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    out = runner.OUT
    paths = [out / "task_a/predictions.jsonl", out / "task_b/predictions.jsonl", out / "task_c/predictions.jsonl", out / "task_d/predictions.jsonl"]
    if any(not path.is_file() for path in paths):
        raise RuntimeError("03G raw prediction files are incomplete")
    split, groups, hierarchy, sufficiency, all_packages = runner.load_frozen_inputs()
    categories, _ = runner.vocabularies(hierarchy)
    scan = scan_corpus(runner.SOURCE_ROOT, source_revision=runner.SOURCE_REVISION, fail_closed=True)
    by_group = {source_case_group_id(case): case for case in scan.cases}
    packages = runner.package_by_group(groups, all_packages)
    task_a = read_jsonl(paths[0]); task_b = read_jsonl(paths[1]); task_c = read_jsonl(paths[2]); task_d = read_jsonl(paths[3])
    if not all(len(rows) == 57 for rows in (task_a, task_b, task_c, task_d)):
        raise RuntimeError("03G raw prediction count is not 57 per task")
    write_json(out / "task_a/metrics.json", {**runner.base_meta(), **runner.task_metrics(task_a, "TASK_A", hierarchy, categories), "baseline_reference": "baseline_verification.json::TASK_A"})
    write_json(out / "task_a/confusion_matrix.json", {category: {pred: sum(row.get("prediction", {}).get("fault_category") == pred and row["target"]["fault_category"] == category for row in task_a) for pred in categories} for category in categories})
    write_json(out / "task_b/metrics.json", {**runner.base_meta(), **runner.task_metrics(task_b, "TASK_B", hierarchy, categories), "baseline_reference": "baseline_verification.json::TASK_B_ORACLE_CATEGORY_ROOT_CAUSE"})
    write_json(out / "task_b/per_category.json", runner.per_group_metrics(task_b, "TASK_B", hierarchy, categories)["category"])
    candidate_size_rows = {}
    for size in sorted({len(hierarchy["categories"][row["authoritative_category_context"]]) for row in task_b}):
        subset = [row for row in task_b if len(hierarchy["categories"][row["authoritative_category_context"]]) == size]
        candidate_size_rows[str(size)] = {"count": len(subset), "root_cause_exact_count": sum(row.get("prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in subset), "root_cause_exact_rate": sum(row.get("prediction", {}).get("root_cause") == row["target"]["root_cause"] for row in subset) / len(subset)}
    write_json(out / "task_b/per_candidate_set_size.json", {"task": "TASK_B", "definition": "descriptive slices by frozen authoritative category candidate-set size", "by_candidate_set_size": candidate_size_rows})
    write_json(out / "task_c/metrics.json", {**runner.base_meta(), **runner.task_metrics(task_c, "TASK_C", hierarchy, categories), "stage1_source": "task_a/predictions.jsonl; no regeneration", "baseline_reference": "baseline_verification.json::TASK_C_HIERARCHICAL_SELF_PREDICTED"})
    write_json(out / "task_c/error_decomposition.json", {**runner.base_meta(), "category_bottleneck_count": sum(row["stage1_prediction"].get("fault_category") != row["target"]["fault_category"] for row in task_c), "within_category_discrimination_failure_count": sum(row["stage1_prediction"].get("fault_category") == row["target"]["fault_category"] and row.get("stage2_prediction", {}).get("root_cause") != row["target"]["root_cause"] for row in task_c), "stage1_protocol_failure_count": sum(row["stage1_protocol_failure"] for row in task_c), "stage2_invalid_after_valid_stage1_count": sum(row["stage1_prediction"].get("fault_category") == row["target"]["fault_category"] and not row["stage2_enum_valid"] for row in task_c)})
    write_json(out / "task_d/metrics.json", {**runner.base_meta(), **runner.task_metrics(task_d, "TASK_D", hierarchy, categories)})

    registry = read_json(out / "model_registry.json")
    tokenizer = AutoTokenizer.from_pretrained(str(registry["snapshot_path"]), local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = runner.task_prompts(categories, hierarchy)
    token_rows = {task: [] for task in ("TASK_A", "TASK_B", "TASK_C_STAGE1", "TASK_C_STAGE2", "TASK_D")}
    package_tokens = []
    for index, group in enumerate(groups):
        package = packages[group]
        package_tokens.append(len(tokenizer(json.dumps(package, ensure_ascii=False, sort_keys=True, separators=(",", ":")), add_special_tokens=False)["input_ids"]))
        predicted_category = task_a[index].get("prediction", {}).get("fault_category")
        messages = (
            ("TASK_A", runner.messages_for("TASK_A", package, prompts)),
            ("TASK_B", runner.messages_for("TASK_B", package, prompts, category_context=task_b[index]["authoritative_category_context"], candidates=hierarchy["categories"][task_b[index]["authoritative_category_context"]])),
            ("TASK_C_STAGE1", runner.messages_for("TASK_C_STAGE1", package, prompts)),
            ("TASK_C_STAGE2", runner.messages_for("TASK_C_STAGE2", package, prompts, predicted_category=predicted_category or "<INVALID_STAGE1_CATEGORY>", candidates=hierarchy["categories"].get(predicted_category, []))),
            ("TASK_D", runner.messages_for("TASK_D", package, prompts)),
        )
        for task, value in messages:
            token_rows[task].append(len(tokenizer.apply_chat_template(value, tokenize=True, add_generation_prompt=True, enable_thinking=False)["input_ids"]))
    context_limit = int(read_json(Path("results/incident_telemetry_03e/Qwen__Qwen3.5-2B/model_metadata.json")).get("native_context_length", 262144))
    oom_count = sum(bool(row.get("failure") and "out of memory" in str(row.get("failure")).lower()) for rows in (task_a, task_b, task_c, task_d) for row in rows)
    write_json(out / "tokenization_metrics.json", {**runner.base_meta(), **{task: runner.stats(values) for task, values in token_rows.items()}, "package_input_tokens": runner.stats(package_tokens), "context_limit_tokens": context_limit, "context_overflow_count": sum(value + runner.MAX_NEW_TOKENS > context_limit for values in token_rows.values() for value in values), "truncation_count": 0, "oom_count": oom_count, "measurement": "exact local tokenizer apply_chat_template; no truncation; Task C Stage 1 is Task A persisted prediction", "tokenizer_loaded": True, "model_loaded": False})

    load = registry.get("load", {})
    all_rows = task_a + task_b + task_c + task_d
    task_wall = {"TASK_A": sum(row.get("wall_seconds", 0.0) for row in task_a), "TASK_B": sum(row.get("wall_seconds", 0.0) for row in task_b), "TASK_C_STAGE2": sum(row.get("wall_seconds", 0.0) for row in task_c), "TASK_D": sum(row.get("wall_seconds", 0.0) for row in task_d)}
    write_json(out / "hardware_metrics.json", {**runner.base_meta(), "software": {"torch": registry.get("torch_version"), "transformers": registry.get("transformers_version"), "bitsandbytes": registry.get("bitsandbytes_version"), "cuda": registry.get("cuda_version")}, "gpu_model": registry.get("gpu_model"), "physical_vram_bytes": registry.get("physical_vram_bytes"), "model_load": load, "peak_allocated_bytes": max(row.get("peak_allocated_bytes", 0) for row in all_rows), "peak_reserved_bytes": max(row.get("peak_reserved_bytes", 0) for row in all_rows), "task_wall_seconds": task_wall, "total_wall_seconds": sum(task_wall.values()) + load.get("model_load_seconds", 0.0), "total_generation_count": 228, "oom_count": oom_count, "context_overflow_count": sum(value + runner.MAX_NEW_TOKENS > context_limit for values in token_rows.values() for value in values), "tokens_per_second": {task: (sum(row.get("generated_tokens", 0) for row in rows) / sum(row.get("wall_seconds", 0.0) for row in rows)) if sum(row.get("wall_seconds", 0.0) for row in rows) else None for task, rows in (("TASK_A", task_a), ("TASK_B", task_b), ("TASK_C_STAGE2", task_c), ("TASK_D", task_d))}})

    bucket_analysis = {}
    for task, rows in (("TASK_A", task_a), ("TASK_B", task_b), ("TASK_C", task_c), ("TASK_D", task_d)):
        bucket_analysis[task] = {bucket: runner.task_metrics([row for row in rows if row["evidence_bucket"] == bucket], task, hierarchy, categories) for bucket in ("FULL", "PARTIAL", "ZERO")}
    write_json(out / "evidence_bucket_analysis.json", {**runner.base_meta(), "interpretation": "analysis slices only; ZERO does not imply no diagnostic information", "process_label_coverage_is_not_classification_sufficiency": True, "by_task": bucket_analysis})
    train_support = read_json(Path("results/incident_telemetry_03f/train_label_support.json"))["by_root_cause"]
    train_cross = []
    for group, b_row, c_row in zip(groups, task_b, task_c):
        root = b_row["target"]["root_cause"]
        train_cross.append({"source_case_group": group, "root_cause": root, "category": b_row["target"]["fault_category"], "train_count": train_support[root]["count"], "low_support": train_support[root]["count"] < 3, "task_b_exact": b_row.get("prediction", {}).get("root_cause") == root, "task_c_category_exact": c_row.get("stage1_prediction", {}).get("fault_category") == c_row["target"]["fault_category"], "task_c_root_exact": c_row.get("stage2_prediction", {}).get("root_cause") == root})
    write_json(out / "train_support_cross_analysis.json", {**runner.base_meta(), "low_support_definition": "TRAIN count <3", "low_support_labels": sorted(root for root, item in train_support.items() if item["count"] < 3), "records": train_cross})
    baseline = read_json(out / "baseline_verification.json")
    max_category_count = max(baseline["TASK_A"]["category_counts"].values())
    baseline["TASK_A"]["majority_categories"] = sorted(category for category, count in baseline["TASK_A"]["category_counts"].items() if count == max_category_count)
    baseline["TASK_A"]["majority_tie"] = len(baseline["TASK_A"]["majority_categories"]) > 1
    write_json(out / "baseline_verification.json", baseline)
    write_json(out / "specialization_selection.json", {**runner.base_meta(), **runner.classify_selection(task_a, task_b, task_c, baseline), "selection_policy": "03F frozen qualitative criteria; no post-hoc numerical threshold", "qlora_started": False})

    immutable = {"03f_decomposition_fingerprint": runner.DECOMPOSITION_FINGERPRINT, "03f1_representation_fingerprint": runner.REPRESENTATION_FINGERPRINT, "split_manifest_sha256": runner.file_hash(runner.SPLIT_PATH), "screen_manifest_sha256": runner.file_hash(runner.SCREEN_PATH), "03f1_artifacts_sha256": runner.fp({str(path): runner.file_hash(path) for path in sorted(Path("results/incident_telemetry_03f1").rglob("*")) if path.is_file()})}
    artifacts = {str(path.relative_to(out)): runner.file_hash(path) for path in sorted(out.rglob("*")) if path.is_file() and path.name != "artifact_fingerprints.json"}
    write_json(out / "artifact_fingerprints.json", {**runner.base_meta(), "contract_fingerprint": runner.fp(read_json(out / "experiment_contract.json")), "native_vocabulary_fingerprint": read_json(out / "native_vocabulary.json").get("fingerprint"), "generation_fingerprint": read_json(out / "generation_contract.json").get("generation_fingerprint"), "artifacts": artifacts, "immutable_inputs": immutable, "raw_predictions_preserved": True, "finalized_without_regeneration": True})
    print(json.dumps({"task_a": read_json(out / "task_a/metrics.json"), "task_b": read_json(out / "task_b/metrics.json"), "task_c": read_json(out / "task_c/metrics.json"), "task_d": read_json(out / "task_d/metrics.json"), "decision": read_json(out / "specialization_selection.json")["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
