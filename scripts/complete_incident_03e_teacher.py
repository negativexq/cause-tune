#!/usr/bin/env python3
"""Complete the 03E 4B teacher-forced artifact after summary-only failure.

The preceding screening process completed model generation for this mode but
failed while serializing exception records.  This recovery reruns only the
missing 4B next-action mode; closed-loop and oracle results are never invoked.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import torch

from causetune.cloudopsbench import scan_corpus
from causetune.cloudopsbench.training_contract import normalize_trajectory
from causetune.cloudopsbench.screening import MODEL_IDS, SOURCE_REVISION, SPLIT_FINGERPRINT, build_validation_subsplit, source_case_group_id
from run_incident_03e_screen import (
    MODEL_REVISIONS,
    common_metadata,
    hardware_probe,
    load_model_metadata,
    model_loader,
    tokenize_histories,
    write_json,
    write_jsonl,
    run_teacher_forced,
)


def main() -> int:
    root = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
    out = Path("results/incident_telemetry_03e")
    split = json.loads(Path("results/incident_telemetry_03d/split_manifest.json").read_text())
    sub = json.loads((out / "validation_subsplit.json").read_text())
    if sub["fingerprint"] != build_validation_subsplit(split)["fingerprint"]:
        raise ValueError("validation screen manifest changed")
    scan = scan_corpus(root, source_revision=SOURCE_REVISION, fail_closed=True)
    groups = set(sub["screen_case_groups"])
    cases = [case for case in scan.cases if source_case_group_id(case) in groups]
    eligible_paths = {}
    for case in cases:
        for path in ("golden_path1", "golden_path2"):
            try:
                normalized = normalize_trajectory(root, case, path)
                if all(step["replay_status"] != "RESOLVED_FROM_GOLDEN_TRACE" for step in normalized["replay_steps"]):
                    eligible_paths[source_case_group_id(case)] = path
                    break
            except Exception:
                continue
    model_id = "Qwen/Qwen3.5-4B"
    model, tokenizer, metadata = model_loader(model_id, MODEL_REVISIONS[model_id])
    train_cases = [case for case in scan.cases if any(item["source_case_group"] == source_case_group_id(case) and item["split"] == "TRAIN" for item in split["entries"])]
    token_metrics = tokenize_histories(tokenizer, root, cases, train_cases, {item["source_case_group"]: item["split"] for item in split["entries"]})
    records, metrics = run_teacher_forced(model, tokenizer, root, cases, eligible_paths)
    model_out = out / "Qwen__Qwen3.5-4B"
    write_jsonl(model_out / "next_action_predictions.jsonl", records)
    write_json(model_out / "next_action_metrics.json", {**common_metadata(root), **metrics, "recovery": "4B teacher-forced generation recovery after post-generation summary bug; closed-loop/oracle not rerun"})
    write_json(model_out / "tokenization_metrics.json", {**common_metadata(root), **token_metrics})
    hardware = hardware_probe(model, tokenizer, token_metrics)
    write_json(model_out / "hardware_metrics.json", {**common_metadata(root), "gpu": torch.cuda.get_device_name(), "software": {"torch": torch.__version__, "transformers": __import__("transformers").__version__, "bitsandbytes": __import__("bitsandbytes").__version__}, **hardware})
    metadata["tokenization"] = token_metrics
    write_json(model_out / "model_metadata.json", {**common_metadata(root), **metadata})
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({"status": "pass", "model": model_id, "teacher_records": len(records), "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
