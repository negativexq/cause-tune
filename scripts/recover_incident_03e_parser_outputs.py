#!/usr/bin/env python3
"""Recompute 03E derived metrics from already persisted model outputs.

This is an offline parser recovery for Qwen chat-template control tokens.  It
never loads a model, calls a provider, or creates new model output.  Raw
decoded strings remain in the JSONL records.
"""

from __future__ import annotations

import json
from pathlib import Path

from causetune.cloudopsbench.screening import compact_failure_category, parse_final_diagnosis, score_diagnoses


ROOT = Path("results/incident_telemetry_03e")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def recover_oracle(rows: list[dict]) -> tuple[list[dict], dict]:
    recovered = 0
    for row in rows:
        prediction, category = parse_final_diagnosis(str(row.get("raw_output", "")))
        if prediction is not None:
            recovered += int(row.get("prediction") is None)
            row["prediction"] = prediction
            row["parse_category"] = category
            row["parser_recovered"] = True
        else:
            row["parser_recovered"] = False
    metrics = score_diagnoses(rows)
    metrics["parser_recovery"] = {"control_token_cleanup": True, "recovered_predictions": recovered, "raw_outputs_preserved": True}
    return rows, metrics


def recover_closed(rows: list[dict]) -> tuple[list[dict], dict, dict]:
    recovered = 0
    for row in rows:
        found = None
        for event in row.get("events", []):
            prediction, category = parse_final_diagnosis(str(event.get("raw_output", "")))
            if prediction is not None:
                found = (prediction, category)
                break
        if found is not None:
            prediction, category = found
            recovered += int(row.get("prediction") is None)
            row["prediction"] = prediction
            row["parse_category"] = category
            row["failure_category"] = None
            row["failure_class"] = compact_failure_category(row)
            row["parser_recovered"] = True
        else:
            row["parser_recovered"] = False
    metrics = score_diagnoses(rows)
    metrics.update({
        "valid_tool_call_rate": sum(item.get("valid_tool_calls", 0) > 0 for item in rows) / len(rows) if rows else 0.0,
        "invalid_tool_call_rate": sum(item.get("invalid_tool_calls", 0) > 0 for item in rows) / len(rows) if rows else 0.0,
        "mean_tool_steps": sum(item.get("valid_tool_calls", 0) for item in rows) / len(rows) if rows else 0.0,
        "loop_rate": sum(item.get("failure_category") == "loop" for item in rows) / len(rows) if rows else 0.0,
        "max_step_exhaustion_rate": sum(item.get("failure_category") == "max_step_exhaustion" for item in rows) / len(rows) if rows else 0.0,
        "failure_categories": {
            key: sum(1 for item in rows if str(item.get("failure_class") or item.get("failure_category") or "unknown") == key)
            for key in sorted({str(item.get("failure_class") or item.get("failure_category") or "unknown") for item in rows})
        },
        "parser_recovery": {"control_token_cleanup": True, "recovered_predictions": recovered, "raw_outputs_preserved": True},
    })
    failure_analysis: dict[str, dict[str, dict[str, float | int]]] = {}
    for dimension in ("source_fault_category", "source_fault_type", "source_system", "upstream_difficulty", "metrics_available", "code_available"):
        buckets: dict[str, list[dict]] = {}
        for row in rows:
            buckets.setdefault(str(row.get(dimension)), []).append(row)
        failure_analysis[dimension] = {
            key: {"count": len(group), "joint_exact": sum(bool(item.get("prediction") and item["prediction"] == item["target"]) for item in group) / len(group)}
            for key, group in sorted(buckets.items())
        }
    failure_analysis["failure_categories"] = {}
    for row in rows:
        category = str(row.get("failure_class") or row.get("failure_category") or "unknown")
        failure_analysis["failure_categories"][category] = failure_analysis["failure_categories"].get(category, 0) + 1
    return rows, metrics, failure_analysis


def main() -> int:
    recovered_summary = {}
    for model_dir in sorted(ROOT.glob("Qwen__*")):
        closed_path = model_dir / "closed_loop_predictions.jsonl"
        oracle_path = model_dir / "oracle_evidence_predictions.jsonl"
        if not closed_path.is_file() or not oracle_path.is_file():
            raise FileNotFoundError(f"incomplete candidate artifacts: {model_dir}")
        closed, closed_metrics, failure_analysis = recover_closed(read_jsonl(closed_path))
        oracle, oracle_metrics = recover_oracle(read_jsonl(oracle_path))
        write_jsonl(closed_path, closed)
        write_json(model_dir / "closed_loop_metrics.json", {**json.loads((model_dir / "closed_loop_metrics.json").read_text()), **closed_metrics})
        write_json(model_dir / "closed_loop_failure_analysis.json", failure_analysis)
        write_jsonl(oracle_path, oracle)
        write_json(model_dir / "oracle_evidence_metrics.json", {**json.loads((model_dir / "oracle_evidence_metrics.json").read_text()), **oracle_metrics})
        recovered_summary[model_dir.name] = {"closed_loop_recovered": sum(bool(row.get("parser_recovered")) for row in closed), "oracle_recovered": sum(bool(row.get("parser_recovered")) for row in oracle)}
    write_json(ROOT / "parser_recovery.json", {"experiment": "03E", "model_execution": False, "provider_called": False, "test_model_facing": False, "raw_outputs_preserved": True, "candidates": recovered_summary})
    print(json.dumps(recovered_summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
