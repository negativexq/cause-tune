#!/usr/bin/env python3
"""Rerun only closed-loop after the deterministic optional-source fix.

The oracle and prompt are unchanged.  This entry point is used because the
first closed-loop pass exposed a replay contract gap for absent code sources;
it does not regenerate oracle evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

from run_incident_03e2 import (  # type: ignore
    SPLIT_FINGERPRINT,
    TOOL_SCHEMAS,
    _fp,
    load_model,
    meta,
    read_json,
    run_closed_loop,
    scan_corpus,
    source_case_group_id,
    tool_matrix,
    write_json,
    write_jsonl,
)


def main() -> int:
    root = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
    out = Path("results/incident_telemetry_03e2")
    split = read_json(Path("results/incident_telemetry_03d/split_manifest.json"))
    if split.get("fingerprint") != SPLIT_FINGERPRINT:
        raise ValueError("frozen split changed")
    manifest = read_json(Path("results/incident_telemetry_03e/validation_subsplit.json"))
    scan = scan_corpus(root, source_revision="03c415e5709297432282fbbfd499f1bca0f8c347", fail_closed=True)
    by_group = {source_case_group_id(case): case for case in scan.cases}
    cases = [by_group[group] for group in sorted(manifest["screen_case_groups"])]
    tools = tool_matrix(root, cases)
    write_json(out / "tool_executability.json", {**meta(root), **tools, "tool_schema_fingerprint": _fp(TOOL_SCHEMAS), "corrected_after_initial_contract_audit": True})
    contract = read_json(out / "v2_contract.json")
    contract["tool_contract"] = tools
    contract["contract_correction"] = "Optional case-local source absence now returns an explicit deterministic unavailable result; no fake observation and no ground-truth repair. Prompt and vocabulary unchanged."
    write_json(out / "v2_contract.json", contract)
    model, tokenizer, _ = load_model()
    closed, _ = run_closed_loop(model, tokenizer, root, cases)
    write_jsonl(out / "closed_loop_predictions.jsonl", closed)
    print(json.dumps({"closed_loop_records": len(closed), "environment_contract_failures": sum(row["environment_contract_failures"] for row in closed)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
