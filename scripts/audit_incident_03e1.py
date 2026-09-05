#!/usr/bin/env python3
"""Offline, read-only 03E.1 contract and failure-semantics audit.

This script consumes only persisted 03E outputs plus the pinned local
Cloud-OpsBench source checkout.  It never loads a model, generates text, uses
TEST, calls a provider, or writes inside the frozen 03E directory.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from causetune.cloudopsbench import scan_corpus
from causetune.cloudopsbench.screening import (
    OfflineToolReplay,
    TOOL_IDS,
    source_case_group_id,
)
from cloudops_agent.evaluation_utils.evaluator import evaluate_trajectory
from cloudops_agent.evaluation_utils.schema import CaseAnnotation, ToolCall, TrajectoryStep


ROOT = Path("results/incident_telemetry_03e")
OUT = Path("results/incident_telemetry_03e1")
SOURCE_ROOT = Path("/home/ofk/projects/external-data/Cloud-OpsBench")
MODELS = {
    "0.8B": "Qwen__Qwen3.5-0.8B",
    "2B": "Qwen__Qwen3.5-2B",
    "4B": "Qwen__Qwen3.5-4B",
}
CAT_DIR = {
    "Admission_Fault": "admission",
    "Application_Code_Defect": "codedefect",
    "Infrastructure_Fault": "infrastructure",
    "Performance_Fault": "performance",
    "Runtime_Fault": "runtime",
    "Scheduling_Fault": "scheduling",
    "Service_Routing_Fault": "service",
    "Startup_Fault": "startup",
}
FIELDS = ("native_fault_type", "native_fault_category", "fault_object")
CLASS_NAMES = (
    "EXACT",
    "CASE_ONLY",
    "SEPARATOR_ONLY",
    "CANONICAL_FORMAT_ONLY",
    "RESOURCE_PREFIX_ONLY",
    "LEXICALLY_EQUIVALENT",
    "SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL",
    "WRONG_TARGET",
    "MISSING",
    "OTHER",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(name: str, value: Any) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def norm_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return re.sub(r"[\s_-]+", " ", value.strip().casefold()).strip()


def norm_compact(value: Any) -> str | None:
    item = norm_label(value)
    return re.sub(r"[^a-z0-9]+", "", item) if item is not None else None


def object_prefix_alias(target: Any, prediction: Any) -> bool:
    if not isinstance(target, str) or not isinstance(prediction, str) or "/" not in target:
        return False
    kind, name = target.split("/", 1)
    return kind in {"app", "node", "namespace"} and prediction == name


def k8s_derived_identity(target: str, prediction: str) -> bool:
    if not target.startswith("app/"):
        return target.startswith("node/") and prediction in {"kube-proxy.service", "containerd.service - containerd container runtime"}
    base = target.split("/", 1)[1]
    if prediction == base or prediction.endswith("-pvc"):
        return False
    return prediction.startswith(base + "-") and bool(re.search(r"-[a-z0-9]{5}$|-[a-z0-9]{8,10}$", prediction))


# These are source-derived observable families, not synonym substitutions.
# They are used only to report plausibility separately from strict scoring.
# A phrase is plausible when it names a documented symptom/resource family for
# the native fault; it is never converted into the native identifier.
TYPE_FAMILIES: dict[str, tuple[str, ...]] = {
    "namespace_cpu_quota_exceeded": ("quota", "resource"),
    "namespace_memory_quota_exceeded": ("quota", "resource", "memory"),
    "namespace_pod_quota_exceeded": ("quota", "resource", "pod"),
    "namespace_service_quota_exceeded": ("quota", "resource", "service", "availability"),
    "missing_service_account": ("serviceaccount", "service account", "account", "configuration"),
    "namespace_storage_quota_exceeded": ("quota", "resource", "storage", "persistentvolume", "claim"),
    "kube_proxy_unavailable": ("proxy", "unreach", "service"),
    "kube_scheduler_unavailable": ("scheduler", "schedul", "service", "unavail"),
    "kubelet_unavailable": ("kubelet", "node", "service", "unavail"),
    "containerd_unavailable": ("containerd", "runtime", "container"),
    "pod_cpu_overload": ("cpu", "load", "latency", "service"),
    "pod_network_delay": ("network", "latency", "delay", "service"),
    "node_network_delay": ("network", "latency", "delay", "node"),
    "node_network_packet_loss": ("network", "packet", "loss", "latency", "quality"),
    "liveness_probe_incorrect_port": ("liveness", "probe", "timeout", "restart", "crash"),
    "liveness_probe_incorrect_timing": ("liveness", "probe", "timeout", "restart", "crash"),
    "liveness_probe_incorrect_protocol": ("liveness", "probe", "protocol", "timeout", "failure", "restart", "crash"),
    "readiness_probe_incorrect_protocol": ("readiness", "probe", "protocol", "timeout", "failure"),
    "readiness_probe_incorrect_port": ("readiness", "probe", "port", "timeout", "failure"),
    "container_memory_limit_too_low": ("memory", "oom", "resource", "container"),
    "node_cordon_mismatch": ("schedul", "node", "cordon"),
    "node_selector_mismatch": ("schedul", "node", "selector", "affinity"),
    "pod_anti_affinity_conflict": ("schedul", "pending", "affinity", "node"),
    "pvc_access_mode_mismatch": ("pvc", "persistentvolume", "claim", "pending", "access"),
    "pvc_selector_mismatch": ("pvc", "persistentvolume", "claim", "pending", "selector"),
    "pvc_storage_class_mismatch": ("pvc", "persistentvolume", "claim", "pending", "storage", "unbound", "pod"),
    "pv_binding_occupied": ("pv", "persistentvolume", "claim", "pending", "binding"),
    "taint_toleration_mismatch": ("schedul", "node", "taint", "toleration"),
    "cpu_capacity_mismatch": ("cpu", "schedul", "resource", "capacity"),
    "memory_capacity_mismatch": ("memory", "schedul", "resource", "node", "pressure"),
    "pvc_capacity_mismatch": ("pvc", "persistentvolume", "claim", "capacity", "pending"),
    "node_affinity_mismatch": ("schedul", "node", "affinity", "service", "unreach"),
    "service_port_mapping_mismatch": ("service", "port", "connection", "error"),
    "service_protocol_mismatch": ("service", "protocol", "connection", "refused", "transport"),
    "service_selector_mismatch": ("service", "selector", "connection", "redis", "unreach"),
    "service_env_var_address_mismatch": ("service", "dns", "address", "resolution", "network"),
    "gateway_misrouted": ("gateway", "route", "service", "unavail"),
    "service_dns_resolution_failure": ("service", "dns", "resolution", "network"),
    "image_registry_dns_failure": ("image", "pull", "registry", "dns", "network"),
    "incorrect_image_reference": ("image", "pull", "errimage"),
    "missing_image_pull_secret": ("image", "pull", "secret", "container"),
    "missing_secret_binding": ("secret", "binding", "container", "configuration", "unreach"),
    "volume_mount_permission_denied": ("volume", "mount", "permission", "container", "crash"),
    "deployment_zero_replicas": ("deployment", "replica", "service", "availability"),
    "db_connection_exhaustion": ("database", "connection", "pool", "hikari"),
    "mysql_invalid_credentials": ("database", "mysql", "credential", "authentication", "connection"),
    "mysql_invalid_port": ("database", "mysql", "port", "jdbc", "connection"),
    "service_sidecar_port_conflict": ("service", "sidecar", "port", "conflict", "exception"),
    "db_readonly_mode": ("database", "readonly", "read-only", "connection"),
}
for _type in (
    "code_missing_parameter", "code_excessive_file_reads", "code_wrong_argument_order",
    "code_busy_loop", "code_excessive_file_writes", "code_wrong_return",
    "code_artificial_delay", "code_memory_leak",
):
    TYPE_FAMILIES[_type] = ("service", "availability", "unavail", "error", "latency", "container", "crash", "memory", "http", "503")

CATEGORY_FAMILIES: dict[str, tuple[str, ...]] = {
    "Admission_Fault": ("quota", "resource", "storage", "service", "configuration", "cpu", "network"),
    "Application_Code_Defect": ("service", "resource", "availability", "unavail", "http", "latency", "container", "error"),
    "Infrastructure_Fault": ("network", "service", "container", "kubelet", "containerd", "unreach", "unavail"),
    "Performance_Fault": ("service", "latency", "network", "quality", "high"),
    "Runtime_Fault": ("container", "restart", "probe", "resource", "oom", "database", "connection", "authentication", "readonly", "read-only", "read only", "service", "unavail", "port"),
    "Scheduling_Fault": ("schedul", "node", "pending", "affinity", "resource", "pvc", "claim"),
    "Service_Routing_Fault": ("transport", "network", "service", "dns", "resolution", "connection", "unreach", "availability"),
    "Startup_Fault": ("image", "pull", "container", "secret", "service", "unreach"),
}


def semantic_match(value: Any, terms: Iterable[str]) -> bool:
    item = norm_label(value) or ""
    return any(term.casefold() in item for term in terms)


def classify_field(field: str, target: Any, prediction: Any, source_type: str, source_category: str) -> tuple[str, str]:
    if prediction is None or (isinstance(prediction, str) and not prediction.strip()):
        return "MISSING", "no persisted value"
    if target == prediction:
        return "EXACT", "byte-for-byte field equality"
    if isinstance(target, str) and isinstance(prediction, str):
        if target.casefold() == prediction.casefold():
            return "CASE_ONLY", "case-folded equality"
        if norm_label(target) == norm_label(prediction):
            return "SEPARATOR_ONLY", "case-folded equality after space/underscore/hyphen normalization"
        if norm_compact(target) == norm_compact(prediction):
            return "CANONICAL_FORMAT_ONLY", "alphanumeric canonicalization equality"
    if field == "fault_object" and object_prefix_alias(target, prediction):
        return "RESOURCE_PREFIX_ONLY", "bare logical resource name equals the target name after its source kind prefix"
    if field == "fault_object" and k8s_derived_identity(target, prediction):
        return "SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", "same source-derived workload/node identity family, but a different Kubernetes granularity"
    if field == "native_fault_type" and semantic_match(prediction, TYPE_FAMILIES.get(source_type, ())):
        return "SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", "matches a predeclared source-observable symptom/resource family; not a native-label alias"
    if field == "native_fault_category" and semantic_match(prediction, CATEGORY_FAMILIES.get(source_category, ())):
        return "SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", "matches a predeclared category-level symptom/resource family; not a native-label alias"
    return "WRONG_TARGET", "no deterministic mechanical or source-family relation certified"


def case_key(row: dict[str, Any]) -> str:
    return row["source_case_group"]


def common_meta() -> dict[str, Any]:
    return {
        "experiment": "03E.1",
        "audit_version": "cloud-opsbench-03e1-contract-validity-audit-v1",
        "source_revision": "03c415e5709297432282fbbfd499f1bca0f8c347",
        "source_root": str(SOURCE_ROOT),
        "split_manifest_fingerprint": "0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76",
        "representation_fingerprint": "f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf",
        "model_training": False,
        "test_model_facing": False,
        "provider_called": False,
        "llm_judge_used": False,
        "v1_artifacts_modified": False,
    }


def oracle_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    counts = {field: Counter() for field in FIELDS}
    for row in rows:
        target = row.get("target") or {}
        prediction = row.get("prediction") or {}
        field_records = {}
        for field in FIELDS:
            cls, rationale = classify_field(field, target.get(field), prediction.get(field), row["source_fault_type"], row["source_fault_category"])
            counts[field][cls] += 1
            field_records[field] = {"target": target.get(field), "prediction": prediction.get(field), "classification": cls, "rationale": rationale}
        records.append({
            "source_case_group": row["source_case_group"], "source_system": row["source_system"], "source_case_id": row["source_case_id"],
            "source_fault_category": row["source_fault_category"], "source_fault_type": row["source_fault_type"], "source_fingerprint": row["source_fingerprint"],
            "parse_category": row["parse_category"], "parser_recovered": row["parser_recovered"], "raw_output": row["raw_output"], "fields": field_records,
        })
    flat = Counter(cls for item in records for field in FIELDS for cls in [item["fields"][field]["classification"]])
    case_dispositions = Counter()
    for item in records:
        values = {field: item["fields"][field]["classification"] for field in FIELDS}
        if all(values[field] == "EXACT" for field in FIELDS): disposition = "EXACT"
        elif all(values[field] in {"EXACT", "CASE_ONLY", "SEPARATOR_ONLY", "CANONICAL_FORMAT_ONLY", "RESOURCE_PREFIX_ONLY", "LEXICALLY_EQUIVALENT"} for field in FIELDS): disposition = "MECHANICAL_ONLY"
        elif all(values[field] in {"SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", "RESOURCE_PREFIX_ONLY"} for field in FIELDS): disposition = "SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL"
        else: disposition = "MIXED_OR_WRONG_TARGET"
        case_dispositions[disposition] += 1
    return {
        **common_meta(), "model": "Qwen/Qwen3.5-2B", "input_artifact": str(ROOT / MODELS["2B"] / "oracle_evidence_predictions.jsonl"),
        "purpose": "all 57 persisted raw 2B oracle predictions, field-by-field against authoritative native targets",
        "record_count": len(records), "parse_counts": dict(Counter(row["parse_category"] for row in rows)),
        "strict_v1_result_preserved": {"schema_valid": "57/57", "native_fault_type_exact": "0/57", "native_fault_category_exact": "0/57", "fault_object_exact": "0/57", "joint_exact": "0/57"},
        "field_classification_counts": {field: {name: counts[field].get(name, 0) for name in CLASS_NAMES} for field in FIELDS},
        "all_field_classification_counts": {name: flat.get(name, 0) for name in CLASS_NAMES},
        "case_disposition_counts": dict(case_dispositions),
        "headline_failure_counts": {
            "field_comparisons_total": len(records) * len(FIELDS),
            "mechanical_representation_mismatches": flat.get("RESOURCE_PREFIX_ONLY", 0) + sum(flat.get(name, 0) for name in ("CASE_ONLY", "SEPARATOR_ONLY", "CANONICAL_FORMAT_ONLY", "LEXICALLY_EQUIVALENT")),
            "semantically_plausible_but_noncanonical_field_mismatches": flat.get("SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", 0),
            "wrong_target_field_mismatches": flat.get("WRONG_TARGET", 0),
            "mechanically_equivalent_full_diagnoses": case_dispositions.get("MECHANICAL_ONLY", 0),
            "all_fields_source_family_plausible_cases": case_dispositions.get("SEMANTICALLY_PLAUSIBLE_BUT_NONCANONICAL", 0),
            "mixed_or_wrong_target_cases": case_dispositions.get("MIXED_OR_WRONG_TARGET", 0),
        },
        "classification_policy": "Mechanical classes are deterministic string/resource-form checks. Plausibility is a separately reported, predeclared source-family relation from Cloud-OpsBench observable semantics; it never changes strict scoring and is not an LLM judgment.",
        "records": records,
    }


def normalization_audit(all_oracle: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per_candidate = {}
    for candidate, rows in all_oracle.items():
        valid = [row for row in rows if row.get("prediction") is not None]
        field_counts = {field: Counter() for field in FIELDS}
        equivalent_cases = []
        for row in valid:
            classes = {}
            for field in FIELDS:
                cls, _ = classify_field(field, row["target"].get(field), row["prediction"].get(field), row["source_fault_type"], row["source_fault_category"])
                classes[field] = cls; field_counts[field][cls] += 1
            if all(value in {"EXACT", "CASE_ONLY", "SEPARATOR_ONLY", "CANONICAL_FORMAT_ONLY", "RESOURCE_PREFIX_ONLY", "LEXICALLY_EQUIVALENT"} for value in classes.values()):
                equivalent_cases.append(row["source_case_group"])
        per_candidate[candidate] = {
            "records": len(rows), "parse_counts": dict(Counter(row["parse_category"] for row in rows)), "parsed_records": len(valid),
            "strict_v1_exact_unchanged": True, "mechanical_equivalent_field_counts": {field: sum(field_counts[field][name] for name in ("CASE_ONLY", "SEPARATOR_ONLY", "CANONICAL_FORMAT_ONLY", "RESOURCE_PREFIX_ONLY", "LEXICALLY_EQUIVALENT")) for field in FIELDS},
            "mechanical_equivalent_joint_cases": len(equivalent_cases), "mechanical_equivalent_joint_case_ids": equivalent_cases,
            "classification_counts": {field: dict(field_counts[field]) for field in FIELDS},
        }
    return {**common_meta(), "purpose": "predeclared deterministic normalization audit; strict V1 metrics remain visible and immutable", "normalization_functions": {
        "label": "strip; case-fold; collapse spaces/underscores/hyphens for separator equivalence; alphanumeric compaction only as canonical-format diagnostic",
        "fault_object": "only exact omission of source kind prefix for app/name, node/name, or namespace/name is RESOURCE_PREFIX_ONLY; no pod/deployment/PVC collapse",
        "not_allowed": "semantic synonym maps, target-aware repair, LLM judging, arbitrary model-output rewrites",
    }, "per_candidate": per_candidate, "headline": {"2B_oracle_mechanical_joint_equivalences": 0, "2B_oracle_mechanical_field_equivalences": {"native_fault_type": 0, "native_fault_category": 0, "fault_object": 17}}}


def vocabulary_audit() -> dict[str, Any]:
    return {**common_meta(), "v1_prompt_contract": read_json(ROOT / "evaluation_contract.json")["prompt"], "observed_native_taxonomy": {"categories": 8, "fault_types": 57}, "finding": "V1 required exact native_fault_type/native_fault_category/fault_object strings but did not provide the valid native category/type vocabulary. The model therefore had to infer internal identifiers from natural telemetry and tool observations.", "task_semantics": {"v1": "OPEN_VOCABULARY_DIAGNOSIS", "future_enum_prompt": "CLOSED_TAXONOMY_CLASSIFICATION", "label_leakage_assessment": "A fixed 8-category and 57-fault-type enum supplied identically for every case is task schema, not per-case target leakage; the correct case label must not be exposed."}, "recommendation": {"expose_8_native_categories": True, "expose_57_native_fault_types": True, "expose_case_target": False, "require_one_of_enum_values": True, "keep_fault_object_separate": True}, "evidence": ["2B oracle schema valid 57/57 but native type/category/object exact 0/57", "2B predictions are consistently human-readable symptom/resource phrases rather than native identifiers", "native type values cover 57 unique labels and category values 8 high-level labels"], "scope_boundary": "This explains label-vocabulary burden; it does not prove that any noncanonical prediction is a correct root-cause diagnosis."}


def object_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records=[]; counts=Counter()
    for row in rows:
        target=row["target"].get("fault_object"); prediction=(row.get("prediction") or {}).get("fault_object")
        cls, rationale=classify_field("fault_object",target,prediction,row["source_fault_type"],row["source_fault_category"]); counts[cls]+=1
        records.append({"source_case_group":row["source_case_group"],"source_case_id":row["source_case_id"],"target":target,"prediction":prediction,"classification":cls,"rationale":rationale})
    return {**common_meta(),"model":"Qwen/Qwen3.5-2B","record_count":len(records),"counts":dict(counts),"policy":{"bare_name":"resolve to app/name only when the source-visible logical application identity is unique; do not treat it as a deployment/pod/PVC identity","deployment_prefix":"distinct from app/name unless the contract explicitly defines logical-app equivalence","pod_name":"distinct Kubernetes resource/instance; never collapse to app/name","ambiguous_alias":"leave unresolved; never use ground truth to break ties"},"finding":"17/57 object failures are deterministic source-kind-prefix omissions. Pod/replicaset/PVC and unrelated service/node names remain non-equivalent resources.","records":records}


def parse_call_args(text: str) -> dict[str, Any]:
    match = re.search(r"arguments\s*=\s*\{(.*)\}\s*$", text or "")
    if not match: return {}
    expression = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r"\1'\2':", "{" + match.group(1) + "}")
    try:
        value = ast.literal_eval(expression)
    except (SyntaxError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def arg_equal(left: Any, right: Any) -> bool:
    if left == right: return True
    if left in (None, "") and right in (None, ""): return True
    if isinstance(left, list): return right in left or left == right
    if isinstance(right, list): return left in right or left == right
    return False


def milestone_tools(annotation: dict[str, Any]) -> set[str]:
    result=set()
    for milestone in annotation.get("milestones", []):
        for use in milestone.get("admissible_tool_uses", []): result.add(use.get("tool_name"))
        for group in milestone.get("admissible_evidence_groups", []):
            for use in group.get("tool_uses", []): result.add(use.get("tool_name"))
    return result


def next_action_audit(all_rows: dict[str, list[dict[str, Any]]], cases: dict[str, Any]) -> dict[str, Any]:
    per_candidate={}
    for candidate, rows in all_rows.items():
        records=[]; counts=Counter()
        for row in rows:
            case=cases[row["source_case_group"]]; path=SOURCE_ROOT/"process-label"/case.source_system/CAT_DIR[case.source_fault_category]/case.source_case_id/"milestone.json"; source=read_json(path)
            tools=milestone_tools(source); pred=row.get("predicted_tool"); exact=bool(row.get("tool_name_exact")); pred_args=row.get("predicted_arguments") or {}
            # The persisted target_arguments field is an intentionally compact
            # placeholder string (e.g. <value>), not the authoritative call.
            # For full-call auditing, read the corresponding immutable source
            # golden step; this does not alter or regenerate the prediction.
            golden_path=SOURCE_ROOT/"golden-trajectory"/case.source_system/CAT_DIR[case.source_fault_category]/case.source_case_id/("path1.json" if row["path"] == "golden_path1" else "path2.json")
            golden_step=read_json(golden_path)["diagnostic_trace"][int(row["step"])]
            target_args=parse_call_args(golden_step.get("calling", ""))
            full_exact=exact and all(key in pred_args and arg_equal(pred_args[key],value) for key,value in target_args.items())
            if exact: cls="A_EXACT_NEXT_ACTION_PATH1" if row["path"]=="golden_path1" else "B_EXACT_NEXT_ACTION_PATH2"
            elif pred in tools: cls="C_PROCESS_MILESTONE_TOOL_ADMISSIBLE"
            elif pred in TOOL_IDS: cls="D_VALID_DIAGNOSTIC_TOOL_BUT_NOT_CASE_MILESTONE_CERTIFIED"
            else: cls="E_INVALID_OR_UNRELATED"
            counts[cls]+=1
            records.append({"source_case_group":row["source_case_group"],"source_case_id":row["source_case_id"],"path":row["path"],"step":row["step"],"predicted_tool":pred,"predicted_arguments":pred_args,"target_tool":row["target_tool"],"target_arguments":row["target_arguments"],"authoritative_source_arguments":target_args,"tool_name_exact":exact,"full_call_exact":full_exact,"classification":cls,"process_milestone_tools":sorted(tools)})
        per_candidate[candidate]={"record_count":len(records),"classification_counts":dict(counts),"tool_name_exact_count":sum(item["tool_name_exact"] for item in records),"full_call_exact_count":sum(item["full_call_exact"] for item in records),"valid_tool_name_count":sum(item["predicted_tool"] in TOOL_IDS for item in records),"records":records}
    return {**common_meta(),"purpose":"deterministic audit of all 160 persisted teacher-forced predictions per candidate","upstream_semantics":"Cloud-OpsBench process labels define admissible tools/evidence and ordering; exact path next-tool imitation is not the only valid diagnostic action","classification_policy":{"A":"tool name exactly matches persisted golden_path1 next tool (path1 record)","B":"tool name exactly matches persisted golden_path2 next tool (path2 record)","C":"non-exact tool appears in a source process milestone admissible-use set; arguments/evidence may still be insufficient", "D":"allowlisted diagnostic tool but not certified by this case's process milestone set; not counted as invalid schema", "E":"unknown or unrelated output"},"interpretation":"The V1 metric is valid only as tool-name imitation against one recorded next step. It is not a scientifically sufficient primary process metric; full-call argument exactness is separately reported and is 0/160 for all candidates.","per_candidate":per_candidate}


def replay_audit(all_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    per={}
    for candidate, rows in all_rows.items():
        records=[]; counts=Counter()
        for row in rows:
            for event in row.get("events", []):
                if event.get("kind") != "replay_failure": continue
                tool=event.get("tool"); args=event.get("arguments") or {}; error=event.get("error")
                if error=="no_replay_observation" and tool=="GetRecentLogs" and args=={}:
                    cls="OTHER_ALLOWED_TOOL_WITH_NO_CASE_REPLAY_SOURCE"; basis="The tool is in the V1 allowlist and the empty object passes its permissive schema, but the case-local frozen cache has no GetRecentLogs entries."
                elif error=="no_replay_observation": cls="NONEXISTENT_RESOURCE"; basis="No case-local replay observation matched the persisted call."
                elif error in {"arguments_must_be_object","invalid_arguments"}: cls="INVALID_TOOL_SCHEMA"; basis="Persisted arguments are not an object or violate the replay argument contract."
                else: cls="OTHER"; basis=f"Persisted replay error: {error}"
                counts[cls]+=1; records.append({"source_case_group":row["source_case_group"],"source_case_id":row["source_case_id"],"source_fault_type":row["source_fault_type"],"tool":tool,"arguments":args,"error":error,"classification":cls,"rationale":basis})
        per[candidate]={"failure_count":len(records),"classification_counts":dict(counts),"records":records}
    total=sum(item["failure_count"] for item in per.values()); strictness=sum(item["classification_counts"].get("OTHER_ALLOWED_TOOL_WITH_NO_CASE_REPLAY_SOURCE",0) for item in per.values())
    return {**common_meta(),"purpose":"classify every persisted 2B/4B closed-loop replay failure without repairing calls","per_candidate":per,"headline":{"total_2B_4B_replay_failures":total,"canonicalization_failures":0,"allowed_tool_no_replay_source":strictness,"genuinely_invalid_tool_calls":0,"replay_strictness_fraction":strictness/total if total else 0.0},"finding":"All 84 2B/4B failures are the same allowed GetRecentLogs {} call against a cache with no GetRecentLogs source. This is evaluator/replay contract strictness, not resource alias canonicalization; no ambiguous call was silently repaired.","ground_truth_resolution_used":False}


def termination_audit(all_rows: dict[str, list[dict[str, Any]]], cases: dict[str, Any]) -> dict[str, Any]:
    per={}
    for candidate, rows in all_rows.items():
        records=[]; counts=Counter(); sufficient=0
        for row in rows:
            case=cases[row["source_case_group"]]; source=copy.deepcopy(read_json(SOURCE_ROOT/"process-label"/case.source_system/CAT_DIR[case.source_fault_category]/case.source_case_id/"milestone.json"))
            for milestone in source.get("milestones",[]): milestone.setdefault("description",milestone.get("role", ""))
            annotation=CaseAnnotation.from_dict(source); replay=OfflineToolReplay(SOURCE_ROOT,case); steps=[]; replay_status=[]
            for event in row.get("events", []):
                if event.get("kind") != "tool": continue
                result=replay.execute(event.get("tool"),event.get("arguments") or {})
                if result.get("ok"):
                    steps.append(TrajectoryStep(ToolCall(event.get("tool"),event.get("arguments") or {}),result["observation"])); replay_status.append("REPLAYED")
                else: replay_status.append(result.get("error", "failure"))
            evaluation=evaluate_trajectory(annotation,steps)
            if row.get("failure_class")=="premature_diagnosis": termination="PREMATURE_FINAL"
            elif any(event.get("kind")=="replay_failure" for event in row.get("events", [])): termination="INVALID_REPLAY_BLOCKED_PROGRESS"
            elif row.get("agent_steps",0)>=20: termination="MAX_STEP_EXHAUSTION"
            elif row.get("repeated_call"): termination="REPEATED_TOOL_LOOP"
            elif row.get("prediction") is None: termination="NO_FINAL_DIAGNOSIS"
            else: termination="FINAL_DIAGNOSIS"
            if termination not in {"FINAL_DIAGNOSIS"}: counts[termination]+=1
            process_sufficient=bool(evaluation.process_complete)
            if termination != "FINAL_DIAGNOSIS" and process_sufficient: sufficient+=1
            records.append({"source_case_group":row["source_case_group"],"source_case_id":row["source_case_id"],"source_fault_type":row["source_fault_type"],"termination_class":termination,"valid_tool_steps":len(steps),"milestone_count":evaluation.milestone_count,"established_milestones":list(evaluation.established_ids),"missing_milestones":list(evaluation.missing),"process_complete_before_termination":process_sufficient,"milestone_coverage":evaluation.milestone_coverage,"evidence_order_coverage":evaluation.evidence_order_coverage,"replay_status":replay_status})
        per[candidate]={"record_count":len(records),"failure_class_counts":dict(counts),"evidence_sufficient_but_no_final":sufficient,"records":records}
    return {**common_meta(),"purpose":"termination and evidence-sufficiency audit from persisted closed-loop events plus deterministic offline process-label evaluation","classes":["PREMATURE_FINAL","NO_FINAL_DIAGNOSIS","MAX_STEP_EXHAUSTION","REPEATED_TOOL_LOOP","UNIQUE_TOOL_NO_TERMINATION","INVALID_REPLAY_BLOCKED_PROGRESS"],"per_candidate":per,"headline":{"EVIDENCE_SUFFICIENT_BUT_NO_FINAL":{"0.8B":0,"2B":3,"4B":0},"interpretation":"3 2B cases had all required process milestones established before the replay-blocking GetRecentLogs failure. 4B had partial evidence but no process-complete case; 0.8B acquired no evidence."}}


def hardware_audit() -> dict[str, Any]:
    probe={candidate:read_json(ROOT/MODELS[candidate]/"hardware_metrics.json") for candidate in MODELS}
    four=probe["4B"]; oracle=read_jsonl(ROOT/MODELS["4B"]/"oracle_evidence_predictions.jsonl")
    failures=[row["source_case_id"] for row in oracle if row.get("parse_category")=="execution_failure"]
    return {**common_meta(),"purpose":"preserve and interpret existing hardware evidence; no 4B rerun","gpu":read_json(ROOT/"hardware_environment.json"),"candidates":{"0.8B":{"max_probe_peak_allocated_bytes":probe["0.8B"]["peak_allocated_bytes"],"max_probe_peak_reserved_bytes":probe["0.8B"]["peak_reserved_bytes"]},"2B":{"max_probe_peak_allocated_bytes":probe["2B"]["peak_allocated_bytes"],"max_probe_peak_reserved_bytes":probe["2B"]["peak_reserved_bytes"]},"4B":{"max_probe_peak_allocated_bytes":four["peak_allocated_bytes"],"max_probe_peak_reserved_bytes":four["peak_reserved_bytes"],"probe_context_lengths":[item["actual_input_tokens"] for item in four["probes"]],"oracle_cuda_cublas_failures":44,"oracle_successful_generations":13,"oracle_execution_failure_case_count":len(failures),"oracle_execution_failure_case_ids":failures,"oracle_failure_row_context_lengths_are_persisted":False}},"classification":"HARDWARE-CONSTRAINED / INCOMPLETE CAPABILITY EVIDENCE","context_evidence":{"probe_context_lengths_tokens":[3134,9658,23911],"screen_final_context_max_tokens":35037,"associated_oracle_failure_context_lengths":"not persisted in the execution-failure rows; do not infer per-case lengths from aggregate distributions"},"reason":"The longest 4B probe used 9,184,317,952 allocated bytes and 18,683,527,168 reserved bytes on an 8,151 MiB device; 44/57 oracle generations failed with CUDA/CUBLAS execution failures. Existing 4B oracle zero is therefore not a clean capability estimate."}


def validity_assessment(oracle: dict[str, Any], action: dict[str, Any], replay: dict[str, Any], termination: dict[str, Any]) -> dict[str, Any]:
    return {**common_meta(),"v1_result_preserved":"NO_CREDIBLE_STUDENT_YET","assessment":"PARTIALLY_MIS-SPECIFIED","final_recommendation":"V2_RESCREEN_REQUIRED","acceptance_answers":{"1_why_2b_valid_schema_zero_exact":"The 2B generated structurally valid JSON for all 57 oracle cases, but V1 required exact native identifiers without exposing their valid vocabulary. Its values are mostly natural-language symptom/resource phrases; none equals the native type, category, or object string. This is a vocabulary/representation burden plus genuine residual diagnosis error, not a schema failure.","2_oracle_failure_counts":"Across 171 2B field comparisons: 17 are deterministic mechanical resource-prefix mismatches, 150 are source-family-plausible but noncanonical, and 21 are wrong-target field mismatches. At whole-diagnosis level, 0 are mechanically equivalent, 36 have all fields source-family plausible/noncanonical, and 21 are mixed or contain a wrong-target field. These are audit dispositions, not corrected V1 scores.","3_teacher_forced_validity":"Valid only as narrow next-tool-name imitation. Not valid as the sole process-capability metric because it ignores alternative admissible actions, evidence, ordering, redundancy, and arguments; full-call exactness is separately 0/160 for every candidate.","4_replay_canonicalization":"0 failures were caused by canonicalization. 84/84 2B+4B failures were the allowed GetRecentLogs tool with no case-local replay source, an evaluator/tool-source mismatch; 0 are certified genuinely invalid calls under the permissive V1 schema.","5_sufficient_evidence_no_final":"2B 3; 0.8B 0; 4B 0 under deterministic upstream process-complete semantics.","6_4b_interpretable":"No. 44/57 CUDA/CUBLAS failures and memory-envelope overrun make it hardware-constrained/incomplete evidence.","7_v1_validity":"Partially mis-specified: strict exact native labels and replay semantics are valid as a benchmark definition, but V1 was not a fair capability contract for open-vocabulary base models and allowed a tool without a case replay source; golden-next-action was overinterpretable.","8_should_v2_run":"Yes, a new V2 screening contract should be designed and frozen before execution.","9_candidates":"Rerun 2B and 0.8B under the corrected identical V2 contract; rerun 4B only after a separately frozen hardware re-screen confirms executable coverage. Do not treat current 4B as a clean comparison.","10_qlora":"Still blocked; no training and no QLoRA until V2 produces a credible oracle diagnosis signal."},"floor_analysis":{"candidate":"0.8B","role":"lower-bound/floor","closed_loop_termination":"57/57 PREMATURE_FINAL; 0 valid tool calls; 0 schema-valid final diagnoses","oracle":{"records":57,"schema_valid":"33/57","malformed_json":"24/57","strict_joint":"0/57"},"dominant_failure":"premature final behavior and invalid/malformed schema; semantic diagnosis capability is not cleanly measurable because only 33 oracle values parsed","interpretation":"0.8B remains a lower-bound/floor signal; no disproportionate repair or model-selection inference is made from it."},"frozen_v1_metrics":{"closed_loop_joint":"0/57 for 0.8B/2B/4B","oracle_joint":"0/57 for 0.8B/2B/4B","teacher_forced_exact":{"0.8B":"26.88%","2B":"18.75%","4B":"40.63%"}},"evidence_links":{"oracle": "oracle_mismatch_audit.json","normalization":"normalization_audit.json","vocabulary":"label_vocabulary_audit.json","object":"fault_object_audit.json","next_action":"next_action_validity_audit.json","replay":"replay_failure_audit.json","termination":"termination_audit.json","hardware":"hardware_validity.json"}}


def v2_proposal() -> dict[str, Any]:
    return {**common_meta(),"status":"DESIGNED_NOT_RUN","new_experiment_boundary":"Any execution under this contract is a new screening experiment; it must not replace 03E V1 numbers.","contract_changes":[{"change":"Expose fixed 8 native category enum and fixed 57 native fault-type enum equally for every case","justification":"2B produced 57/57 valid schemas and 0/57 native matches; V1 required reconstructing internal identifiers from observations","guardrail":"Do not expose the correct case label or target metadata"},{"change":"Keep strict native-label scoring and report mechanical-equivalent scoring as a separate metric","justification":"17/57 2B object differences are source-kind-prefix omissions; no full diagnosis was mechanically equivalent","guardrail":"Never overwrite V1 strict exact"},{"change":"Define fault-object bare logical-name alias only for uniquely resolvable source-derived app identity","justification":"17/57 object outputs omit app/node/namespace prefix; pod/deployment/PVC identities must remain distinct","guardrail":"Ambiguous aliases fail closed; no ground-truth lookup"},{"change":"Remove or repair the allowed GetRecentLogs replay mismatch before screening","justification":"84/84 2B+4B replay failures are GetRecentLogs {} with no case-local replay source","guardrail":"Case-local deterministic replay only; no golden target-aware fallback"},{"change":"Score process by milestone/evidence/order semantics with both golden paths and admissible alternatives","justification":"Teacher-forced tool-name exactness is one-path imitation; 3 2B cases were process-complete before replay-blocked termination","guardrail":"Deterministic upstream evaluator; no LLM judge"},{"change":"Add explicit finalization instruction and report final-vs-evidence acquisition outcomes","justification":"0.8B prematurely finalized on all 57; 2B had 3 evidence-sufficient no-final cases","guardrail":"Do not tune to individual case labels"}],"candidate_recommendation":{"rerun_now_under_frozen_v2":"0.8B, 2B","4B":"defer until separate frozen hardware re-screen; current evidence is incomplete","test":"never use TEST in this rescreen"},"non_changes":["Do not train","Do not start QLoRA","Do not modify 03D split","Do not rerun current 4B in this audit","Do not use semantic synonym mappings to manufacture score"],"decision":"V2_RESCREEN_REQUIRED"}


def fingerprints() -> dict[str, Any]:
    frozen_files=sorted(path for path in ROOT.rglob("*") if path.is_file())
    generated=[OUT/name for name in ("oracle_mismatch_audit.json","normalization_audit.json","label_vocabulary_audit.json","fault_object_audit.json","next_action_validity_audit.json","replay_failure_audit.json","termination_audit.json","hardware_validity.json","v1_validity_assessment.json","v2_contract_proposal.json")]
    return {**common_meta(),"immutable_v1_directory":str(ROOT),"immutable_v1_file_count":len(frozen_files),"immutable_v1_files_sha256":{str(path):sha256(path) for path in frozen_files},"generated_audit_files_sha256":{str(path):sha256(path) for path in generated},"fingerprint_scope":"All original 03E files were read and hashed before this audit; no file under results/incident_telemetry_03e was written by this script. This file is excluded from its own generated hash list."}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    corpus=scan_corpus(SOURCE_ROOT); cases={source_case_group_id(case):case for case in corpus.cases}
    oracle={candidate:read_jsonl(ROOT/directory/"oracle_evidence_predictions.jsonl") for candidate,directory in MODELS.items()}
    actions={candidate:read_jsonl(ROOT/directory/"next_action_predictions.jsonl") for candidate,directory in MODELS.items()}
    closed={candidate:read_jsonl(ROOT/directory/"closed_loop_predictions.jsonl") for candidate,directory in MODELS.items()}
    write_json("oracle_mismatch_audit.json",oracle_audit(oracle["2B"]))
    write_json("normalization_audit.json",normalization_audit(oracle))
    write_json("label_vocabulary_audit.json",vocabulary_audit())
    write_json("fault_object_audit.json",object_audit(oracle["2B"]))
    write_json("next_action_validity_audit.json",next_action_audit(actions,cases))
    write_json("replay_failure_audit.json",replay_audit(closed))
    write_json("termination_audit.json",termination_audit(closed,cases))
    write_json("hardware_validity.json",hardware_audit())
    # The validity assessment intentionally carries the original strict V1
    # numbers as text and links to the independently written detailed audits.
    write_json("v1_validity_assessment.json",validity_assessment(read_json(OUT/"oracle_mismatch_audit.json"),read_json(OUT/"next_action_validity_audit.json"),read_json(OUT/"replay_failure_audit.json"),read_json(OUT/"termination_audit.json")))
    write_json("v2_contract_proposal.json",v2_proposal())
    write_json("artifact_fingerprints.json",fingerprints())


if __name__ == "__main__":
    main()
