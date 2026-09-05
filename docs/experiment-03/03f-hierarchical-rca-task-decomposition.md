# Experiment 03F — Hierarchical RCA task decomposition and contract freeze

## Status

03F is a CPU-only decomposition and contract-freeze experiment. It ran no
language model, used no GPU or provider, performed no training, created no
adapter, and made no TEST prompt or prediction. The full 03D split remains
unchanged: TRAIN 510, VALIDATION 123, TEST 121.

The 03E series established that untouched Qwen3.5-2B was too weak for the full
agentic chain: incident → autonomous tools → evidence → stopping → category →
native root cause → fault object. 03E.2 ended with `BASE_TOO_WEAK` and
`DO_NOT_START_QLORA`. Larger hardware and capacity-gradient experiments are
intentionally out of scope. 03F therefore isolates bounded diagnosis before
specializing tool policy.

The future architecture is:

```text
Diagnostic Tool Policy -> Bounded Evidence -> Hierarchical RCA Specialist
                         -> Fault Object Resolver -> Structured Diagnosis
```

The agentic problem is not abandoned; the later system can recompose these
components after a compact-model diagnosis capability is established.

## Native hierarchy

The registry is derived from every pinned-corpus `metadata.json`, grouping
native `root_cause` values by native `fault_taxonomy`. It is not a manually
reconstructed taxonomy. All 57 native root causes map to exactly one category,
with no conflicts:

| Category | Native root-cause candidates |
|---|---:|
| Admission_Fault | 6 |
| Application_Code_Defect | 8 |
| Infrastructure_Fault | 4 |
| Performance_Fault | 4 |
| Runtime_Fault | 12 |
| Scheduling_Fault | 12 |
| Service_Routing_Fault | 6 |
| Startup_Fault | 5 |

The hierarchy fingerprint is recorded in `freeze_manifest.json` and changing
the ordered registry changes the decomposition freeze fingerprint.

## Candidate tasks

### TASK A — category classification

Input is one bounded, target-blind evidence package and the sanitized incident
request. Output is exactly `{"fault_category": "<native category>"}`. The
complete eight-category enum is exposed. Root cause, object, explanation,
confidence, and reasoning are forbidden. Primary future metrics are category
exact, schema-valid, enum-valid, and a confusion matrix.

### TASK B — oracle-category root cause

Input is the same evidence plus an authoritative category context. Output is
exactly `{"root_cause": "<native root cause>"}`, restricted to the candidate
set for that supplied category. This is deliberately an oracle ablation and is
always named `ORACLE_CATEGORY_ROOT_CAUSE`; it is not end-to-end RCA. Metrics
are root-cause exact, schema-valid, enum-valid, and per-category accuracy.

### TASK C — hierarchical self-predicted RCA

Stage 1 predicts a category from evidence. Stage 2 receives the same evidence
and only the Stage-1 predicted category, then chooses from that predicted
category's native candidate set. It never receives the authoritative category
in Stage 2. Metrics separate category error from within-category
discrimination error: category exact, root-cause exact, hierarchical joint
exact, Stage-2 conditional accuracy given correct category, and Stage-2 failure
caused by the wrong Stage-1 category.

### TASK D — fault-object resolution

This is an independent secondary head, not the first specialization target.
Output is exactly `{"fault_object": "<resource identity>"}`. Strict object
exact and normalized object exact are reported. The only normalization is the
previously justified, deterministic bare application name → `app/name` mapping
when that identity is uniquely visible. Different Kubernetes resources are
not collapsed, and no semantic object judge is used.

The primary specialization choice is among A/B/C. Preference is C, then B,
then A, but C qualifies only if both stages show meaningful signal.

## Evidence contract

03F uses only source-cache-resolved observations from the existing 03D/03D.1
representation. The deterministic policy is path1 preferred; path2 is used
only if path1 has zero executable observations. No model-accuracy signal
selects a path, and paths are not automatically doubled into two training
examples. In this audit all 633 TRAIN+VALIDATION cases selected path1.

Each model-facing package contains only a sanitized incident request and
bounded `{tool, observation}` entries. It contains no target, process label,
golden final answer, source path, or provenance field. TEST is excluded before
package construction; the package artifact contains 510 TRAIN and 123
VALIDATION rows only.

A fixed 30,000-character total package budget is enforced by a deterministic
equal per-observation line-package cap found by binary search. This cap is
independent of labels, process semantics, model behavior, and future scores.
The auditor may compare packages with process labels, but the packager does
not use process labels to select evidence.

## Evidence sufficiency

Across the 633 non-TEST packages there are 2,154 upstream process-label
evidence patterns. The package retains 1,735 (80.55%), leaving 419 missing:

- full pattern coverage: 244 cases;
- partial coverage: 79 cases;
- zero coverage: 310 cases.

These numbers are an audit of whether the bounded source evidence can support
the upstream process semantics. They are not model scores. The large zero
coverage group is an unresolved risk for future diagnosis screening and
training; it must not be hidden by target-aware evidence selection.

## Context-size audit

The previous 03D.1 full-history distribution had median 10,524 characters and
maximum 150,042. The 03F bounded package and fixed task contracts produce:

| Task | Median chars | P95 chars | Max chars | Median tokens | Max tokens |
|---|---:|---:|---:|---:|---:|
| A | 7,110 | 29,347 | 30,233 | 2,346 | 12,205 |
| B | 7,291 | 29,604 | 30,490 | 2,374 | 12,256 |
| C | 8,887 | 31,124 | 32,010 | 2,713 | 12,572 |
| D | 6,929 | 29,166 | 30,052 | 2,307 | 12,166 |

Task A median is 67.56% of the prior full-history median and its maximum is
20.15% of the prior maximum. Task C includes the complete hierarchy contract,
so it is the largest decomposed input but remains below 32K serialized
characters. Exact tokens use only the already-local Qwen tokenizer; no model
was loaded.

## TRAIN support and future record counts

TRAIN contains 510 cases. Category counts are:

| Category | TRAIN cases |
|---|---:|
| Admission_Fault | 40 |
| Application_Code_Defect | 69 |
| Infrastructure_Fault | 27 |
| Performance_Fault | 51 |
| Runtime_Fault | 93 |
| Scheduling_Fault | 109 |
| Service_Routing_Fault | 65 |
| Startup_Fault | 56 |

Per-root-cause TRAIN support ranges from 1 to 40, with median 8. Ten labels
have low support (<3): `code_memory_leak`,
`namespace_storage_quota_exceeded`, `node_network_delay`,
`node_network_packet_loss`, `pv_binding_occupied`,
`pvc_access_mode_mismatch`, `pvc_capacity_mismatch`,
`pvc_selector_mismatch`, `pvc_storage_class_mismatch`, and
`volume_mount_permission_denied`. No oversampling or synthesis is performed.

The deterministic future record estimates are one record per TRAIN case for A
(510), B (510), and D (510 valid object targets). C has zero records now because
Stage-1 predictions must not be fabricated; its future construction is one
two-stage record per TRAIN case. Alternate golden views are a future ablation,
not automatic corpus doubling.

## Trivial baselines and frozen 03G policy

The fixed-seed baselines are frozen before 03G: seed `20260303` for uniform
random draws, plus TRAIN-derived majority baselines. Uniform category chance
is 12.5%. Uniform B chance is category-size weighted (14.04% on the screen).
Uniform hierarchical joint expected accuracy is 1.75% on the screen. The
screen-draw realizations and majority results are persisted in
`trivial_baselines.json`; they are reference baselines, not model outputs.

03G may select a task only if untouched 2B is materially above its deterministic
trivial baseline, clearly below saturation, schema/enum reliable, context and
local hardware practical, and every target class has enough TRAIN support,
with no environment-contract confound. No post-hoc numerical threshold is
defined. If multiple tasks qualify, the frozen order is C, B, A. If C lacks a
meaningful signal in either stage, it cannot be selected. If no task qualifies,
the policy is `TASK_REDESIGN_REQUIRED`; 03H/QLoRA remains blocked.

## Roadmap pivot

03A, 03B, Renderer R0 (supporting only), 03C, 03D, 03D.1, and 03E/03E.1/03E.2
are complete. 03E concluded that 2B is `BASE_TOO_WEAK` for full joint agentic
RCA. 03F is active and freezes this decomposition. 03G is future decomposed
untouched-2B screening; 03H is future/blocked QLoRA; 03I is future tool-policy
specialization; 03J is future recomposed closed-loop evaluation. RCAEval is
reserved for later external generalization.

## Limitations

This experiment proves contract consistency and measures evidence support; it
does not establish that any decomposed task is learnable. The 310 zero-coverage
cases, sparse support for ten root causes, open-resource object identity, and
the distinction between authoritative process evidence and one expert path
remain risks. Future 03G must report raw task-specific measurements and apply
the frozen qualitative policy without changing the evidence package or task
definitions after inference begins.

No model was loaded. No model was trained. No QLoRA was performed. No TEST
inference occurred.
