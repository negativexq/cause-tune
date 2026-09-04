# Experiment 03D — Training Contract, Evidence Packaging, and Protected Split Freeze

## Status

**COMPLETE — TRAINING CONTRACT AND PROTECTED SPLITS FROZEN.** This milestone
uses the pinned Cloud-OpsBench checkout audited in 03C and produces compact
manifests/references only. No model was selected, loaded, trained, or
evaluated.

The source revision is
`03c415e5709297432282fbbfd499f1bca0f8c347`; the source-manifest fingerprint is
`5c1644083fd477d43d3eb4a6822baa419e84589ad856ebf550383f9e54391867`; and the
03D freeze component fingerprint is
`2a1be7f63f42e7900968cbc0c334b5fbf7274a31dd60435101a084aac382ccaf`.
The external checkout is not vendored. Recreate the manifests with:

```bash
uv run python scripts/freeze_incident_03d.py \
  --root /path/to/Cloud-OpsBench \
  --output-dir results/incident_telemetry_03d
```

## Why the primary formulation is hybrid

The 03C context audit found that every raw combined snapshot exceeds the
8,192-token audit budget. Flattening Kubernetes state, logs, metrics, tool
cache, and code into a single SFT prompt is therefore rejected as the primary
formulation. 03D freezes a bounded diagnostic interaction: a sanitized initial
incident request, allowlisted tool actions, deterministic offline observations,
and a structured final diagnosis target.

The primary future training comparison is:

1. tool/action supervision plus deterministic final diagnosis; and
2. a static packaged evidence view as a future ablation.

Neither view is trained or evaluated in 03D.

## Authoritative target contract

Cloud-OpsBench native metadata remains authoritative. The target is derived
mechanically from `metadata.result`:

- `native_fault_type` ← `root_cause`
- `native_fault_category` ← `fault_taxonomy`
- `fault_object` ← `fault_object`

No confidence score, generated diagnosis prose, or free-form chain-of-thought
is a correctness target. The source-native fault type is not collapsed into the
older 03A ontology.

## Model-visible boundary

The initial model-visible context is the leakage-checked, sanitized generic
incident query. It excludes source labels, target metadata, benchmark paths,
case identity, process labels, golden trajectories, source fingerprints, and
the final diagnosis target. Tool actions are restricted to the offline replay
contract; observations are bounded packages with explicit original/package
sizes and modality states.

The supported tool IDs are `GetResources`, `DescribeResource`, `GetAppYAML`,
`GetServiceDependencies`, `CheckServiceConnectivity`, `GetAlerts`,
`GetRecentLogs`, `GetErrorLogs`, `ListCodeFiles`, `GetSourceCode`,
`GetClusterConfiguration`, and `CheckNodeServiceStatus`. Unknown tools fail
closed.

## Evidence packaging

Packaging version `cloud-opsbench-evidence-packaging-v1` uses a 32,768-character
per-observation bound. Complete observations are preserved when they fit. An
oversized observation receives deterministic line-based head/tail plus uniform
middle sampling, with an omission flag; no truncation is silent and no target,
process label, or golden answer is used for selection. The full raw snapshot is
never the primary input.

The real 1,508 trajectory-derived records contain 7,391 tool observations.
The original observation-character distribution is min 0, median 1,666,
p90 5,824, p95 11,108, p99 30,319, max 104,745. The packaged maximum is
32,768. Sixty-nine observations require the explicit omission policy. 1,007
observations use an immutable golden-trace fallback because their serialized
trace output does not exactly match a case-local cache value; these are flagged
in `packaging_audit.json`, not silently treated as cache hits.

Metrics and code retain source semantics: missing `metrics.csv` is
`MODALITY_NOT_AVAILABLE`, not an invented empty metric stream; code is a
conditional tool modality and is not placed in every initial context.

## Trajectory normalization and supervision

Each case produces two distinct derived records, one for each golden path. The
record contains a source-case reference, split, tool/action message specs,
observation references and packaging metadata, plus a deterministic final
diagnosis target. Raw observation text is reconstructed from the pinned source
at execution time rather than duplicated in the manifest.

Golden traces expose `diagnostic_trace` steps with `tool_name`, `calling`, and
`output`; no explicit final-answer field was present in the audited source.
Free-form reasoning/scratchpad is therefore classified **DROP** for primary
supervision. Future SFT should supervise assistant tool/action calls and the
structured deterministic final diagnosis, but not user messages, tool
observations, provenance, process labels, or unrestricted reasoning.

## Duplicate and contamination policy

03D fingerprints exact serialized trajectories, normalized trajectories,
tool-call sequences, and tool-call plus normalized argument patterns. The real
source has 1,508 trajectories, 98 exact duplicate signature groups, 98
normalized duplicate groups, 116 tool-sequence groups, and 122 argument-pattern
groups. Common tool policies are reported but are not automatically
contamination groups. No cross-source-case normalized trajectory group was
observed; source-case identity remains the stronger grouping boundary.

All raw artifacts, process labels, both golden paths, future derived examples,
and future augmentation from a source case inherit one immutable source-case
group. No identified contamination group may cross protected partitions.

## Protected split

The deterministic case-grouped split uses a documented 70/15/15 approximation,
with fixed ordering and atomic grouping. The frozen counts are:

| Split | Cases | Share | Native fault types |
| --- | ---: | ---: | ---: |
| TRAIN | 510 | 67.639% | 57 |
| VALIDATION | 123 | 16.313% | 57 |
| TEST | 121 | 16.048% | 57 |

All 754 source cases occur exactly once, and both paths remain with their case.
All 57 native fault types are represented in every partition; no rare label
has fewer than three cases in the full source corpus. The slight deviation
from target percentages is accepted in favor of deterministic label coverage
and atomic grouping.

TEST is sealed evidence. It cannot be used for model selection, prompt or
packaging tuning, threshold/checkpoint selection, LoRA decisions, augmentation,
or ablations. 03E may use TRAIN/VALIDATION for untouched base-model screening;
TEST is first consumed only by the explicitly defined final evaluation stage.

Analytical slices are labels over this same manifest: EASY, MEDIUM, HARD,
CODE_DEFECT, METRICS_AVAILABLE, METRICS_UNAVAILABLE, ONLINE_BOUTIQUE,
TRAIN_TICKET, plus exploratory `CROSS_SYSTEM` and `UNSEEN_FAULT_TYPE` labels
when their definitions are scientifically meaningful. Train-Ticket is not
called topology OOD: 03C found only one native fault type shared across the
two systems, so cross-system analysis is strongly label-confounded.

## Static baseline and provenance

The static baseline is a bounded packaged-view manifest containing only
sanitized incident context, modality availability, source references, and
packaging policy. It is not a raw-snapshot copy and contains no target. The
trajectory and static descendants are linked to their source case through
`source_case_derivation_graph.json`; descendants inherit source revision, case
group, split, and target fingerprint. Future renderer-prototype augmentation
must inherit the source-case split and may not cross this boundary.

## Artifacts

`results/incident_telemetry_03d/` contains compact, reproducible contracts and
manifests: split distribution, target/model-visible/tool/packaging contracts,
normalization and supervision policies, duplicate/contamination audits, static
and trajectory example manifests, packaging audit, source derivation graph,
freeze manifest, audit summary, and artifact fingerprints. It does not contain
the external raw dataset.

## Limitations and 03E boundary

- The corpus remains a benchmark of injected incidents, not unrestricted
  production traffic.
- The 32,768-character observation policy is a deterministic contract, not a
  demonstrated compact-model context solution.
- Golden trajectories are expert tool traces without an explicit final-answer
  object; their usefulness for trajectory SFT remains an empirical 03E/03F
  question.
- Code is available only for Online Boutique, so CODE_DEFECT analysis is
  modality/system confounded.
- Native labels are broad and highly imbalanced in semantic structure even
  though the protected split covers each type.

03E may screen untouched base models using validation only. It must not revise
the frozen split, leakage boundary, packaging policy, target contract, or
supervision policy after inspecting TEST.
