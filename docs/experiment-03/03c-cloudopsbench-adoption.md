# Experiment 03C — Cloud-OpsBench Adoption and Corpus Audit

## Status

**COMPLETE — PINNED REAL CORPUS AUDITED.** This milestone adopts Cloud-OpsBench
as the primary empirical corpus for the next incident-specialization phase. It
does not train or evaluate a model, freeze a final split, or make a performance
claim. The pinned checkout was audited read-only at the exact revision below.

The audited upstream revision is
`03c415e5709297432282fbbfd499f1bca0f8c347` (`main`). The upstream repository
is [Cloud-OpsBench](https://github.com/LLM4Ops/Cloud-OpsBench), released under
the MIT license. The license file hash at that revision is
`d19af713964259f9ab5d662bfc706d2c072ee9211495d3d6d4c3c7588f59abbc`.

## Why the primary corpus changed

Experiment 02 measured a strong gain on a controlled synthetic incident task.
That result does not establish robustness on fault-injected cloud systems.
Cloud-OpsBench provides a new evidence boundary: immutable case snapshots from
two Kubernetes-based systems, with native fault labels, alerts, logs, cluster
state, optional metrics/code, process labels, and two auxiliary diagnostic
trajectories. CauseTune preserves those native labels rather than collapsing
them into the frozen 03A ontology.

The upstream README documents 754 cases, 57 native fault types, 550 Online
Boutique cases, and 204 Train-Ticket cases. The audit observed exactly those
counts. Observed categories were Admission 58, Scheduling 164, Startup 86,
Runtime 141, Service Routing 91, Performance 76, Infrastructure 40, and
Application Code Defect 98. Upstream difficulty metadata was easy 391,
medium 189, and hard 174.

## Source registry and case boundary

`configs/cloud_opsbench_03c.json` pins the official URL, revision, MIT license,
license hash, expected layout, documented counts, and adapter version. The
dataset is not vendored. Use:

```bash
CLOUD_OPSBENCH_ROOT=/path/to/Cloud-OpsBench \
uv run python scripts/audit_cloud_opsbench_03c.py \
  --output-dir results/incident_telemetry_03c_cloudops
```

The scanner is read-only. It represents each case with source system, case ID,
native category/type, upstream difficulty, source metadata availability,
artifact references, and a revision-bound source fingerprint. Missing
`metrics.csv` and `code/` are reported as optional modalities; missing core
case artifacts fail closed.

All 754 cases had `metadata.json`, `tool_cache.json`, `k8s_states.json`,
`logs.json`, `alert.json`, process labels, and both golden trajectory files.
`metrics.csv` was present for 170/754 cases (22.55%); its absence is valid for
early lifecycle cases. `code/` was present for 550/754 cases (72.94%), all
Online Boutique cases; Train-Ticket has no code field in this revision.

The model-visible boundary is explicit. Raw alerts, logs, Kubernetes state,
metrics, and permitted code are possible observations. Metadata target fields,
process labels, golden trajectories, source paths, source fingerprints, and
explicit final answers are supervision/provenance or forbidden-input fields.
No target is manufactured by an LLM or by the audit.

## Native taxonomy and target audit

All 57 upstream fault types are preserved with their original category. The
source metadata represents these native types as stable lower-snake-case
`result.root_cause` values (while the upstream taxonomy table also documents
human-readable fault names). The
mapping report compares them to the frozen 03A IDs using conservative labels:
`EXACT_MATCH`, `COMPATIBLE_ALIAS`, `SOURCE_MORE_SPECIFIC`,
`CAUSETUNE_MORE_SPECIFIC`, `PARTIAL_SEMANTIC_OVERLAP`, `UNMAPPED`, and
`CONFLICT`. A normalized CauseTune ID is optional audit metadata; the mapping
does not mutate 03A history or force unsupported equivalences.

The target audit reports which diagnosis signals are actually present in source
metadata, including native fault type/category and any mechanically available
component, namespace, service, diagnosis, query, or difficulty fields. It does
not invent structured targets that the source does not provide.

In the real corpus, `metadata.result.root_cause`, `fault_taxonomy`, and
`fault_object` were present in all 754 cases. The metadata query had six generic
incident-symptom variants; it was not a direct root-cause label, but remains a
field requiring explicit 03D input-boundary review. The audit found no literal
native fault-type label in the golden trajectory surfaces. Tool-cache keys are
observable tool names/results, but cached target answers and collection
metadata must be transformed out before model input. Alerts, logs, Kubernetes
state, and available metrics are legitimate observable evidence, not automatic
leakage.

## Golden trajectories and context size

The scanner audits path availability, approximate step counts, tool names,
serialization sizes, exact/normalized duplicates, and literal native-label
occurrences. It records deterministic byte/character distributions when no
tokenizer is loaded. It also compares process-label and trajectory references;
semantic correctness is deliberately deferred until the source formats are
reviewed in 03D.

All 754 cases had both trajectories (1,508/1,508 paths), with no missing or
one-sided cases. A trajectory contains a `diagnostic_trace` tool sequence; no
explicit final-answer field was present. Across paths, trace steps were min 6,
median 18, p90 29, p95 35, p99 346, max 684. There were 97 exact duplicate
paths beyond first occurrences and 98 normalized duplicates. The most frequent
tools were `GetResources` (1,508), `DescribeResource` (938), `GetAppYAML`
(734), and `GetErrorLogs` (588). These are structural observations, not evidence
that trajectory SFT is appropriate yet.

Per-case context audit reports min, median, p90, p95, p99, and max sizes for
Kubernetes state, logs, metrics, alerts, tool cache, code, combined snapshots,
and golden trajectories. No lossy truncation is applied. If a compact model
cannot consume a flattened snapshot, 03D must choose evidence packaging,
retrieval, staged diagnosis, or bounded telemetry windows explicitly.

The combined raw snapshot was 1,386,321 bytes at the median, 16,116,515 bytes
at p95, and 25,554,683 bytes at maximum. Using the audit’s tokenizer-independent
bytes/4 heuristic, all 754 cases exceeded an 8,192-token compact-context budget.
One-shot flattening is therefore not realistic without explicit packaging or
retrieval; no data was truncated.

The trajectories are not converted to SFT examples here. The audit provides
evidence for a later choice between static snapshot supervision, trajectory/tool
use supervision, or a hybrid; that choice belongs after corpus normalization
and protected split design.

## Split feasibility and protected evidence

03C evaluates candidate strategies only; it does not freeze the final split.
The report covers case holdout, native fault-type holdout, category holdout,
cross-system holdout, and code-availability holdout. It distinguishes unseen
label classification from topology/system generalization and records their
confounds.

Candidate statistics use deterministic case-group hashing only for feasibility,
not as the final split. Case holdout produced train 523, validation 69, and
protected test 162 with broad native-label overlap. A deterministic 12-type
fault holdout produced train 409, validation 183, and test 162 with zero
train/test label overlap, so it is an unseen-label task. Startup category
holdout produced train 464, validation 204, and test 86, again with no
train/test label overlap. Cross-system holdout produced train 414, validation
136, and Train-Ticket test 204: only `pod_network_delay` overlapped; nine
native labels were Train-Ticket-only. It is therefore not a clean topology-OOD
test. A code-availability proposal produced train 178, validation 26, and
code-present test 550, but is strongly confounded by system and modality.

The strongest primary proposal is case-grouped holdout with known-label
coverage, supplemented by explicitly named fault-type and cross-system slices.
03D must define topology separately; Cloud-OpsBench’s system split cannot be
called topology generalization when it also changes native label coverage.

The protected policy is case-atomic: all derived variants, process labels, and
golden trajectories stay with the source case. Held-out cases are selected
before augmentation, and RCAEval remains outside training/development as a
candidate independent cross-dataset evaluation for 03H.

## Relationship to existing synthetic infrastructure

The 03B deterministic scenario engine remains useful for controlled source-gap
coverage, counterfactuals, abstention cases, and later ablations. It is no
longer the primary empirical source of incident truth. The previous LLM
telemetry pilot is retained as Renderer Prototype R0: provider-neutral,
offline-safe, and optional future augmentation infrastructure. It is not model
evidence and is not called by this audit.

## Artifacts

With a local source checkout, the CLI writes concise artifacts under
`results/incident_telemetry_03c_cloudops/`: source manifest, census, modality
availability, native taxonomy, mapping, target availability, context audit,
golden trajectory audit, leakage policy, split feasibility, concrete leakage
audit, artifact fingerprints, and final audit summary. The external dataset
itself is never copied into CauseTune.

## Limitations and 03D boundary

- The source is a benchmark corpus, not unrestricted production traffic.
- This milestone has no model capability or quality evidence.
- The upstream native taxonomy is broader and structurally different from the
  frozen 03A ontology; mappings remain reviewable rather than silently merged.
- Trajectory semantic agreement, near-duplicate detection, normalization, and
  final case-level splits remain unresolved.
- A cross-system split can confound system, topology, and native fault-label
  differences.

03D will design and freeze case-level splits, normalize model-visible inputs,
complete contamination/leakage audits, and establish protected evidence
partitions. It must not consume RCAEval into training or select a model before
that boundary is frozen.
