# Experiment 03A — Causal Telemetry Incident Specialization

## Status

**CONTRACT-FROZEN FOUNDATION ONLY.** Experiment 03A contains no generated
dataset, provider call, model evaluation, training run, or evidence that a new
specialist model works. It freezes the interfaces and integrity rules for the
future Experiment 03 pipeline.

## Scientific question

Experiment 03 will ask whether a small language model can learn robust
production-incident diagnosis from a causally controlled synthetic telemetry
corpus, and whether that specialization generalizes to unseen scenarios,
topologies, telemetry surfaces, and generator families.

Experiment 02 demonstrated specialization on a controlled incident benchmark,
but its benchmark and renderer structure are not sufficient evidence for this
stronger claim. Experiment 03 therefore creates a new evidence boundary. The
Experiment 02 benchmark is historical evidence only: it is not reused for
Experiment 03 model selection, tuning, checkpoint selection, ablations, or
final claims.

## Evidence boundary and authority

The causal data flow is:

```text
Incident ontology
    -> deterministic canonical scenario
    -> authoritative causal state and labels
    -> telemetry renderer
    -> logs / metrics / events / changes / dependency health
    -> model input
```

The `CanonicalScenario` is the sole source of ground truth. It owns the root
cause, fault domain, affected component, causal evidence definitions,
distractors, answerability, expected runbook, and split identity. Rendered text
is an observation surface, never a label source.

The provider-neutral renderer accepts a canonical scenario and returns
`RenderedTelemetry`. It may paraphrase logs, vary terminology and ecosystem
wording, and inject only distractors already declared by the scenario. It may
not choose or modify the root cause, affected component, failure family,
causal evidence identity, remediation/runbook, answerability, abstention
behavior, or split membership. The rendered schema deliberately contains no
authoritative label fields, and the boundary validator rejects unsupported
evidence IDs, changed evidence kinds, and unauthorized required-evidence
removal.

## Versioned ontology

The V1 ontology is in `src/causetune/incident_telemetry/ontology.py` and uses
stable IDs under `incident-telemetry-ontology-v1`. It contains these bounded
domains and failure modes:

| Fault domain | Failure-mode IDs |
| --- | --- |
| RESOURCE | `container_memory_exhaustion`, `memory_leak`, `cpu_throttling`, `disk_io_saturation`, `disk_pressure` |
| DATABASE | `db_connection_pool_exhaustion`, `db_query_regression`, `db_capacity_saturation` |
| DEPENDENCY | `downstream_dependency_timeout`, `downstream_dependency_unavailable`, `rate_limit_misconfiguration` |
| NETWORK | `dns_resolution_failure`, `connection_refused`, `tls_certificate_expiration` |
| APPLICATION | `thread_pool_exhaustion`, `worker_pool_exhaustion`, `configuration_regression` |
| MESSAGING | `kafka_consumer_lag`, `message_broker_unavailable` |
| CACHE | `cache_stampede`, `cache_unavailable` |
| KUBERNETES_RUNTIME | `readiness_probe_failure`, `crash_loop`, `scheduling_failure`, `node_resource_pressure` |
| SCALING | `insufficient_replicas`, `autoscaling_failure`, `traffic_capacity_exhaustion` |

This is a V1 bound, not a claim that every real production failure fits the
taxonomy. Multi-fault and cascading-failure generation is intentionally
deferred.

## Canonical scenario schema

`CanonicalScenario` is a typed immutable model with:

- scenario ID and schema version;
- root-cause ID, fault domain, root-cause family, and affected component;
- typed service topology and runtime/ecosystem;
- typed causal state variables, causal evidence definitions, and distractors;
- recent changes and dependency state;
- case type, difficulty, answerability, and expected runbook ID;
- archetype, generator/template families, split-group and topology-group IDs;
- counterfactual pair identity when applicable; and
- provenance with seed, renderer metadata slots, and artifact fingerprints.

`STANDARD`, `HARD`, and `COUNTERFACTUAL` are answerable case types.
`INCOMPLETE` retains the canonical cause while requiring an
`INSUFFICIENT_EVIDENCE` abstention contract. `HEALTHY_CONTROL` has no true
incident and no root cause.

## Telemetry schema

`RenderedTelemetry` contains only a scenario ID, schema version, observations,
renderer metadata, and provenance hashes. Each `TelemetryObservation` has a
stable evidence ID, observation kind, timestamp, optional component, text, and
optional JSON value. Supported observation kinds are application logs, runtime
events, metrics, changes, dependency health, alerts, and topology context.

The canonical evidence and distractor definitions must declare every evidence
ID that appears in a rendering. Required causal evidence must be present except
when an `INCOMPLETE` scenario explicitly marks it removable. Unsupported
evidence cannot be silently introduced by a renderer.

No chain-of-thought is stored or required.

## Model output contract

The frozen `DiagnosticOutput` schema is:

```json
{
  "root_cause": "failure_mode_id or null",
  "affected_component": "component_id or null",
  "evidence_ids": ["evidence_id"],
  "remediation_runbook_id": "runbook_id or null",
  "needs_more_data": false,
  "required_evidence": []
}
```

An abstaining output sets `needs_more_data` to `true`, asserts no diagnosis or
remediation, cites no diagnostic evidence, and requests declared evidence IDs.
Self-reported confidence is intentionally absent; if added later it will not
be treated as calibrated probability.

## Split protocol and leakage policy

Split assignment happens on canonical scenarios before rendering. The checked-in
split names are `TRAIN`, `VALIDATION`, `ID_TEST`, `HARD_TEST`, `TEMPLATE_OOD`,
`TOPOLOGY_OOD`, `GENERATOR_OOD`, `COUNTERFACTUAL_TEST`, `ABSTENTION_TEST`, and
`SEALED_HOLDOUT`.

Assignments are explicit in a `SplitManifest`; there is no random
example-level split helper. The validator fails closed when:

- a scenario, paraphrase/archetype group, or protected topology group crosses
  split boundaries;
- counterfactual pair members are missing, split apart, or not both marked
  `COUNTERFACTUAL`;
- template families in `TEMPLATE_OOD`, topology groups in `TOPOLOGY_OOD`, or
  generator families in `GENERATOR_OOD` occur in `TRAIN`;
- manifest assignments disagree with the canonical scenario metadata; or
- scenario IDs are duplicated or omitted.

`SEALED_HOLDOUT` is excluded from threshold selection, checkpoint selection,
model choice, prompt tuning, ablations, and data tuning. After its one planned
evaluation, `SplitManifest.consume_sealed_holdout()` marks the manifest as
consumed evidence. Repeated evaluation is a future policy violation, not a
selection workflow.

## Provenance and fingerprints

Canonical JSON serialization is sorted and compact. Scenario, rendering, and
split-manifest fingerprints use SHA-256. Provenance records experiment,
ontology, scenario, telemetry, and output-contract versions; archetype,
generator, and template families; deterministic seed; renderer provider/model
and prompt-template slots; canonical and rendered hashes; and the split-manifest
hash. A future renderer must fill provider metadata without changing labels.

## Dataset audit contract

`audit_dataset` reports class/family and difficulty distributions and fails
closed on schema errors, unknown taxonomy IDs, root-cause/domain mismatches,
invalid evidence references, answerability contradictions, duplicate canonical
content, exact rendered duplicates, split leakage, counterfactual-pair
violations, and forbidden OOD overlap. Embedding-based near-duplicate semantic
detection is intentionally deferred; it may be added as a later audit without
changing this V1 contract.

## Evaluation contract

Future runs will use deterministic metrics for root-cause exact accuracy,
affected-component accuracy, diagnosis joint exact, evidence precision/recall/
F1, remediation/runbook accuracy, valid JSON, schema-valid rate, abstention
precision/recall/F1, and false-diagnosis rate on insufficient-evidence cases.
Metrics are sliceable by root-cause family, fault domain, difficulty, topology
family, runtime/ecosystem, generator family, template family, and evaluation
split. Primary evaluation requires no LLM-as-judge.

Answer-quality metrics use answerable cases as their denominator. Abstention
metrics use all cases; false-diagnosis rate uses only insufficient-evidence
cases. This denominator policy is fixed before any future model is evaluated.

## Intentionally deferred to 03B+

- deterministic scenario archetype and causal-state generation;
- telemetry rendering, provider selection, prompt templates, and dataset
  generation;
- large-corpus creation and final split freezing;
- untouched base-model capability-gap screening;
- model selection, QLoRA specialization, and training ablations;
- sealed benchmark execution and result interpretation; and
- real/manual incident corpus development subject to licensing and privacy.

Experiment 03A freezes the foundation only. It contains no evidence that a new
specialist model works.
