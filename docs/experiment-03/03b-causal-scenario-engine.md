# Experiment 03B — Deterministic Causal Scenario Engine

## Status

**COMPLETE FOUNDATION MILESTONE.** Experiment 03B implements the deterministic
canonical-scenario engine and a small CPU-safe reference corpus. It contains no
LLM-rendered telemetry, model evaluation, model selection, fine-tuning, or
performance evidence.

## Objective and evidence boundary

Experiment 03 asks whether a language model can learn incident diagnosis from a
causally controlled telemetry corpus and generalize across scenarios,
topologies, telemetry surfaces, and generator families. Experiment 02 is
historical evidence only and is not reused for Experiment 03 selection,
tuning, checkpoint selection, ablations, or final claims.

03B establishes the authoritative scenario layer needed for that new boundary:

```text
versioned ontology
    -> declarative archetype
    -> deterministic canonical scenario
    -> future telemetry renderer
    -> model input
```

The canonical scenario owns causal truth. A future renderer may translate that
truth into observable telemetry, but it cannot create or edit labels. The 03B
reference corpus is fixture/reference evidence, not training data or a
benchmark result.

## Archetypes and ontology coverage

Archetypes are typed, versioned, declarative descriptions of scenario families.
Each defines compatible fault domain, affected-component roles, runtimes,
topologies, causal variables and invariants, evidence plan, distractor rules,
change/dependency constraints, remediation runbook, supported difficulties, and
split-group family. Natural-language telemetry is deliberately absent.

The initial 03B catalog implements these 12 ontology IDs:

| Domain | Implemented archetypes |
| --- | --- |
| RESOURCE | `container_memory_exhaustion`, `cpu_throttling` |
| DATABASE | `db_connection_pool_exhaustion`, `db_query_regression` |
| DEPENDENCY | `downstream_dependency_timeout` |
| NETWORK | `dns_resolution_failure` |
| APPLICATION | `thread_pool_exhaustion`, `configuration_regression` |
| MESSAGING | `kafka_consumer_lag` |
| CACHE | `cache_stampede` |
| KUBERNETES_RUNTIME | `readiness_probe_failure` |
| SCALING | `insufficient_replicas` |

The 16 intentionally deferred IDs are `memory_leak`, `disk_io_saturation`,
`disk_pressure`, `db_capacity_saturation`, `downstream_dependency_unavailable`,
`rate_limit_misconfiguration`, `connection_refused`,
`tls_certificate_expiration`, `worker_pool_exhaustion`,
`message_broker_unavailable`, `cache_unavailable`, `crash_loop`,
`scheduling_failure`, `node_resource_pressure`, `autoscaling_failure`, and
`traffic_capacity_exhaustion`. They remain valid 03A ontology IDs; they are not
silently treated as implemented scenario families.

## Topology and runtime dimensions

Topology is selected from a bounded catalog rather than an unconstrained random
graph:

- `WEB_DB`: frontend → API → Postgres
- `WEB_CACHE_DB`: frontend → API → Redis/Postgres
- `CHECKOUT`: frontend → checkout API → payment API/Redis/Postgres
- `ASYNC_WORKER`: API → broker → worker → Postgres
- `EVENT_PIPELINE`: producer → Kafka → consumer → database
- `MULTI_SERVICE`: gateway → orders/inventory/payment APIs

Every affected component is resolved from an existing topology role, and every
dependency reference is checked against the graph. Runtime/ecosystem is a
separate bounded dimension: `python_fastapi`, `python_django`, `java_spring`,
`go_http`, and `node_nestjs`. Archetypes declare compatibility explicitly, so
the same causal family can vary runtime where defensible without admitting
impossible combinations.

## Canonical causal state and evidence

Each generated scenario contains typed causal variables with archetype-specific
invariants. Examples include memory limit versus working set, CPU demand versus
limit and throttle ratio, connection-pool saturation, query latency, consumer
processing rate versus incoming rate, and replica capacity versus demand.
Invalid values fail closed; the engine never repairs an invalid scenario.

Semantic evidence definitions have stable IDs and deterministic attributes, but
no final natural-language text. Evidence roles are:

- `CAUSAL`: directly supports the authoritative diagnosis;
- `SUPPORTING`: consistent context that is not independently causal;
- `DISTRACTOR`: a true, approved non-causal observation;
- `MISSING_REQUIRED`: an explicitly removed required evidence definition for an
  incomplete case.

Hard and counterfactual scenarios include deterministic distractors selected by
archetype rule. Distractors are kept separate from causal evidence and cannot
overwrite causal state or accidentally become a second full causal signature.
Standard cases remain answerable with little noise. Incomplete cases remove
only archetype-declared removable required evidence and become
`INSUFFICIENT_EVIDENCE`; they are not produced by arbitrary random deletion.

## Counterfactual pairs

`CounterfactualPair` is first-class metadata. Pair members share topology,
runtime, affected service role, group identity, and the explicitly held
constant state variables. The metadata lists changed variables and root cause,
and validation rejects unrelated scenarios relabeled as a pair. The reference
fixture includes a controlled `container_memory_exhaustion` versus
`cpu_throttling` pair with a shared symptom envelope.

## Determinism and provenance

`ScenarioGenerator` uses an explicit per-request `random.Random` instance;
global random state and generation order do not affect an explicitly seeded
request. Scenario IDs derive from stable generation identity and do not depend
on future rendered telemetry. Canonical JSON serialization and SHA-256
fingerprints cover scenarios and manifests.

Provenance records the experiment, ontology, scenario, telemetry, and output
contract versions; archetype, generator, and template families; seed; canonical
scenario hash; and split-manifest hash. Renderer provider fields remain empty
until 03C and cannot become a source of ground truth.

## Pre-render split manifest

The reference manifest is assigned before any rendering. It uses the frozen 03A
split names: `TRAIN`, `VALIDATION`, `ID_TEST`, `HARD_TEST`, `TEMPLATE_OOD`,
`TOPOLOGY_OOD`, `GENERATOR_OOD`, `COUNTERFACTUAL_TEST`, `ABSTENTION_TEST`, and
`SEALED_HOLDOUT`.

Assignments carry stable scenario, split-group, topology-group, archetype,
generator-family, template-family, and pair identities. The validator rejects
cross-split protected groups, split-apart counterfactuals, and forbidden OOD
overlap with training. A sealed holdout can be marked consumed only through
the explicit manifest transition; it is not a tuning or selection surface.

The fixture manifest is intentionally small: 12 TRAIN, 8 VALIDATION, 5 ID_TEST,
5 HARD_TEST, 4 TEMPLATE_OOD, 4 TOPOLOGY_OOD, 4 GENERATOR_OOD, 4
ABSTENTION_TEST, 4 SEALED_HOLDOUT, and 2 COUNTERFACTUAL_TEST scenarios, for 52
canonical scenarios total.

## Audit and CLI

`audit_dataset` is fail-closed and reports structured findings for schema and
ontology validity, root-cause/domain compatibility, topology/component and
dependency references, causal-state invariants, evidence roles, answerability,
distractors, counterfactual pairs, duplicate canonical content, and grouped
split leakage. Embedding-based near-duplicate detection is intentionally
deferred.

The CPU-safe entrypoint is:

```bash
uv run python scripts/generate_incident_03b_reference.py \
  --output-dir results/incident_telemetry_03b --seed 302
```

It writes small inspectable `reference_scenarios.jsonl`,
`reference_manifest.json`, `audit.json`, `distribution.json`, and
`counterfactual_pairs.json` artifacts. It performs no network/provider call,
model loading, GPU work, telemetry rendering, or dataset-scale generation.

## Limitations and exact 03C boundary

The catalog covers a deliberately small cross-domain V1 subset. It does not
yet model multi-fault or cascading incidents, semantic near-duplicates, or
real/manual incident data. Topology and runtime catalogs are bounded and the
reference corpus is a construction fixture, not evidence of model capability.

03C begins only after this boundary: it may implement a provider-neutral
telemetry renderer and controlled rendering/dataset-generation workflow. The
renderer must consume `CanonicalScenario` and preserve its authority boundary;
it may not create labels, alter answerability, or introduce unsupported causal
evidence. No 03B artifact demonstrates that any language model can solve these
incidents.
