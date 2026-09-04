# Renderer Prototype R0 — Controlled LLM Telemetry Rendering Pilot

> **Superseded as the primary Experiment 03C direction.** This implementation
> is retained as supporting renderer infrastructure: **IMPLEMENTED / LIVE
> PROVIDER NOT RUN**, not scientific model evidence, and not on the current
> critical path. It may support controlled augmentation after the Cloud-OpsBench
> corpus split and contamination boundary are frozen.

## Status

**IMPLEMENTED / LIVE PROVIDER NOT RUN.** The rendering boundary, one
OpenAI-compatible adapter, deterministic fake provider, pilot orchestration,
audits, and artifacts are implemented. The checked-in pilot is an offline
fake-provider dry-run; it is not evidence about an LLM or about diagnostic
model quality. No specialist model has been trained or evaluated.

## Purpose and authority boundary

03C tests whether an authoritative `CanonicalScenario` can be translated into
realistic telemetry surfaces without allowing a renderer to become the source
of ground truth:

```text
CanonicalScenario -> RenderPlan -> provider response -> validated RenderedTelemetry
```

The scenario remains authoritative for root cause, domain, affected component,
answerability, runbook, evidence roles, missing-evidence semantics, topology,
identity, pair relationships, and split membership. The renderer produces only
observable realizations of predeclared semantic evidence.

## RenderPlan

`build_render_plan` exposes exactly the declared causal/supporting/distractor
evidence definitions, including evidence ID, semantic kind, source component,
role, deterministic attributes, omission permission, permitted surface, and
renderer-family variation. It does not expose root-cause IDs, runbooks,
archetype IDs, split names, generator metadata, or a diagnostic output schema.

03A’s existing `RenderedTelemetry` model remains the accepted domain object.
The provider wire format uses a bounded uppercase surface vocabulary and is
converted to the frozen lower-case observation kinds only after validation.

## Surfaces and renderer families

The V1 surfaces are `APPLICATION_LOG`, `RUNTIME_EVENT`, `METRIC`,
`DEPLOYMENT_CHANGE`, `DEPENDENCY_HEALTH`, and `ALERT`. Stable evidence IDs are
preserved; providers cannot invent, duplicate, or relabel them.

Three distinct renderer families are configured:

| Family | Prompt contract |
| --- | --- |
| `concise_ops_v1` | Compact operational records with one observable fact per record |
| `verbose_enterprise_v1` | Complete contextual sentences and incident-timeline phrasing |
| `ecosystem_native_v1` | Runtime-appropriate terminology for the supplied ecosystem |

These are different prompt/template contracts, not the same prompt with a
different random seed. Their IDs and prompt versions are recorded in the
pilot manifest and provenance, making a future `GENERATOR_OOD` holdout
representable.

Runtime hints cover `python_fastapi`, `python_django`, `java_spring`,
`go_http`, and `node_nestjs`. They affect wording only; they cannot alter
causal state or labels.

## Provider interface and safety

`ProviderClient` is provider-neutral and returns a raw structured response with
optional provider-reported token usage. `FakeTelemetryProvider` is deterministic
and is the default. `OpenAICompatibleProvider` is the single real adapter; it
uses only stdlib HTTP, reads credentials from an environment variable, records
model/generation configuration, has a timeout and bounded network retries, and
never persists credentials.

Network execution requires the explicit `--provider-live` flag and a model
identifier in `CAUSETUNE_03C_MODEL`. The default CLI path cannot call a
provider. Provider failures, malformed JSON, and contract-invalid responses
remain failed attempts; no second model or semantic repair is used.

The prompt explicitly says: “You are a telemetry renderer, not an incident
diagnostician.” It forbids diagnosis, remediation, unsupported evidence,
evidence-ID changes, missing-evidence resurrection, and causal-role changes.

## Deterministic validation and leakage protection

After every provider response, the pilot deterministically checks strict
structured shape, declared evidence coverage, intentional omissions, component
identity, permitted surface, duplicate IDs, canonical scenario fingerprint,
split-manifest fingerprint, and the frozen 03A renderer validator.

Leakage checks reject root-cause and ontology machine labels, remediation IDs,
answerability/case labels, internal archetype IDs, split names, generator and
template metadata, and diagnostic-output field names in rendered content.
Generic operational words such as “database” and “dependency” remain legal
telemetry vocabulary; the check targets exact machine labels rather than
ordinary domain language.

## Retry and artifact policy

The pilot has three bounded rendering attempts per manifest entry. Retries do
not change the scenario or render seed. Raw responses are written once per
attempt under `raw/`; rejected attempts retain category, reason, and raw hash
under `rejected/`; accepted `RenderedTelemetry` is a derived artifact under
`accepted/`. A non-empty output directory is refused to prevent accidental raw
artifact replacement.

Every accepted record includes canonical scenario and render hashes, renderer
family/template version, provider adapter/model, generation parameters, render
seed, attempt, raw-response hash, pilot-manifest fingerprint, and usage when
available. Timestamps are deterministic in the fake provider and are not used
in canonical scenario fingerprints.

## Pilot sampling and audit

The pilot builds a manifest before provider calls from the 03B reference
corpus, excludes `SEALED_HOLDOUT`, and renders the remaining 48 scenarios with
three renderer families: 144 bounded entries. This is a dry-run fixture, not a
final training corpus. The manifest records canonical IDs/fingerprints,
renderer family, provider config, render seed, and expected evidence IDs.

The offline fake pilot completed with 144 attempted, 144 accepted, 0 rejected,
0 retries, 0 exact duplicate rendered surfaces, and a deterministic inspection
sample of 20 accepted cases. These numbers describe fake-provider contract
execution only, not LLM quality or acceptance in production-scale generation.

`audit.json` reports acceptance and failure categories by provider/model and
renderer family, plus root cause, domain, runtime, topology, and difficulty.
`distribution.json` adds lightweight exact/normalized message duplicate rates,
vocabulary size, and per-family repetition. Lexical diversity does not prove
semantic diversity; embedding-based near-duplicate detection remains deferred.
`inspection_sample.json` contains no expected diagnosis and is safe as a
surface-review artifact.

## Limitations and 03D boundary

Telemetry remains synthetic, even when a live LLM renderer is used. Realistic
looking logs are not equivalent to real production telemetry. The fake pilot
does not establish provider behavior, and no model-performance claim exists.
The current pilot is small, uses the bounded 03B scenario catalog, and does
not consume `SEALED_HOLDOUT`.

03D is the next boundary: execute controlled production-scale rendering only
after provider selection and credential review, run full integrity/leakage and
near-duplicate audits, freeze the resulting corpus, and preserve the sealed
holdout. 03D must not train or evaluate a specialist model as part of that
corpus-freeze milestone.
