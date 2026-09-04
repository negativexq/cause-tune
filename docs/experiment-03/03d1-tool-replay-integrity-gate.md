# Experiment 03D.1 — Tool-Replay Parity and Cumulative Context Integrity Gate

## Status

**PASS — READY_FOR_03E.** This is a provider/model-free validation gate over
the frozen 03D representation. It does not alter the source-case split and did
not load, evaluate, or train a model.

The 03D split fingerprint remains
`0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76`.
The 03D.1 executable representation fingerprint is
`f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf`.

## Replay parity

Across 7,391 tool observations:

| Class | Count |
| --- | ---: |
| EXACT_REPLAY | 6,384 |
| CANONICALLY_EQUIVALENT_REPLAY | 633 |
| SAFE_SOURCE_DERIVED_REPLAY | 214 |
| GOLDEN_ONLY_FALLBACK | 160 |
| UNRESOLVED | 0 |

The 1,007 original fallbacks classify as:

- 633 `NORMALIZATION_MISMATCH`: structured cache values and trace values are
  semantically equal after JSON/Python-literal normalization. This covers 445
  `GetAlerts` and 188 `GetClusterConfiguration` observations.
- 214 `SOURCE_ARTIFACT_DERIVABLE`: 196 `GetSourceCode` outputs are recovered
  from deterministic `code/` file selection and 18 `GetRecentLogs` outputs are
  recovered from deterministic `logs.json` service/line selection.
- 160 `GOLDEN_ONLY_NOT_REPLAYABLE`: all are `GetRecentLogs` outputs for which
  no cache entry or exact source-derived selection exists.

No malformed or unknown tool was observed. Exact classifications, fingerprints,
case identity, split, and reproducibility flags are in
`results/incident_telemetry_03d1/fallback_classification.json`.

The golden-only policy is **auxiliary-only**. Affected paths are excluded from
the executable primary SFT representation rather than silently retaining an
unavailable observation or truncating an expert trajectory. This leaves 1,157
eligible TRAIN/VALIDATION trajectory paths and 351 auxiliary paths; the
regenerated executable records are in
`results/incident_telemetry_03d1/primary_trajectory_manifest.jsonl`. All TEST
paths remain non-model-facing protected evidence.

## Packaging and evidence retention

The 32,768-character observation policy produces 69 omissions. The offline
auditor compared packages against process-label evidence patterns after
packaging; this does not influence packaging selection. All 5,323 observed
required-pattern matches were retained (100.0%); zero required evidence
matches were lost. Candidate budgets produced:

| Budget | Affected observations | Affected trajectories | Required evidence lost |
| ---: | ---: | ---: | ---: |
| 8,192 chars | 551 | 348 | 0 |
| 16,384 chars | 199 | 142 | 0 |
| 24,576 chars | 100 | 65 | 0 |
| 32,768 chars | 69 | 39 | 0 |

The 32,768-character bound is retained because it minimizes omissions while
preserving every audited required-pattern match. This is an integrity result,
not a model-quality result.

## Cumulative context

With full lossless history and the 32,768-character observation bound, final
diagnosis contexts measured:

| Statistic | Characters | Bytes |
| --- | ---: | ---: |
| Minimum | 2,046 | 2,046 |
| Median | 10,524 | 10,532 |
| P75 | 19,785 | 19,855 |
| P90 | 32,696 | 32,702 |
| P95 | 56,611 | 56,611 |
| P99 | 82,341 | 82,341 |
| Maximum | 150,042 | 150,042 |

The same full-history distribution applies to complete trajectory final
contexts. A 64k-token candidate context (using the conservative bytes/4
estimate) covers the observed maximum; a 32k context leaves five final
contexts above budget. The 8k-token estimate is insufficient for 151/1,508
final contexts, 16k for 49, and 32k for five. These are estimates, not
tokenizer measurements.

Deterministic compact-history variants are much smaller: structured compact
history has median 1,667 and maximum 4,722 characters; observation-summary
state has median 1,314 and maximum 3,803. Both replace observation content
with references or metadata and are therefore lossy. Full history remains the
preferred primary policy when the candidate context supports it.

## Representation boundary and 03E

The frozen 03D split is unchanged. The new representation contract distinguishes
split freeze from executable representation freeze. Primary SFT may use only
TRAIN/VALIDATION paths with exact, canonical, or safe source-derived replay;
golden-only paths are auxiliary-only. TEST remains sealed and is not used for
model selection, prompt/packaging tuning, or candidate screening.

03E may screen untouched base models using TRAIN/VALIDATION. Candidate models
should preferably support at least a 64k-token context for lossless full-history
screening under this estimate; a 32k candidate requires an explicit policy for
the five-case overflow tail. No model selection decision was made in 03D.1.

## Limitations

- The source trace/cache mismatch is a property of the pinned benchmark
  artifacts; the fallback policy should be rechecked if the source revision
  changes.
- Process-label regex matches are a mechanical retention proxy, not a proof
  that every semantic evidence relation survives packaging.
- Full-history contexts have a long tail beyond 64k tokens.
- Code availability is system-confounded and metrics are absent by source
  design in most cases.
