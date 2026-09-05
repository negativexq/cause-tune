# Experiment 03E.2 — Frozen V2 capability rescreen

## Status

The frozen V2 rescreen ran only `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`, on the unchanged 57-case
`VALIDATION_SCREEN`. No TEST or `VALIDATION_SELECTION` material was made
model-facing. No training or LoRA adapter was created.

V1 and 03E.1 remain immutable historical evidence. Their scores were not
replaced by these results.

## Why V2 changed the contract

03E.1 found that 2B produced 57/57 schema-valid V1 oracle responses but
0/57 strict native diagnoses. The field audit found source-family-plausible
noncanonical descriptions, 17 mechanical resource-prefix mismatches, and 21
genuinely wrong audited target-field comparisons. Purely mechanical diagnosis
equivalence was still 0/57, so V2 was not allowed to claim that normalization
fixed the V1 diagnosis result.

V1 also recorded 84/84 replay failures for `GetRecentLogs {}` because that
empty invocation had no executable case-local cache source. V2 disables that
invocation/tool path. Optional absent sources such as code now return an
explicit deterministic “source unavailable” runtime result; this is not a
fabricated observation and does not consult the target. The corrected pass had
zero environment-contract replay failures.

V1 exact next-action scoring was retained only as historical evidence. V2 does
not run teacher-forced next-action imitation as a primary metric. Process
behavior is scored from upstream milestone/evidence semantics.

## Frozen V2 contract

Before the first generation, V2 froze the ordered native vocabulary from
`UPSTREAM_FAULT_TAXONOMY`: 8 native categories and all 57 native fault-type
identifiers. The vocabulary was identical for every case and the current
case's label was never highlighted. This is a closed-taxonomy task schema,
not per-case target leakage; no correct case answer was supplied.

The final target schema was exactly:

```json
{"root_cause": "<native fault type>", "fault_category": "<native category>", "fault_object": "<open resource identity>"}
```

Only `root_cause` and `fault_category` were closed enums. `fault_object`
remained open and source-grounded. The only normalization accepted in scoring
was a bare application name to `app/<name>` when that application identity was
uniquely visible in the model-visible observations. Kubernetes pod,
deployment, service, namespace, and node identities were not collapsed.

The prompt also added one generic termination rule: once sufficient evidence
has been collected to select a diagnosis, emit the structured diagnosis
instead of continuing to call tools. There were no examples, few-shot cases,
case-specific repairs, or post-inference prompt edits.

## Results

### Oracle evidence diagnosis

| Metric | V2 result |
|---|---:|
| Schema valid | 57/57 (100.00%) |
| Valid native enum | 55/57 (96.49%) |
| Missing fields | 0 |
| Root cause strict exact | 9/57 (15.79%) |
| Category strict exact | 17/57 (29.82%) |
| Object strict exact | 0/57 (0.00%) |
| Object normalized exact | 18/57 (31.58%) |
| Joint strict exact | 0/57 (0.00%) |
| Joint normalized exact | 1/57 (1.75%) |

The V2 vocabulary improved native-label compliance, but the full diagnosis
signal remained very small and non-saturated. The strict and normalized
metrics are both retained: the normalized object successes do not rewrite the
strict V1 or V2 scores.

For direct comparison, frozen V1 2B oracle evidence was schema-valid 57/57
but native fault type, category, object, and joint exact were all 0/57;
closed-loop joint exact was 0/57 and teacher-forced exact-next-action was
18.75%. V2 therefore improved enum/schema compliance and field-level oracle
signal, but not full diagnosis or closed-loop completion.

### Closed loop

The corrected executable replay pass produced:

- final structured diagnosis: 0/57;
- schema-valid final diagnosis: 0/57;
- valid and executable tool calls: 605 and 605;
- invalid/unadvertised tool selections: 1;
- replay failures: 0;
- environment-contract failures: 0;
- repeated-tool-loop rate: 56/57 (98.25%);
- no-final diagnoses: 57/57;
- max-step exhaustion: 0/57;
- premature final: 0/57.

The first closed-loop pass is preserved separately as
`closed_loop_predictions_initial_contract_gap.jsonl`. It contained 50
`GetSourceCode` missing-source failures and is not used as the final V2
closed-loop score. The final score uses the corrected explicit-unavailable
runtime semantics.

### Process and termination behavior

Using upstream milestone/evidence matching, the final closed-loop traces had:

- `PROCESS_ADMISSIBLE_PROGRESS`: 28/605 actions (4.63%);
- `VALID_BUT_REDUNDANT`: 577/605 actions (95.37%);
- `process_complete`: 2/57 cases (3.51%);
- `EVIDENCE_SUFFICIENT_BUT_NO_FINAL`: 2/57 cases (3.51%).

No deterministic teacher-forced exact-next-action metric was used as a primary
selection measure. Exact golden-path imitation remains auxiliary-only.

Termination failures were 56 repeated tool loops and one no-final/tool
selection failure. The two evidence-sufficient cases demonstrate a small
termination-policy failure component, but the dominant failure is failure to
make process progress and finalize across the screen.

## Hardware

The 2B run was locally feasible with NF4 4-bit, BF16 compute, double
quantization, greedy decoding, `enable_thinking=False`, 256 new tokens, and a
20-step maximum. V2 observed peak memory was 6,123,361,280 allocated bytes and
8,793,358,336 reserved bytes; the largest observed oracle prompt was 35,339
tokens and the largest closed-loop prompt was 5,100 tokens.

4B was not rerun. Its V1 evidence remains hardware-constrained and incomplete:
44/57 oracle calls failed with CUDA/CUBLAS errors, so it is not a clean
capability estimate.

## V1 → V2 interpretation and gate

V1 was partially mis-specified: native labels were open-vocabulary outputs,
the empty `GetRecentLogs` replay path was not executable, exact golden-next
action was too restrictive, and termination semantics were underspecified.
V2 repaired those contract defects without per-case prompt tuning.

V2 shows that 2B can comply with the fixed diagnosis schema and enum at a
high rate in oracle mode, but it does not show usable closed-loop protocol
behavior or meaningful joint diagnosis accuracy. The pattern is therefore:

`BASE_TOO_WEAK`

The final recommendation is:

`DO_NOT_START_QLORA`

03F remains blocked. A future task redesign may be warranted if the goal is to
study process-capable diagnosis, but no QLoRA specialization was started by
03E.2.

Contract, vocabulary, prompt, raw predictions, metrics, hardware evidence,
and fingerprints are under `results/incident_telemetry_03e2/`.
