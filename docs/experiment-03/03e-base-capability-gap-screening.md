# Experiment 03E — Untouched Base-Model Capability-Gap Screening

**Status: COMPLETE — validation-only base screen.** No model was trained, no
adapter was created, and no TEST example was model-facing. This milestone asks
which untouched Qwen3.5 candidate is the smallest credible student for future
specialization on the frozen Cloud-OpsBench contract.

## Frozen evidence boundary

The screen used Cloud-OpsBench revision
`03c415e5709297432282fbbfd499f1bca0f8c347`, the immutable 03D split
(`510 TRAIN / 123 VALIDATION / 121 TEST`), split fingerprint
`0ed9845d566e661ed0772fe82617cb1676cdca266f27f5badeed1f49ee5f5c76`, and
representation fingerprint
`f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf`.

`VALIDATION_SCREEN` was frozen before execution as the lexicographically first
case per native fault type: 57 cases covering all 57 native types. The remaining
66 VALIDATION cases form `VALIDATION_SELECTION` and received no model
predictions. TEST remained sealed.

## Candidates and common conditions

The exact post-trained candidates and resolved revisions were:

| Candidate | Revision | Parameters | Native context |
| --- | --- | ---: | ---: |
| `Qwen/Qwen3.5-0.8B` | `2fc06364715b967f1860aea9cf38778875588b17` | 873,438,784 | 262,144 |
| `Qwen/Qwen3.5-2B` | `15852e8c16360a2fea060d615a32b45270f8a8fc` | 2,274,069,824 | 262,144 |
| `Qwen/Qwen3.5-4B` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | 4,659,865,088 | 262,144 |

All used the same frozen prompt, tool schemas, deterministic greedy decoding,
`enable_thinking=False`, `max_new_tokens=256`, and a 20-step agent limit. The
primary hardware-constrained configuration was NF4 4-bit with BF16 compute and
double quantization on an RTX 5070 Laptop GPU (8,151 MiB reported VRAM). Model
licenses are Apache-2.0 at the pinned snapshots. Config, tokenizer, and
chat-template hashes are recorded in `results/incident_telemetry_03e/`.

## Results on VALIDATION_SCREEN

### Primary closed-loop agent

| Candidate | Native type exact | Category exact | Fault object exact | Joint exact | Schema valid | Valid tool-call rate | Invalid/replay-call rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 0/57 | 0/57 | 0/57 | **0/57** | 0.00% | 0.00% | 0.00% |
| 2B | 0/57 | 0/57 | 0/57 | **0/57** | 52.63% | 100.00% | 47.37% |
| 4B | 0/57 | 0/57 | 0/57 | **0/57** | 0.00% | 100.00% | 100.00% |

The 0.8B model stopped with premature/non-tool output on all cases. The 2B
model produced 30 schema-valid final answers, but none matched the native
target and 27 cases failed replay after invalid arguments. The 4B model made
valid tool selections but all 57 cases eventually hit an invalid/replay call
before a final diagnosis.

### Oracle-evidence diagnosis

This secondary mode supplied leakage-safe observations from an executable
golden history and asked only for the structured diagnosis.

| Candidate | Native type exact | Category exact | Fault object exact | Joint exact | Schema valid |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.8B | 0/57 | 0/57 | 0/57 | **0/57** | 33/57 (57.89%) |
| 2B | 0/57 | 0/57 | 0/57 | **0/57** | 57/57 (100.00%) |
| 4B | 0/57 | 0/57 | 0/57 | **0/57** | 7/57 (12.28%) |

All candidates therefore lack a positive diagnosis learnability signal under
the decomposition that removes evidence acquisition from the problem.

### Teacher-forced next action

There were 160 eligible executable golden steps per candidate. Tool-name exact
rates were 26.88% (0.8B), 18.75% (2B), and 40.63% (4B). Tool-name validity and
argument-schema validity were 100% for all three in this teacher-forced mode.
The 4B model shows a process-capability gradient, but that did not translate
into diagnosis accuracy.

The frozen static packaged-view ablation was not run: the 03D static manifest is
provenance-only and does not contain a separate materialized input view. Running
it here would have introduced a new packaging strategy.

## Token and context measurements

Actual candidate tokenizers were resolved at the pinned revisions; all three
share the same tokenizer hash. The screen final-context distribution was
`min 1,938 / median 3,999 / p90 10,754 / p95 22,506 / p99 29,487 / max 35,037`
tokens. TRAIN-derived final contexts used for feasibility statistics were
`min 1,931 / median 3,134 / p90 7,446 / p95 9,658 / p99 20,772 / max 23,911`.
The screen’s initial/action contexts had median 2,468 and max 35,041 tokens;
individual observations had median 505 and max 15,207 tokens.

Final-context threshold counts were recomputed from the frozen TRAIN and
VALIDATION_SCREEN representation and are stored in each candidate’s
`tokenization_metrics.json`. No TEST content was tokenized for model-facing
work. The candidates’ native 262,144-token architectural limit was therefore
not the practical constraint; local memory and latency were.

## Hardware feasibility

All length probes technically completed at TRAIN median, p95, and max context.
Peak allocated memory at the max 23,911-token probe was 3.303 GiB for 0.8B,
4.449 GiB for 2B, and 8.554 GiB for 4B. The 4B maximum probe exceeded the
machine’s 8 GiB allocated-memory envelope; its 968.2-second latency also makes
that operating point impractical. Allocator reserved-memory values are
reported in the per-candidate artifacts and are not treated as physical VRAM.

## Interpretation and recommendation

The result is **NO_CREDIBLE_STUDENT_YET**. This is not a case where the largest
model should be selected by score: all three candidates scored zero joint
diagnosis on both the primary closed loop and oracle-evidence diagnosis. The 2B
candidate is the most balanced protocol/footprint option (100% oracle schema
validity, 52.63% closed-loop schema validity, and 4.449 GiB max allocated probe)
but still has no diagnosis signal. The 4B process score is higher, yet its
oracle diagnosis is also zero and its longest probe exceeds the local memory
envelope.

The exact recommendation for 03F is: **do not start QLoRA yet**. First review
target/prompt/tool compatibility and obtain a credible oracle-evidence signal
under the frozen contract. No 03F implementation or model training is included
here.

## Integrity notes

The initial run completed closed-loop and oracle generation for all three
candidates. A serialization bug occurred after 4B teacher-forced generation;
only that missing teacher artifact was recovered with a bounded 4B teacher
execution. The post-processing parser was then corrected offline for preserved
Qwen control tokens, and derived diagnosis metrics were recomputed from the
persisted raw decoded outputs. No closed-loop/oracle generation was repeated,
no TEST predictions were produced, and no output was semantically repaired.

All raw strings remain in the prediction JSONL artifacts. The recovery and
candidate-comparison metadata are recorded under
`results/incident_telemetry_03e/`.

## Limitations

- The screen contains one deterministic validation case per native type, not a
  performance estimate for the full 123-case VALIDATION partition.
- The native fault types and target objects are heterogeneous; zero-shot
  structured diagnosis is substantially harder than schema emission.
- The static ablation was not executable without creating new input material.
- Hardware measurements are local probes, not production serving benchmarks.
- TEST remains protected and is reserved for the final Experiment 03 evaluation.
