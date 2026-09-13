# v1.0 Readiness Audit

Status: `PASS`.

This audit is generated from persisted CauseTune evidence. It does not
rerun semantic experiments. EX-LV is optional and is not required for v1.0.

| Experiment | Contract | Model | Benchmark hash | Checkpoint | Verification |
| --- | --- | --- | --- | ---: | --- |
| E04-A | `e04a-selection-v1` | Qwen/Qwen3-4B | `not applicable` | 100 | PASS for all four valid semantic runs; 75% technical failure retained separately |
| E04-B | `e04b-selection-v1` | Qwen/Qwen3-4B | `not applicable` | 100 | PASS for r8, r16, and r32 |
| E04-C | `e04c-selection-v1` | Qwen/Qwen3-4B | `not applicable` | 100 | PASS for 1e-4, 2e-4, and 4e-4 |
| E05 | `e05-one-shot-blind-v1` | Qwen/Qwen3-4B | `b3daed4f49b123b0270baebf49e7609c06f26e5bb10fd125185d6e0864644eaf` | — | PASS for base, E02, and E04; raw predictions and transitions preserved |
| E06 | `e06-one-shot-final-blind-v1` | microsoft/Phi-4-mini-instruct | `3811b2d5698a8a6530abbe550be0df7d83c4062098326f8b4384369681cda714` | 125 | PASS for capability screen, training evidence, and separate final base/tuned benchmark |
| E07 | `e07-boundary-v1` | Qwen/Qwen3-4B | `c797b3a8378a2cbb2cb1e2a27e05a561087a62ed3f9adc74aa050d9f3f42a800` | — | PASS for all three systems; scorer correction recorded without semantic rerun |
| E08 | `e08-efficiency-frontier-v1` | Qwen/Qwen3-4B | `b3daed4f49b123b0270baebf49e7609c06f26e5bb10fd125185d6e0864644eaf` | — | PASS: deterministic analysis of persisted E02/E04 cost and E05 blind evidence |

## Mandatory gate closure

- E04 controlled optimization studies: `PASS`.
- E05 fresh blind generalization: `PASS`; E04 diagnosis regression versus E02 remains `-5.83 pp`.
- E06 cross-model replication: `PASS`; Phi capability gap and final held-out evaluation are separate.
- E07 failure boundary: `PASS`; E04 false confident diagnosis remains non-zero.
- E08 efficiency frontier: `PASS`; E04 is a cheaper negative trade-off, not quality-preserving.

## Limitations

- Synthetic benchmark construction and taxonomy alignment.
- Limited model-family coverage despite the controlled Phi replication.
- One-shot blind challenges and static/manual fixture limitations where disclosed.
- Hardware-specific cost measurements.
- Adapter weights are local-only; lightweight hashes, manifests, metrics, and locations are persisted.
