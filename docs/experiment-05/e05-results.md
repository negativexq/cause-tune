# Experiment 05 — Fresh blind generalization

Status: `PASS`.

E05 evaluated one untouched Qwen3-4B base, the original E02 adapter, and the
final E04-selected adapter exactly once each on the frozen 120-case challenge.
The challenge is a fresh frozen synthetic blind challenge from an independent
held-out generator family. It reuses CauseTune’s taxonomy and strict scorer,
which is disclosed in the frozen protocol; it is not a real-world OOD or
production-accuracy claim.

## Frozen evidence

- Benchmark fingerprint: `b3daed4f49b123b0270baebf49e7609c06f26e5bb10fd125185d6e0864644eaf`
- Cases: 120 (`standard` 60, `hard` 36, `transfer` 24)
- Construction: 96 generated cases and 24 static-authored,
  generator-independent fixtures
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Prompt, decoding, and scorer were frozen before model evaluation.
- Raw outputs were written before scoring; each system used a fresh model
  reload and one deterministic generation per case.

## Results

| System | Diagnosis exact | Resolution exact | Failure-mode macro F1 | Strict JSON |
|---|---:|---:|---:|---:|
| Untouched base | 80/120 (66.7%) | 34/120 (28.3%) | 73.95% | 110/120 (91.7%) |
| Original E02 adapter | 118/120 (98.3%) | 118/120 (98.3%) | 98.73% | 119/120 (99.2%) |
| Final E04 adapter | 111/120 (92.5%) | 111/120 (92.5%) | 94.09% | 115/120 (95.8%) |

The E04 recipe therefore preserves a large gain over the untouched base, but
does not match the original E02 adapter on this fresh challenge. This is a
generalization regression, not a reason to alter the already-frozen E04
selection.

## Diagnosis transitions

| Comparison | Wrong → correct | Correct → wrong | Persistent correct | Persistent wrong |
|---|---:|---:|---:|---:|
| Base → E04 | 31 | 0 | 80 | 9 |
| E02 → E04 | 0 | 7 | 111 | 2 |

The E02 → E04 transition contains seven newly introduced diagnosis errors and
no E02 errors corrected by E04. All per-case transitions, raw predictions,
scored predictions, slice metrics, and hashes are persisted under
`results/experiment_05/`.

## Limitations

The challenge is synthetic and taxonomy-aligned, with limited model-family
coverage and static fixture limitations. Results are one-shot deterministic
evaluations on this frozen challenge and should not be generalized to
production incident accuracy.
