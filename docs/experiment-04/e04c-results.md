# Experiment 04C — learning-rate sensitivity results

## Gate

`G04C PASS`. The three declared semantic runs completed, each persisted raw
predictions and provenance, and each evidence bundle reproduced successfully
with the offline verifier. No technical failure or retry occurred.

## Frozen comparison

| Variant | Learning rate | Selected checkpoint | Diagnosis | Resolution | Failure-mode macro F1 | Strict JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| lr1e-4 | 0.0001 | 100 | 288/288 | 288/288 | 1.0 | 288/288 |
| lr2e-4 | 0.0002 | 100 | 288/288 | 288/288 | 1.0 | 288/288 |
| lr4e-4 | 0.0004 | 100 | 288/288 | 288/288 | 1.0 | 288/288 |

All variants used the E04-A selected 25% subset and E04-B selected rank 8,
alpha 32. All used validation-only checkpoint selection, stopped after 100
actual optimizer steps for validation non-improvement, and had no critical
failure-family or schema-validity regression. The complete provenance-backed
comparison is in `results/experiment_04c/validation_comparison.json`.

## Selection

The frozen validation-only rule found all three variants eligible. Their
quality, checkpoint stability, and earliest stable convergence were tied, so
the predeclared final training-cost tie-break selected `lr1e-4`. This is the
selected learning rate under the predeclared E04 controlled study, not a claim
of global optimality.

E03 and all blind evidence were excluded from selection. Adapter weights are
local-only and are not committed; manifests, hashes, metrics, predictions,
selection, and verification records are persisted.
