# Experiment 04B — LoRA-capacity results

## Gate status

`G04B PASS`. The r8, r16, and r32 semantic runs all completed under the
frozen E04-B contract and all three evidence bundles passed offline
verification. The selected E04-A 25% subset and fixed alpha 32 were preserved;
E03 was sealed and not used for rank selection.

## Validation comparison

| Rank | Diagnosis | Resolution | Failure-mode macro F1 | Selected checkpoint | Actual steps |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 288/288 | 288/288 | 1.000 | 100 | 100 |
| 16 | 288/288 | 288/288 | 1.000 | 100 | 100 |
| 32 | 288/288 | 288/288 | 1.000 | 100 | 100 |

All three runs also retained perfect culprit accuracy, action accuracy,
evidence F1, valid JSON, strict JSON, and per-family failure-mode F1. The
machine-readable comparison records trainable parameters, tokens, wall time,
VRAM, fingerprints, raw-prediction presence, and artifact-hash provenance.

## Frozen selection

All three ranks satisfy the predeclared validation tolerance rule. The smallest
eligible capacity is therefore selected:

```text
selected rank: 8
selected alpha: 32
selected data fraction: 25%
```

The selection is a result under the predeclared E04 controlled study, not a
claim of globally optimal adapter capacity. Records are under
`results/experiment_04b/`; local adapter weights remain excluded from source
control under the repository's large-artifact policy.
