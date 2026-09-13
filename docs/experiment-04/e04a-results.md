# Experiment 04A — Data-efficiency results

## Gate status

`G04A PASS`. The four semantic runs were verified offline before selection:
25%, 50%, 75% retry-01, and 100%. The original 75% attempt remains preserved
as `TECHNICAL_FAILURE` and was excluded from quality comparison. E03 was not
used for tuning or selection.

## Validation comparison

| Variant | Examples | Diagnosis | Resolution | Failure-mode macro F1 | Selected checkpoint | Actual steps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 25% | 600 | 288/288 | 288/288 | 1.000 | 100 | 100 |
| 50% | 1200 | 288/288 | 288/288 | 1.000 | 100 | 100 |
| 75% retry-01 | 1800 | 287/288 | 287/288 | 1.000 | 100 | 100 |
| 100% | 2400 | 288/288 | 288/288 | 1.000 | 100 | 100 |

All variants retained 1.000 culprit accuracy, action accuracy, evidence F1,
valid JSON, and strict JSON compliance. Per-family failure-mode F1 was 1.000
for every valid run. The persisted comparison also records token counts,
wall-clock time, VRAM, trainable parameters, family metrics, fingerprints,
and artifact-hash provenance.

## Frozen selection

The best validation result is tied at 1.000 diagnosis exact, 1.000 resolution
exact, and 1.000 failure-mode macro F1. All four valid variants satisfy the
predeclared tolerance rule, so the smallest eligible fraction is selected:

```text
selected fraction: 25%
selected examples: 600
selected subset hash: eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56
```

This is a validation-saturation finding for the frozen task, not evidence that
25% is globally sufficient for fresh generalization. The machine-readable
comparison and selection records are `results/experiment_04a/`; the raw run
bundles remain under `runs/experiment_04a/`. Adapter checkpoint weights are
local-only and ignored by repository policy; the run-local hash indexes and
their hash inventories in the comparison preserve their provenance.
