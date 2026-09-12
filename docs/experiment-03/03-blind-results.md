# Experiment 03 — Blind Generalization Result

The frozen blind V2 benchmark contains 60 cases (`24/24/12` standard/hard/
transfer). Its fingerprint is
`10ffbcf9d60edc872d60c706f9c800beac1540b54cd08ad060e5169faa3a8150`; the
independent generator and exact-overlap audit passed before inference.

The untouched Qwen3-4B base and the existing selected E02 step-100 adapter were
evaluated once each with the same prompt, greedy generation and scorer. No
checkpoint switching or prompt change occurred.

| Metric | Base | Tuned | Delta |
| --- | ---: | ---: | ---: |
| Diagnosis exact | 55.00% (33/60) | **90.00% (54/60)** | **+35.00 pp** |
| Resolution exact | 28.33% (17/60) | **90.00% (54/60)** | **+61.67 pp** |
| Culprit accuracy | 71.67% | **90.00%** | +18.33 pp |
| Failure-mode accuracy | 61.67% | **90.00%** | +28.33 pp |
| Evidence F1 | 72.73% | **94.74%** | +22.01 pp |
| Strict JSON | 80.00% | **90.00%** | +10.00 pp |

Raw predictions and aggregate evaluations are persisted under
`results/experiment_03_blind/`. Re-running the deterministic incident scorer
from the persisted raw predictions reproduced both diagnosis and resolution
metrics. This is a single blind generalization result, not a claim that the
synthetic benchmark represents production incidents.
