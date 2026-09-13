# Experiment 06 — Cross-model replication results

## G06 status

`PASS`

The selected non-Qwen model was `microsoft/Phi-4-mini-instruct` at pinned
revision `cfbefacb99257ffa30c83adab238a50856ac3083`. Attempt 0 failed before
semantic generation because the remote `modeling_phi3.py` imported unavailable
`LossKwargs`. The native Transformers recovery (`trust_remote_code=false`)
loaded the same weights and completed a non-benchmark smoke. The original
failure remains at
`results/experiment_06/capability_gap/technical_failure.json`.

## Capability-gap screen

The frozen 48-case screen was evaluated exactly once after recovery. The
screen fingerprint is
`677d6e569c09031c2256161e01dcf48747794678c371948564db97b3e1328505`.

The untouched Phi model scored 0/48 diagnosis exact, 0/48 resolution exact,
0.0 failure-mode macro F1, and 0/48 strict JSON. The predeclared threshold
therefore produced `CAPABILITY_GAP_PRESENT`; training was justified. This
screen was not used as final evidence.

## Controlled training

Training used the frozen E04-selected methodology: 600 examples from the
25% subset, the unchanged 288-case validation corpus, AdamW at `1e-4`, LoRA
rank 8/alpha 32, NF4/BF16 double quantization, and validation-only checkpoint
selection. Native Phi's fused projections were used explicitly:
`qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj`.

Training stopped at 125 optimizer steps under the frozen early-stopping
policy. The selected checkpoint was step 125. Reloaded validation scored:

| Metric | Result |
| --- | ---: |
| diagnosis exact | 288/288 (100.0%) |
| resolution exact | 288/288 (100.0%) |
| culprit accuracy | 288/288 (100.0%) |
| failure-mode macro F1 | 1.000 |
| recommended action | 288/288 (100.0%) |
| evidence F1 | 1.000 |
| strict JSON | 288/288 (100.0%) |

The training evidence bundle passed offline verification. Adapter weights are
local-only; the selected checkpoint location and hashes are retained in the
local run evidence under `runs/experiment_06/training`.

## Fresh final challenge

The separate final benchmark contains 60 cases with fingerprint
`3811b2d5698a8a6530abbe550be0df7d83c4062098326f8b4384369681cda714`.
It was frozen before evaluation. Exact, normalized, and incident-ID overlap
with the audited sources were all zero. Canonical structural overlap was
reported as taxonomy-shape reuse, not packet contamination.

Each system was semantically evaluated exactly once after fresh reload. The
base model produced 0/60 diagnosis exact, 0/60 resolution exact, 0.0
failure-mode macro F1, 0/60 strict JSON, and 0/60 valid JSON. The tuned model
produced 38/60 diagnosis exact (63.3%), 38/60 resolution exact (63.3%), 0.713
failure-mode macro F1, 39/60 strict JSON (65.0%), and 60/60 valid JSON.
Slice diagnosis exact for tuned Phi was 58.3% standard, 62.5% hard, and
75.0% transfer.

The base-to-tuned diagnosis transitions were:

```text
base wrong -> tuned correct   38
base correct -> tuned wrong    0
persistent correct             0
persistent wrong              22
```

Raw predictions, offline reproductions, transition analysis, and artifact
hashes are under
`results/experiment_06/final_evaluation-retry-01/`. The initial aggregation
attempt failed after both semantic evaluations due to a wrapper bug; it is
retained as a technical failure, and finalization recovered entirely offline
without regenerating either semantic evaluation.

## Interpretation and limitation

Under this synthetic, taxonomy-aligned held-out challenge, the E04-style
recipe transferred to Phi after a large capability gap and materially improved
the untouched model's output quality. This is cross-model replication evidence,
not a claim of real-world production accuracy or universal model-family
behavior. The final challenge shares the taxonomy and structural shapes with
earlier studies, as disclosed by its contamination report.
