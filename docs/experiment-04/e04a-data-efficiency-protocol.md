# Experiment 04A — Data-efficiency protocol

## Status

Protocol frozen before semantic training. The study asks how much of the
specialist training corpus is required to preserve specialization capability.
The primary intervention is training-data fraction only:

| Variant | Training examples |
| --- | ---: |
| 25% | 600 |
| 50% | 1200 |
| 75% | 1800 |
| 100% | 2400 |

The counts are derived from the canonical 2400-example corpus and are asserted
by the subset manifests, not assumed by this document.

## Frozen controls

All variants use the pinned Qwen3-4B revision from the existing local model
registry, the same validation corpus, seed `20260941`, NF4 with BF16 compute and
double quantization, LoRA rank 16 / alpha 32 / dropout 0, the seven existing
projection targets, AdamW at `2e-4`, microbatch 1, gradient accumulation 8,
sequence length 768, assistant-only supervision, deterministic preprocessing,
and the existing incident prompt/evaluation contract.

Validation is the only checkpoint-selection boundary. Evaluation benchmark
outputs are reporting artifacts only and cannot select a variant, checkpoint,
or stopping decision. The E03 blind benchmark is sealed and excluded from every
E04-A training decision.

## Subsets

Subsets are nested and selected within failure family using a stable SHA-256
rank over `(seed, family, incident_id)`, then serialized in incident-ID order.
Difficulty, topology-family, and red-herring distributions are persisted in
each `subset_manifest.json`. A subset's identity is its source fingerprint,
fraction, seed, and selected records; filesystem paths are not identity inputs.

Artifacts:

```text
data/experiment_04a/subsets/
├── study_manifest.json
├── validation/
└── {025,050,075,100}pct/
    ├── train.jsonl
    ├── ground_truth_train.jsonl
    └── subset_manifest.json
```

## Selection rule

After all four semantic runs verify, select the smallest fraction satisfying,
relative to the best validation variant:

* diagnosis exact within 1.0 percentage point;
* resolution exact within 1.0 percentage point;
* failure-mode macro F1 within 1.0 percentage point;
* no failure family more than 5 percentage points below the reference;
* no critical schema-validity regression.

If no reduced fraction qualifies, select 100%. Training cost is only a
tie-breaker after quality eligibility. Negative and dominated variants remain
persisted.

## Isolation statement

E03 is not read by the E04-A selection code. The frozen benchmark fingerprint
is retained as a contract boundary, but benchmark metrics are not an input to
training, checkpoint selection, stopping, or variant selection.
