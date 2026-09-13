# Experiment 05 — fresh blind generalization protocol

## G05A freeze

This benchmark is frozen before any model evaluation. It contains 120 cases:
96 independently generated cases from the new E05 generator namespace and 24
static-authored, generator-independent fixtures. “Static-authored” describes
the fixture source format; it does not claim human authorship.

The taxonomy and strict JSON scorer are reused from CauseTune’s incident
diagnosis contract, which is disclosed. E05 introduces a separate benchmark
namespace, new narrative templates, new case ordering, eight new topology
compositions, changed chronology patterns, different wording distributions,
multi-signal distractors, temporal coincidence, negative and partial evidence
patterns, and rare family/topology combinations.

The frozen benchmark manifest records the generator versions, seed, split and
origin counts, benchmark fingerprint, and contamination audit. The audit
checks exact packet overlap, normalized packet overlap, incident-ID overlap,
and canonical structural signatures against the E04 train/validation inputs,
the E02 benchmark, and the historical E03 blind benchmark.

## Evaluation contract

The benchmark inputs, ground truths, scorer, prompt, and deterministic decode
parameters are fixed in the frozen manifest and evaluation protocol. Exactly
one fresh semantic evaluation is run for each of:

1. untouched base Qwen3-4B;
2. the original selected E02 adapter;
3. the final E04-selected adapter.

Raw predictions are persisted before aggregate metrics. No checkpoint
switching, prompt adjustment, result-driven regeneration, or E04 selection is
allowed. The result is reported as a fresh frozen synthetic blind challenge,
not as real-world OOD or production accuracy.
