# Experiment 06 — Final held-out benchmark protocol

Status: `FROZEN_BEFORE_MODEL_EVALUATION`

The capability-gap screen is a pre-training decision instrument and is not
used as final evidence. This protocol freezes a separate 60-case challenge at
`data/incident_diagnosis_e06_final`, generated with a new seed (`20260914`)
under the independent blind namespace. Taxonomy and scorer reuse are
disclosed; packets, case order, and benchmark fingerprint are new.

The freeze record is `results/experiment_06/final_protocol.json`, with the
contamination report beside it. Exact and normalized packet/id overlap must
be zero. Canonical structural signatures are reported separately because the
taxonomy intentionally reuses structural shapes; this does not constitute
packet contamination.

Before any aggregate metric is inspected, the benchmark inputs, truths,
prompt hash, scorer fingerprint, decoding, model revision, and system list
are frozen. The systems are evaluated exactly once each after a fresh reload:

```text
base  = untouched microsoft/Phi-4-mini-instruct
tuned = selected validation checkpoint from the E06 training run
```

Both use native Transformers loading with `trust_remote_code=false`, the same
prompt, scorer, and deterministic decoding. Raw outputs are persisted before
scoring; offline reproduction and artifact hashes are mandatory. No
checkpoint switching or post-result prompt change is allowed.
