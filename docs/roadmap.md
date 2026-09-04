# CauseTune Roadmap

## Laboratory foundation — completed

- QLoRA vertical slice
- deterministic preprocessing
- assistant-only masking
- frozen evaluation contract
- training telemetry
- failure analysis
- causal shuffle experiment

## M7 — Training efficiency — next

- checkpoint persistence
- validation-based checkpoint selection
- early stopping
- optimizer-step savings
- training-time savings

M7 must select checkpoints using validation only. ID, HARD, and OOD remain sealed for final evaluation.

## Experiment 02 — Production Incident Diagnosis Specialist

### Experiment 02A — Capability-gap baseline — in progress

Freeze the incident benchmark and evaluation contract, then measure untouched Qwen3-4B once. The current phase includes no training data, adapter, or QLoRA run.

### Experiment 02B — Training dataset + QLoRA specialization — blocked

Blocked until the 02A baseline and capability-gap review are complete.

The planned process is:

1. Shortlist candidate specialist domains.
2. Build small frozen base challenge sets.
3. Measure capability gaps on untouched Qwen3-4B.
4. Reject tasks where the base is already too strong.
5. Select one difficult specialist domain.
6. Construct train/validation/test data.
7. Fine-tune and measure exact base → tuned impact.

Experiment 02 is now selected as Production Incident Diagnosis Specialist. Only 02A is in progress; 02B has not started.

## Focused optimization — later

Only after Experiment 02 exists: learning-rate sensitivity, LoRA rank/capacity, target-module efficiency where justified, VRAM/runtime/quality trade-offs, and a fresh untouched final benchmark.

This is deliberately a small laboratory roadmap, not a platform or orchestration plan.
