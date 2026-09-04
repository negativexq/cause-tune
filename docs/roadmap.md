# CauseTune Roadmap

## Laboratory foundation — completed

- QLoRA vertical slice
- deterministic preprocessing
- assistant-only masking
- frozen evaluation contract
- training telemetry
- failure analysis
- causal shuffle experiment

## M7 — Training efficiency — incorporated as laboratory infrastructure

- checkpoint persistence
- validation-based checkpoint selection
- early stopping
- optimizer-step savings
- training-time savings

M7 must select checkpoints using validation only. ID, HARD, and OOD remain sealed for final evaluation.

## Experiment 02 — Production Incident Diagnosis Specialist

### Experiment 02A — Capability-gap baseline — complete

The incident benchmark and evaluation contract were frozen, then untouched Qwen3-4B was measured once. The current result is recorded in `results/incident_diagnosis_base.json`; no training data, adapter, or QLoRA run was used.

### Experiment 02B.1 — Training foundation — current

The independent, balanced train/validation data, contamination checks,
assistant-only formatting, validation-only checkpoint policy, early stopping,
and resolved run manifest are ready. No GPU training or adapter has been run.

### Experiment 02B.2 — QLoRA specialization — next

Blocked pending review of the 02B.1 foundation. It will use one controlled run
and keep the frozen 02A benchmark sealed until checkpoint selection.

The planned process is:

1. Shortlist candidate specialist domains.
2. Build small frozen base challenge sets.
3. Measure capability gaps on untouched Qwen3-4B.
4. Reject tasks where the base is already too strong.
5. Select one difficult specialist domain.
6. Construct train/validation/test data.
7. Fine-tune and measure exact base → tuned impact.

Experiment 02 is now selected as Production Incident Diagnosis Specialist.
02A capability gap is complete; 02B.1 is current and 02B.2 has not started.

## Focused optimization — later

Only after Experiment 02 exists: learning-rate sensitivity, LoRA rank/capacity, target-module efficiency where justified, VRAM/runtime/quality trade-offs, and a fresh untouched final benchmark.

This is deliberately a small laboratory roadmap, not a platform or orchestration plan.
