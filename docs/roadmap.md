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

### Experiment 02B.1 — Training foundation — complete

The independent, balanced train/validation data, contamination checks,
assistant-only formatting, validation-only checkpoint policy, early stopping,
and resolved run manifest were used unchanged by the specialization run.

### Experiment 02B.2 — QLoRA specialization — complete

One controlled QLoRA run completed with Qwen/Qwen3-4B. It stopped at 100 of
600 configured optimizer steps, selected checkpoint 100 using validation only,
and achieved 99.31% (143/144) frozen diagnosis exact match versus the 65.28%
base result. STANDARD/HARD/TRANSFER diagnosis exact was 100.00%/97.92%/100.00%.
The frozen benchmark was generated once after fresh adapter reload. A narrow
post-processing aggregate-split bug was recovered offline from the persisted
144 raw predictions; no model generation was repeated.

The planned process is:

1. Shortlist candidate specialist domains.
2. Build small frozen base challenge sets.
3. Measure capability gaps on untouched Qwen3-4B.
4. Reject tasks where the base is already too strong.
5. Select one difficult specialist domain.
6. Construct train/validation/test data.
7. Fine-tune and measure exact base → tuned impact.

Experiment 02 is now a completed Production Incident Diagnosis specialization.
02A capability-gap measurement, 02B.1 training foundation, and 02B.2 QLoRA
specialization are complete.

## Focused optimization — later

Only after Experiment 02 exists: learning-rate sensitivity, LoRA rank/capacity, target-module efficiency where justified, VRAM/runtime/quality trade-offs, and a fresh untouched final benchmark.

This is deliberately a small laboratory roadmap, not a platform or orchestration plan.
