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

## Experiment 03 — Causal Telemetry Incident Specialization

### 03A — Frozen experimental contract — complete / contract frozen

The typed ontology, canonical scenario contract, telemetry and diagnostic output
schemas, grouped split protocol, provenance fingerprints, and fail-closed audit
foundation are implemented. No dataset, provider rendering, model evaluation,
or training evidence exists yet. Experiment 02 is not reused as an Experiment
03 selection or test set.

### 03B — Deterministic causal scenario engine — complete / supporting infrastructure

The typed deterministic engine now generates authoritative canonical scenarios
from a bounded declarative archetype catalog, validates causal state and
evidence, constructs controlled counterfactual pairs, and writes a small
pre-render reference corpus with grouped split metadata. This milestone has no
LLM-rendered telemetry, model evaluation, training, or model-performance
evidence. See `docs/experiment-03/03b-causal-scenario-engine.md`.

### Renderer Prototype R0 — implemented / live provider not run

The provider-neutral RenderPlan, bounded renderer families, opt-in
OpenAI-compatible adapter, fake-provider dry-run, leakage/semantic validation,
retry policy, and raw/accepted/rejected artifacts are retained as optional
controlled-augmentation infrastructure. No live provider was run and no model
performance evidence exists. R0 is not the current scientific critical path.

### 03C — Cloud-OpsBench adoption and corpus audit — complete

Cloud-OpsBench is the selected primary empirical fault-injection corpus. This
milestone adds a pinned source registry, read-only census, native taxonomy
preservation/mapping report, modality and golden-trajectory audit, context-size
audit, leakage policy, and candidate split feasibility analysis. It does not
vendor the dataset, freeze the final split, train a model, or call a provider.
RCAEval is reserved as a candidate independent cross-dataset benchmark for
03H and is not ingested into training. The pinned real corpus audit observed
754/754 cases and matched the documented 550/204 system counts and 57 native
fault types. See `docs/experiment-03/03c-cloudopsbench-adoption.md`.

### 03D — Training contract, evidence packaging, and protected split freeze — complete

The primary formulation is now a bounded hybrid diagnostic interaction rather
than a flattened raw snapshot. 03D froze the native target contract,
provider-free tool replay, deterministic packaging policy, duplicate/
contamination grouping, and a protected 510/123/121 case-level split for the
pinned 754-case corpus. No model was loaded, selected, trained, or evaluated.
See `docs/experiment-03/03d-training-contract-and-split-freeze.md`.

#### 03D.1 — Tool-replay parity and cumulative-context integrity gate — pass

The representation gate classified all 1,007 historical fallbacks, retained
zero unresolved observations, excluded golden-only paths from executable
primary SFT, and froze representation fingerprint
`f77e8133ef216632a61d9b6200bda1a269967336fe6b2ae4972d8685005a5fbf`. The
source split remains unchanged. See
`docs/experiment-03/03d1-tool-replay-integrity-gate.md`.

### Planned sequence

- 03E — Untouched base-model capability-gap screening
- 03F — QLoRA specialization
- 03G — Controlled data/training ablations
- 03H — Protected in-domain evaluation plus external cross-dataset generalization

Model selection belongs to the capability-gap phase. No model is selected for
Experiment 03 yet; selection belongs to 03E after the dataset is frozen.

## Focused optimization — later

Only after Experiment 02 exists: learning-rate sensitivity, LoRA rank/capacity, target-module efficiency where justified, VRAM/runtime/quality trade-offs, and a fresh untouched final benchmark.

This is deliberately a small laboratory roadmap, not a platform or orchestration plan.
