# CauseTune

## Controlled LLM Post-Training Experiments

**Measure the gap. Fine-tune. Explain the gain. Prove the run.**

CauseTune is a controlled LLM post-training laboratory for measuring capability
gaps, running reproducible QLoRA specialization experiments, analyzing training
dynamics, and testing whether gains survive fresh held-out evaluation. It is a
research workflow with frozen contracts, persisted evidence, and offline
verification—not a serving platform or a collection of generic recipes.

## v1.0 Results

CauseTune v1.0 completed controlled optimization, fresh blind generalization,
cross-model replication, failure-boundary, and quality/cost studies. The
complete release audit is in the [v1.0 readiness report](docs/release-v1.0-readiness.md)
and its [machine-readable evidence manifest](results/release/v1.0-readiness-audit.json).

### E04 selected recipe

Under the predeclared validation-only E04 study, the selected Qwen3-4B recipe
was:

| Control | Selected value |
| --- | --- |
| Model | `Qwen/Qwen3-4B` |
| Training data | 25% / 600 examples |
| LoRA | rank 8, alpha 32 |
| Learning rate | `1e-4` |
| Selected checkpoint | step 100 |
| Validation diagnosis / resolution | 100% / 100% |
| Validation failure-mode macro F1 | 1.0 |
| Validation strict JSON | 100% |

This is validation saturation for the frozen task. It did not prove that the
smaller recipe preserved fresh generalization.

### E05 fresh blind generalization

The independent 120-case synthetic, taxonomy-aligned blind challenge exposed
the most important optimization result:

| System | Diagnosis exact | Resolution exact | Failure-mode macro F1 | Strict JSON |
| --- | ---: | ---: | ---: | ---: |
| Untouched Qwen3-4B base | 66.67% | 28.33% | 73.95% | 91.67% |
| Original E02 adapter | 98.33% | 98.33% | 98.73% | 99.17% |
| E04 selected adapter | 92.50% | 92.50% | 94.09% | 95.83% |

E04 reduced the training footprint while preserving saturated validation
performance, but the fresh E05 blind challenge exposed a **5.83 percentage
point diagnosis regression versus E02**. The result is preserved as evidence;
E04 was not retroactively changed.

### E06 cross-model replication

`microsoft/Phi-4-mini-instruct` showed a capability gap and was trained under
the controlled methodology with a separate final held-out benchmark. The tuned
model reached 38/60 diagnosis exact (63.33%), 38/60 resolution exact (63.33%),
failure-mode macro F1 71.30%, and 100% valid JSON. The untouched Phi baseline
scored 0% diagnosis, 0% resolution, and 0% strict JSON compliance on that
contract. This is limited replication evidence from **one additional model
family**, not a production-accuracy claim.

### E07 and E08 conclusions

E07 tested sufficient, insufficient, contradictory, ambiguous, missing-evidence,
and out-of-taxonomy cases. The E04 adapter achieved 100% sufficient-case
accuracy, but retained a non-zero false-confident diagnosis rate: 9/48 (18.75%)
at the failure boundary.

E08 classified E04 as a **lower-cost negative trade-off**. It reduced the
unique training corpus and wall-clock cost, but was not quality-preserving once
the E05 blind regression was included.

## Experiment Results

| Experiment | Question | Result |
| --- | --- | --- |
| E04-A | How much data is needed? | Validation saturated at 25% / 600 examples |
| E04-B | How much LoRA capacity is needed? | Rank 8 selected under the frozen validation rule |
| E04-C | Which learning rate works under the selected recipe? | `1e-4` selected |
| E05 | Does the optimized recipe survive a fresh blind set? | No — E04 trailed E02 by 5.83 pp diagnosis |
| E06 | Does specialization reproduce on another model family? | Limited replication on Phi-4-mini |
| E07 | What happens under insufficient or ambiguous evidence? | Non-zero false confidence remained |
| E08 | Was the cheaper recipe quality-preserving? | No — negative quality/cost trade-off |

## What CauseTune Tests

CauseTune measures specialization as a controlled sequence rather than as a
single fine-tuning score:

1. Measure the untouched model's capability gap.
2. Freeze the intervention, data roles, model revision, and evaluation rules.
3. Select checkpoints using validation only.
4. Persist raw predictions, provenance, and artifact hashes.
5. Reproduce metrics offline from the evidence bundle.
6. Freeze a fresh blind challenge and measure gains and regressions.
7. Test cross-model behavior, failure boundaries, and training cost.

The project does not provide model serving, routing, gateway infrastructure,
Web UI, MCP, cloud orchestration, automatic hyperparameter search, or a generic
RLHF/trainer catalogue.

## Experimental loop

```text
TASK CANDIDATE
    ↓
FROZEN CHALLENGE BENCHMARK
    ↓
UNTOUCHED BASE MODEL → MEASURE CAPABILITY GAP
    ↓
FINE-TUNE → VALIDATION-ONLY CHECKPOINT SELECTION / EARLY STOPPING
    ↓
IMMUTABLE EVIDENCE BUNDLE → OFFLINE VERIFICATION
    ↓
FRESH FROZEN BLIND EVALUATION → BASE vs TUNED
    ↓
REGRESSION + EFFICIENCY + FAILURE-BOUNDARY ANALYSIS
```

Each experiment asks three questions:

- What could the untouched model do?
- What capability did specialization add, and did it generalize?
- How much training, memory, and adapter capacity did it require?

## Historical E02 measured result

The compact table below reports **diagnosis exact match** on the frozen
Experiment 02 benchmark.

| Experiment | Base | Tuned | Gain | HARD | TRANSFER | Stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production Incident Diagnosis | 65.28% | **99.31%** | **+34.03 pp** | 97.92% | 100.00% | 100 / 600 steps |

This is a synthetic 144-case held-out benchmark, not real-world production
accuracy. The benchmark was excluded from training and checkpoint selection;
an independent validation set selected the adapter checkpoint, followed by a
fresh base-plus-adapter reload and one tuned benchmark generation.

## Experiment 02 — Production Incident Diagnosis

Experiment 02A first froze the benchmark and measured the untouched
Qwen/Qwen3-4B capability gap. Experiment 02B then built independent
train/validation data and executed one controlled QLoRA specialization run.

### Base capability gap

| Slice | Diagnosis exact match | Resolution exact match | Strict JSON |
| --- | ---: | ---: | ---: |
| STANDARD | 66.67% (48/72) | 37.50% (27/72) | 91.67% |
| HARD | 64.58% (31/48) | 35.42% (17/48) | 89.58% |
| TRANSFER | 62.50% (15/24) | 25.00% (6/24) | 79.17% |

### Base vs tuned

| Metric | Base | Tuned | Delta |
| --- | ---: | ---: | ---: |
| Diagnosis exact | 65.28% (94/144) | **99.31% (143/144)** | **+34.03 pp** |
| Resolution exact | 34.72% | **99.31% (143/144)** | **+64.58 pp** |
| Culprit accuracy | 77.78% | **100.00%** | **+22.22 pp** |
| Failure-mode accuracy | 71.53% | **99.31%** | **+27.78 pp** |
| Failure-mode macro F1 | 71.84% | **99.30%** | **+27.47 pp** |
| Action accuracy | 40.97% | **100.00%** | **+59.03 pp** |
| Evidence F1 | 80.88% | **100.00%** | **+19.12 pp** |
| Strict JSON | 88.89% | **100.00%** | **+11.11 pp** |
| Valid JSON | 100.00% | **100.00%** | 0.00 pp |

### Generalization by benchmark slice

| Slice | Base | Tuned | Delta |
| --- | ---: | ---: | ---: |
| STANDARD | 66.67% (48/72) | **100.00% (72/72)** | **+33.33 pp** |
| HARD | 64.58% (31/48) | **97.92% (47/48)** | **+33.33 pp** |
| TRANSFER | 62.50% (15/24) | **100.00% (24/24)** | **+37.50 pp** |

TRANSFER is a transfer-style held-out slice; it is not claimed to be true OOD.

### Validation progression

| Step | Diagnosis | Resolution | Macro F1 | Action | Evidence F1 | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 58.33% | 25.35% | 59.04% | 28.82% | 66.04% | 1.47637 |
| 25 | **100.00%** | 97.92% | **100.00%** | 97.92% | **100.00%** | 0.001167 |
| 50 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000369 |
| 75 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000200 |
| 100 | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000144 |

Diagnosis reached the validation ceiling by step 25. Checkpoint selection
remained validation-only; resolution and teacher-forced loss continued to
improve, so the tie-breakers preferred later checkpoints. Early stopping
terminated the run at step 100 instead of consuming the full 600-step budget.
Validation performance does not guarantee frozen-benchmark performance.

### Training efficiency

| Measure | Result |
| --- | ---: |
| Maximum optimizer-step budget | 600 |
| Actual stop | **100** |
| Stop reason | `validation_no_improvement` |
| Best checkpoint | **step 100** |
| Earliest near-best checkpoint | **step 25** |
| Updates avoided | **500 / 600** |
| Maximum budget avoided | **83.33%** |
| Near-best potential saving | **575 / 600 (95.83%)** |
| Peak allocated VRAM | **5.312 GiB** |
| Trainable parameters | **33,030,144 (~0.814%)** |
| Logical model parameters | 4,055,498,240 |
| Training wall time | 10,263.71 s |
| Validation wall time | 4,455.54 s |

The run used an 8 GB RTX 5070 Laptop GPU. Peak allocated VRAM is the useful
headline memory measure here; framework reserved-memory accounting is not used
as a physical-VRAM claim.

### What actually changed?

| Diagnosis transition | Cases |
| --- | ---: |
| Base wrong → tuned correct | **49** |
| Base correct → tuned wrong | 1 |
| Persistent correct | 94 |
| Persistent wrong | 0 |

| Failure behavior | Base | Tuned |
| --- | ---: | ---: |
| Recent-change / deploy bias | 12 | **0** |
| HARD distractor selection | 17 | **1** |
| Correct culprit / wrong family | 18 | **1** |
| Correct family / wrong culprit | 9 | **0** |
| Correct diagnosis / wrong action | 44 | **0** |
| Invalid culprit values | 13 | **0** |
| Invalid evidence references | 3 | **0** |
| Strict-schema failures | 16 | **0** |

The aggregate gain corresponds to specific mechanically observed failure modes
disappearing, not only to a change in one headline score.

### Failure families

| Failure family | Base | Tuned | Delta |
| --- | ---: | ---: | ---: |
| db_connection_pool_exhaustion | 58.33% | 100.00% | +41.67 pp |
| db_query_regression | 75.00% | 100.00% | +25.00 pp |
| memory_leak | 100.00% | 100.00% | 0.00 pp |
| downstream_dependency_timeout | 33.33% | 91.67% | +58.33 pp |
| cache_stampede | 100.00% | 100.00% | 0.00 pp |
| kafka_consumer_lag | 58.33% | 100.00% | +41.67 pp |
| thread_pool_exhaustion | 83.33% | 100.00% | +16.67 pp |
| disk_io_saturation | 25.00% | 100.00% | +75.00 pp |
| dns_resolution_failure | 100.00% | 100.00% | 0.00 pp |
| tls_certificate_expiration | 91.67% | 100.00% | +8.33 pp |
| rate_limit_misconfiguration | 58.33% | 100.00% | +41.67 pp |
| configuration_regression | 0.00% | 100.00% | +100.00 pp |

The largest gains were `configuration_regression` (0% → 100%),
`disk_io_saturation` (25% → 100%), and `downstream_dependency_timeout`
(33.33% → 91.67%). The only remaining diagnosis error was one HARD downstream
timeout case predicted as DNS resolution failure. There was no family-level
diagnosis regression. Weak frozen families were not oversampled after observing
the base benchmark.

### QLoRA configuration

- Model: `Qwen/Qwen3-4B`
- NF4 4-bit; BF16 compute; double quantization
- LoRA rank `16`; alpha `32`; dropout `0`
- Targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Maximum sequence length: `768`
- Microbatch `1`; gradient accumulation `8`; effective batch `8`
- Learning rate `2e-4`; maximum epochs `2`; maximum optimizer steps `600`
- Actual stop: `100`; assistant-only supervision; `enable_thinking=False`
- Gradient checkpointing; deterministic seeded training order

Logical parameters: `4,055,498,240`
Trainable parameters: `33,030,144 (~0.814453%)`

### Integrity

- Frozen benchmark and evaluation contract were established before training.
- Train/validation data were independent from the benchmark.
- No benchmark-informed oversampling was used.
- Checkpoint selection used validation only.
- One semantic training run executed; one fresh base-plus-selected-adapter reload passed.
- The single tuned benchmark generation completed successfully; a deterministic aggregate-analysis bug was fixed afterward and metrics were recomputed offline from the persisted 144 predictions without regenerating model outputs.
- No LLM judge, manual output repair, or alternate checkpoint evaluation was used; malformed outputs were scored as produced.

## Experiment 01 — Controlled QLoRA Specialization

Experiment 01 used a deliberately narrow customer-support intent-classification
task to validate the laboratory workflow and learn fine-tuning mechanics. It is
separate from the production-incident specialization in Experiment 02.

The verified values below are accuracy metrics:

| Experiment | Validation | ID | HARD | OOD |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-4B | 88.0% | 83.6% | 76.8% | 79.6% |
| M5 QLoRA — unshuffled | 26.8% | 24.8% | 23.6% | 20.0% |
| M6 QLoRA — deterministic shuffle | 99.2% | 97.2% | 96.0% | 92.0% |

M5 used `shuffle=False` over 10 class-contiguous blocks of 200 examples. With
microbatch `1` and gradient accumulation `8`, all 250/250 effective optimizer
windows were single-class. The final 25 optimizer updates contained only
`wrong_item`, and the model catastrophically recentered on that terminal class.

M6 changed exactly one training-affecting variable: the order became a
deterministic seeded shuffle with seed `42`. All 250 windows became mixed-class,
the terminal block disappeared, and held-out performance recovered. The causal
lesson is that train loss alone is insufficient evidence; a balanced dataset
can still produce pathological optimizer windows.

### M6 validation progression

| Step | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 88.4% | 99.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% | 99.2% |

### Experiment 01 failure analysis

| Expected intent → `wrong_item` | M5 | M6 |
| --- | ---: | ---: |
| `duplicate_charge` | 100 | 0 |
| `fraud_suspected` | 100 | 1 |
| `order_missing` | 100 | 6 |
| `refund` | 99 | 1 |
| `cancel_order` | 98 | 2 |

## Experiment 03A — Causal Telemetry Incident Specialization

**Status: CONTRACT-FROZEN foundation.** Experiment 03A freezes a typed causal
scenario, telemetry, output, split, provenance, audit, and evaluation contract
for a fresh evidence boundary. It contains no generated dataset, provider call,
model evaluation, training run, or new specialist-model result. See the
[Experiment 03A contract](docs/experiment-03/03a-frozen-contract.md).

### Experiment 03B — Deterministic causal scenario engine

**Status: COMPLETE foundation milestone.** 03B adds a deterministic,
declarative engine for authoritative canonical incident scenarios, bounded
topologies and runtimes, causal variables, semantic evidence, distractors,
counterfactual pairs, grouped pre-render split manifests, and fail-closed
audits. Its 52-scenario CPU-safe reference corpus is a fixture for contract
validation—not training data, a benchmark, or model evidence. See the
[03B engine record](docs/experiment-03/03b-causal-scenario-engine.md).
03E is the completed validation-only full-agent model screen; decomposed
diagnosis screening is now the 03G milestone and any future training is
task-specific and separately authorized.

### Renderer Prototype R0 — supporting infrastructure

**Status: IMPLEMENTED / LIVE PROVIDER NOT RUN.** The previous synthetic LLM
telemetry pilot is retained as provider-neutral, offline-safe augmentation
infrastructure. It is not scientific model evidence and is not on the current
Experiment 03 critical path. See the [Renderer Prototype R0 record](docs/experiment-03/03c-llm-telemetry-rendering-pilot.md).

### Experiment 03C — Cloud-OpsBench adoption and corpus audit

**Status: COMPLETE — PINNED REAL CORPUS AUDITED.** Cloud-OpsBench is the
selected primary empirical
fault-injection corpus. The pinned, read-only adapter audits its native cases,
modalities, golden trajectories, target availability, context sizes, leakage
risks, and candidate split strategies without copying the external dataset or
calling a provider. The real audit matched 754 cases, 550 Online Boutique,
204 Train-Ticket, and 57 native fault types. No Experiment 03 model has been
trained or evaluated.
See the [03C adoption and audit record](docs/experiment-03/03c-cloudopsbench-adoption.md).

### Experiment 03D — Training contract and protected split freeze

**Status: COMPLETE — TRAINING CONTRACT AND PROTECTED SPLITS FROZEN.** The
primary formulation is a bounded hybrid diagnostic interaction: sanitized
incident context, offline tool replay, bounded observations, and deterministic
structured targets derived from Cloud-OpsBench metadata. The pinned 754-case
corpus is split by immutable source case into 510 TRAIN, 123 VALIDATION, and
121 protected TEST cases; all 57 native fault types are represented in each
partition. Full raw-snapshot flattening is rejected as the primary input, and
no model has been selected, loaded, trained, or evaluated in 03D. See the
[03D training-contract record](docs/experiment-03/03d-training-contract-and-split-freeze.md).

The 03D.1 integrity gate found 6,384 exact, 633 canonical-equivalent, and 214
source-derived replay observations; 160 golden-only observations are retained
for auxiliary analysis only. The protected split is unchanged, and no model
has been selected or evaluated.

### Experiment 03E / 03E.1 / 03E.2 — Full-agent screening and contract audit

**Status: COMPLETE — full joint task too difficult for untouched 2B.** 03E,
03E.1, and 03E.2 preserved the original evidence, audited the contract, and
rescreened only Qwen3.5-2B under the corrected V2 contract. The final result
is `BASE_TOO_WEAK` / `DO_NOT_START_QLORA`; no TEST predictions were produced
and no training was performed. See the [03E screening record](docs/experiment-03/03e-base-capability-gap-screening.md),
[03E.1 audit artifacts](results/incident_telemetry_03e1/), and the
[03E.2 V2 record](docs/experiment-03/03e2-v2-capability-rescreen.md).

### Experiment 03F — Hierarchical RCA task decomposition

**Status: COMPLETE — decomposition contract frozen.** 03F isolates bounded
category, root-cause, hierarchical, and fault-object subtasks over the frozen
Cloud-OpsBench evidence boundary. The full agent loop is separated from
diagnosis so 03G can measure which capability is learnable on the local
compact-model hardware. See the
[03F decomposition record](docs/experiment-03/03f-hierarchical-rca-task-decomposition.md).

### Experiment 03G — Decomposed untouched-2B capability screening

**Status: COMPLETE — Task B selected as a future specialization candidate.**
The untouched pinned Qwen3.5-2B screen measured category, oracle-category
root-cause, hierarchical, and secondary object subtasks over the immutable
03F.1 evidence representation. Task B showed a non-saturated above-baseline
signal; no training or adapter creation occurred. The historical 03E.2
full-agent conclusion remains `BASE_TOO_WEAK`. See the
[03G screening record](docs/experiment-03/03g-decomposed-2b-capability-screening.md).
03H and its documented local runtime-recovery attempts are preserved as
separate feasibility and technical-failure evidence; they did not produce a
scientific Task-B adapter result.

### Experiment 03H — Task-B QLoRA specialization

**Status: BLOCKED — TRAINING CONTRACT NOT FEASIBLE.** Pre-flight froze the
natural 510-record Task-B training contract, but the 8,192-token assistant-only
QLoRA smoke backward exceeded the local 8 GB RTX 5070 Laptop GPU. The larger
12,288-token candidate also failed its feasibility probe. No primary training,
adapter, tuned prediction, or TEST inference was produced. See the
[03H specialization record](docs/experiment-03/03h-task-b-qlora-specialization.md).

### Experiment 03H.1 — Local training-feasibility recovery

The original 03H 8,192-token QLoRA contract remains an immutable negative
feasibility result. A separate gate audited the actual hybrid Qwen3.5 model,
confirmed the PyTorch DeltaNet fallback, and completed only disposable
TRAIN-only optimizer-step probes. The largest feasible documented budget is
4,096 tokens. No primary QLoRA run, adapter, tuned validation result, or TEST
inference was produced by 03H.1. The first 03H.2 primary attempt then failed
before optimizer step 1 with `CUBLAS_STATUS_INTERNAL_ERROR` in the fallback
DeltaNet path; no tuned quality claim is made. 03H.3 then tested runtime
stability at the fixed 4,096/3,072/2,048 ladder and rejected the local Task-B
training contract after allocator/runtime failures; no primary run was
started. See the
[03H.1 feasibility record](docs/experiment-03/03h1-local-training-feasibility-recovery.md).

### Experiment 03I — Qwen3-4B replacement-student qualification

The preselected Qwen3-4B produced a valid above-baseline Task-B base signal
(24/57 exact, 57/57 schema and enum valid). A follow-up harness audit found
that the original forward gate did not match the training contract; the
corrected 4,096-token local runtime contract is qualified. No primary QLoRA
run or persistent adapter was created. See the
[03I forward-harness audit](docs/experiment-03/03i-forward-harness-audit.md).

### Experiment 03J — Qwen3-4B Task-B QLoRA specialization

The first primary QLoRA run was blocked by a CUDA allocator OOM on the first
training-form step, before optimizer step 1. The separate 4096-token smoke
passed, but no scientific adapter, tuned development result, or quality
conclusion exists. This is a runtime failure rather than a negative
specialization result. TEST remains sealed. See
[03J documentation](docs/experiment-03/03j-qwen3-4b-task-b-qlora-specialization.md).

### Experiment 03J.1 — Primary-training execution-parity audit

The exact first 03J record was identical to the corrected 03I record, and both
direct one-step paths passed with matching tensors, flags, optimizer settings,
and pre-forward memory. However, the actual primary-style 4096 stress path
failed with allocator OOM/hang on the first stress record (`0/15`). No
deterministic parity defect was isolated, no repair or adapter was retained,
and no quality evaluation was run. Final decision:
`PRIMARY_PATH_4096_NOT_FEASIBLE`. TEST remains sealed. See the
[03J.1 audit](docs/experiment-03/03j1-primary-training-parity-audit.md).

### 03J.2 — Primary-path sequence recovery — complete / local path not feasible

The pre-registered 3072-token primary-path candidate passed one disposable
step but failed the required 15-case stress gate with allocator OOM/hang. The
pre-registered 2048-token fallback also passed one step but failed the same
stress gate. No scientific training or quality evaluation occurred. Final
status: `LOCAL_PRIMARY_TRAINING_NOT_FEASIBLE`. TEST remains sealed. See the
[03J.2 record](docs/experiment-03/03j2-primary-path-sequence-recovery.md).

### 03K — Sustainable local QLoRA runtime recovery — complete / recovery failed

03K kept the 3072-token Task-B contract and tested the frozen runtime ladder.
The minimal loop, native allocator policy, and real paged 8-bit AdamW each
failed the required sustained stress gate; SDPA was already active. No
scientific training or quality evaluation occurred. Final status:
`RUNTIME_RECOVERY_FAILED`. See the
[03K record](docs/experiment-03/03k-sustainable-local-qlora-runtime-recovery.md).

## Core capabilities

- Versioned experiment contracts and strict configuration validation.
- Deterministic preprocessing, resolved-config hashing, and dataset
  fingerprints.
- Model and revision pinning with doctor/preflight checks.
- QLoRA training with validation-only checkpoint selection and early stopping.
- Immutable evidence bundles, raw predictions, artifact hashes, and tamper
  detection.
- Offline metric reproduction and fresh blind-benchmark freezing.
- Cross-model experiments, failure-boundary evaluation, and quality/cost
  analysis.
- Model-backed CI smoke coverage for the real Transformers Trainer and PEFT
  path.

The measured outputs include task-specific accuracy/F1, slice behavior,
per-family metrics, confusion pairs, failure transitions, finite-loss status,
optimizer steps, validation progression, wall-clock time, token throughput,
peak allocated VRAM, and trainable parameter count.

## Methodology and workflow

The core evidence path is:

```text
Experiment Contract
        │
        ▼
Doctor / Preflight
        │
        ▼
Dataset + Model Fingerprints
        │
        ▼
QLoRA Training
        │
        ▼
Validation-only Checkpoint Selection
        │
        ▼
Immutable Evidence Bundle
        │
        ▼
Offline Verification
        │
        ▼
Frozen Blind Evaluation
        │
        ▼
Regression / Cost Analysis
```

Each valid run records:

- the resolved experiment configuration;
- model ID and pinned revision;
- training, validation, and benchmark fingerprints;
- checkpoint-selection provenance;
- evaluation artifacts and raw predictions;
- artifact hashes and offline-verification state.

## CLI and quick start

The installed CLI exposes only the implemented laboratory workflows:

```bash
causetune --help
causetune doctor --config <config> [--hardware]
causetune prepare --config <config> --run-dir <run-directory>
causetune train --config <config> --run-dir <run-directory>
causetune evaluate --predictions <predictions> --output <evaluation>
causetune compare --base <base-evaluation> --tuned <tuned-evaluation>
causetune verify <run-directory> --offline
```

Training and hardware operations require the relevant experiment configuration
and runtime dependencies. `verify --offline` checks a persisted evidence bundle
without launching model inference.

## Repository layout

```text
configs/          experiment and evaluation configuration
data/             frozen benchmark and training inputs
docs/             experiment record, methodology, and roadmap
results/          concise verified public evidence
scripts/          dataset, audit, training, recovery, and evaluation entrypoints
src/causetune/    reusable package code
tests/            CPU-safe regression tests
README.md
pyproject.toml
```

## Quick verification

These commands do not launch GPU training:

```bash
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pytest -q
git diff --check
```

## CI

Release CI includes two required jobs:

- `cpu-safe`: compilation and the CPU-safe regression suite;
- `training-smoke`: a real Transformers `Trainer` + PEFT LoRA smoke that
  checks optimizer updates, finite loss, adapter saving, fresh reload, and
  generation. It is a runtime smoke, not a full production training run.

## Scope boundary

CauseTune covers offline specialization and evidence generation: preprocessing,
LoRA/QLoRA, quantization, validation, checkpointing, early stopping,
telemetry, VRAM profiling, throughput, diagnostics, frozen evaluation, and
base-vs-tuned measurement.

It does not provide model serving, routing, gateway infrastructure, a Web UI,
MCP, cloud orchestration, automatic hyperparameter search, a recipe zoo, or a
generic RLHF/trainer catalogue. Production promotion and operational control
belong elsewhere.

## Limitations

- Synthetic benchmark construction and taxonomy alignment.
- Limited model-family coverage despite controlled replication on one additional
  non-Qwen family.
- One-shot blind challenges and static/manual fixture limitations where
  disclosed.
- Hardware-specific efficiency measurements.
- Adapter weights are local-only; lightweight hashes, manifests, metrics, and
  locations are persisted.
- Historical E02 and E04 results are bounded by the specific Qwen3-4B task and
  frozen evidence contracts; they are not production accuracy claims.

## EX-LV status

The optional low-VRAM/layer-streaming comparison was not executed:
`EX-LV: NOT_RUN`. Gate infrastructure may exist, but CauseTune does not claim
streaming support, a low-VRAM backend, or a Soup integration.

## Detailed evidence

- [v1.0 readiness audit](docs/release-v1.0-readiness.md)
- [machine-readable v1.0 audit](results/release/v1.0-readiness-audit.json)
- [E04 selected recipe and results](docs/experiment-04/e04-final-recipe.md)
- [E05 fresh blind results](docs/experiment-05/e05-results.md)
- [E06 cross-model results](docs/experiment-06/e06-results.md)
- [E07 failure-boundary results](docs/experiment-07/e07-results.md)
- [E08 efficiency-frontier results](docs/experiment-08/e08-results.md)
- [methodology](docs/methodology.md)
- [roadmap](docs/roadmap.md)

More detail is kept in the experiment records and persisted `results/` bundles;
the README summarizes the evidence rather than replacing those records.
