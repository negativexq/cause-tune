# CauseTune

## LLM Fine-Tuning Laboratory

**Measure the gap. Fine-tune. Explain the gain.**

CauseTune is a hands-on LLM fine-tuning laboratory for measuring specialization
gains, training dynamics, generalization, failure behavior, and efficiency under
constrained hardware.

## Experimental loop

```text
TASK CANDIDATE
    ↓
FROZEN CHALLENGE BENCHMARK
    ↓
UNTOUCHED BASE MODEL → MEASURE CAPABILITY GAP
    ↓
FINE-TUNE → VALIDATION → CHECKPOINT SELECTION / EARLY STOPPING
    ↓
SAME FROZEN EVALUATION → BASE vs TUNED
    ↓
QUALITY + EFFICIENCY + FAILURE ANALYSIS
```

Each experiment asks three questions:

- What could the untouched model do?
- What capability did specialization add, and did it generalize?
- How much training, memory, and adapter capacity did it require?

## Measured results

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

## What CauseTune measures

**Quality** — task-specific accuracy/F1, slice behavior, per-family metrics,
confusion pairs, and failure transitions.

**Training** — loss, finite-loss status, optimizer steps, validation
progression, gradient norm when available, and checkpoint progression.

**Efficiency** — wall-clock time, token throughput, peak allocated VRAM,
trainable parameter count, and optimizer steps required.

**Integrity** — dataset/evaluation fingerprints, deterministic preprocessing,
validation-only selection, explicit experiment controls, and fresh adapter
reload verification.

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
python -m pip install -e ".[dev]"
python -m compileall -q src scripts tests
pytest -q
git diff --check
```

## Scope boundary

CauseTune covers offline specialization and evidence generation: preprocessing,
LoRA/QLoRA, quantization, validation, checkpointing, early stopping,
telemetry, VRAM profiling, throughput, diagnostics, frozen evaluation, and
base-vs-tuned measurement.

It is not Kubeflow, an MLflow replacement, a serving system, a deployment or
routing layer, a canarying system, or a production control plane. Production
promotion and operational control belong elsewhere.

## Limitations

- The benchmark and training corpus are synthetic; 99.31% is benchmark accuracy, not production accuracy.
- Experiment 02 studies only Qwen3-4B and one controlled semantic training run.
- TRANSFER is transfer-style evaluation, not true OOD.
- Synthetic generator structure may make specialization easier than real-world incident diagnosis.
- Repeatedly tuning against this frozen benchmark would weaken the evidence; stronger claims require new untouched challenge sets.

## Future work

The next scientific question is whether the measured specialization gain survives
a new blind benchmark. Useful follow-ups are:

- a fresh blind challenge set with more heterogeneous incident narratives;
- independent generator/template families and manually authored cases where licensing and privacy permit;
- another model family;
- data-efficiency, LoRA-capacity, or learning-rate studies evaluated on new untouched evidence.

More detail: [experiment record](docs/experiments.md),
[methodology](docs/methodology.md), and [roadmap](docs/roadmap.md).
