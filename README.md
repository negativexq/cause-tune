# CauseTune

## LLM Fine-Tuning Laboratory

**Measure the gap. Fine-tune. Explain the gain.**

CauseTune is a hands-on LLM fine-tuning laboratory for measuring specialization gains, training dynamics, and efficiency under constrained hardware.

## What is CauseTune?

CauseTune is a small, readable laboratory for studying whether fine-tuning creates a meaningful, generalizing capability gain. It focuses on measurable specialization, baseline-first experiments, controlled training changes, training dynamics, failure analysis, held-out generalization, quality versus training-cost trade-offs, and constrained-GPU fine-tuning.

CauseTune is intentionally not Kubeflow, MLflow, a distributed training platform, an MLOps control plane, or a serving system. It owns offline model specialization and evidence generation; production promotion, routing, canarying, and rollback belong to ModelOps.

## The experimental loop

```text
TASK CANDIDATE
    ↓
FROZEN CHALLENGE BENCHMARK
    ↓
UNTOUCHED BASE MODEL
    ↓
MEASURE CAPABILITY GAP
    ↓
Is the base already good enough?
    ├── YES → reject the task as an uninteresting fine-tuning experiment
    └── NO → FINE-TUNE → VALIDATION → CHECKPOINT SELECTION / EARLY STOPPING
                         → SAME FROZEN EVALUATION → BASE vs TUNED
                         → QUALITY + EFFICIENCY + FAILURE ANALYSIS
```

Every serious CauseTune experiment should answer:

- **Capability gap:** How weak was the untouched base model?
- **Specialization gain:** How much of that gap did fine-tuning close?
- **Efficiency:** How much training, memory, and adapter capacity were required?

## Why baseline first?

Do not fine-tune first and search for a success story afterward. First prove that the frozen base model has a meaningful capability gap. If a base model already scores 95% on a task, there is little specialization impact to demonstrate.

Illustrative example only — these values are **not current measured results**:

| Base HARD | Tuned HARD | Gain | Best checkpoint | Updates avoided | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42% | 76% | +34 pp | step 80 / 250 | 68% | 4.5 GiB |

## Lab Experiment 01 — Controlled QLoRA Specialization

Experiment 01 used a deliberately narrow customer-support intent-classification task. Its purpose was to validate the laboratory workflow and learn fine-tuning mechanics, not to serve as CauseTune’s final challenge task.

The synthetic benchmark established hands-on evidence for Qwen3-4B fine-tuning, QLoRA, NF4 loading, BF16 compute, PEFT behavior, assistant-only supervision, deterministic preprocessing, frozen evaluation contracts, ID/HARD/OOD evaluation, optimizer-step telemetry, fresh adapter reload, training-order effects, failure analysis, catastrophic recency, and one-variable experiments.

The verified values below are **accuracy metrics**:

| Experiment | Validation | ID | HARD | OOD |
| --- | ---: | ---: | ---: | ---: |
| Base Qwen3-4B | 88.0% | 83.6% | 76.8% | 79.6% |
| M5 QLoRA — unshuffled | 26.8% | 24.8% | 23.6% | 20.0% |
| M6 QLoRA — deterministic shuffle | 99.2% | 97.2% | 96.0% | 92.0% |

M5 used `shuffle=False` over 10 class-contiguous blocks of 200 examples. With microbatch `1` and gradient accumulation `8`, all 250/250 effective optimizer windows were single-class. The final 25 optimizer updates contained only `wrong_item`, and the model catastrophically recentered on that terminal class.

M6 changed exactly one training-affecting variable: the order became a deterministic seeded shuffle with seed `42`. All 250 windows became mixed-class, the terminal block disappeared, and held-out performance recovered. The central lesson is not simply that fine-tuning improved accuracy: train loss alone is insufficient evidence, and a balanced dataset can still produce pathological optimizer windows.

### Failure analysis

| Expected intent → `wrong_item` | M5 | M6 |
| --- | ---: | ---: |
| `duplicate_charge` | 100 | 0 |
| `fraud_suspected` | 100 | 1 |
| `order_missing` | 100 | 6 |
| `refund` | 99 | 1 |
| `cancel_order` | 98 | 2 |

No replacement collapse into another single class occurred. `order_missing` remained the weakest M6 class, especially on HARD/OOD examples; this release does not tune against that observation.

## Training configuration

- Qwen/Qwen3-4B; NF4 4-bit base; BF16 compute; double quantization
- PEFT LoRA: rank `16`, alpha `32`, dropout `0`
- targets: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- assistant-only supervision; `enable_thinking=False`
- microbatch `1`; gradient accumulation `8`; effective batch `8`
- learning rate `2e-4`; 1 epoch; 250 optimizer steps

Logical parameters: `4,055,498,240`<br>
Trainable parameters: `33,030,144` (`~0.814453%`)

PEFT’s logical parameter API is used for structure checks. Naive `sum(p.numel())` on 4-bit packed weights is a physical packed-parameter count, not the logical dense model count.

## Training dynamics

M6 validation accuracy progression:

| Step | 0 | 25 | 50 | 75 | 100 | 125 | 150 | 175 | 200 | 225 | 250 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Accuracy | 88.4% | 99.2% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 99.6% | 99.2% |

Did the model really need all 250 optimizer updates? The late validation regression motivates checkpoint selection and early stopping, but Experiment 01 does not answer that question.

## What CauseTune measures

**Quality** — task-specific accuracy/F1, ID/HARD/OOD behavior, per-class metrics, confusion pairs, and failure transitions.

**Training** — loss, finite-loss status, optimizer steps, validation progression, gradient norm when available, and checkpoint progression.

**Efficiency** — wall-clock time, input and supervised token throughput, peak VRAM, trainable parameter count, and optimizer steps required.

**Integrity** — dataset/evaluation fingerprints, seed, deterministic preprocessing, explicit one-variable differences, and fresh adapter reload verification.

## Experimental setup and integrity

The benchmark has 3,000 synthetic examples: 2,000 train, 250 validation, 250 ID, 250 HARD, and 250 OOD across 10 intents. ID follows the intended held-out distribution. HARD emphasizes confusable, ambiguous, multi-issue, and noisy cases. OOD holds out surface/template families. This benchmark is not production traffic.

Dataset fingerprint: `0efaacae27cbfeb1c304c8ee359384239d4526470a32aded8f0eda39908e9d06`<br>
Frozen evaluation-contract fingerprint: `1b00a333c26c4cbd03b3e04d990fad3b4adf0d9a03443c9fd5f183ee7e8ef94d`

The same strict JSON evaluation contract was used for base and tuned models; malformed outputs were not repaired.

## Selecting future experiments

The next specialization task has not been selected:

1. Shortlist objectively measurable specialist domains.
2. Build small frozen challenge sets.
3. Evaluate untouched Qwen3-4B and measure the capability gap.
4. Discard tasks where the base is already too strong.
5. Select one useful gap, then build train/validation/test data.
6. Fine-tune and measure exact base → tuned impact.

Candidate domains include SRE incident diagnosis, SOC/security alert triage, telecom/network operations, company-specific query/DSL generation, internal tool/API behavior, and industrial fault diagnosis. These are examples only. Experiment 02 has not started or been selected.

## M7 — Checkpoint Selection & Early Stopping

M7 is documented as the next laboratory milestone, not implemented here. It should test validation every N optimizer steps, checkpoint persistence, best-validation metadata, patience, `min_delta`, early stopping, earliest checkpoint within a quality tolerance, and optimizer-step/training-time savings.

Checkpoint selection must use validation only; ID, HARD, and OOD remain sealed until final evaluation. M6 motivates this work but does not prove which checkpoint is optimal.

## Repository layout

```text
configs/          experiment and evaluation configuration
docs/             experiment record, methodology, and roadmap
results/          concise verified public evidence
scripts/          dataset, audit, training, and evaluation entrypoints
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

In scope are preprocessing, QLoRA/LoRA, quantization, checkpointing, early stopping, validation, telemetry, VRAM profiling, throughput, diagnostics, experiment comparison, frozen evaluation, and base-vs-tuned measurement.

Out of scope are Kubernetes, distributed training orchestration, serving infrastructure, production deployment, canarying, routing, online monitoring, model registry services, and API/dashboard work.

## Limitations and future work

The benchmark is synthetic, only one base model family and one GPU environment have been studied, and M6 uses one deterministic seed. The observed ID/HARD/OOD sets should not be repeatedly reused for future selection. Future questions include M7, a gap-first Experiment 02, learning-rate and LoRA-capacity studies, fresh untouched benchmarks, more model families, real domain datasets, and larger post-training methods. None are completed here.

More detail: [experiment record](docs/experiments.md), [methodology](docs/methodology.md), and [roadmap](docs/roadmap.md).
