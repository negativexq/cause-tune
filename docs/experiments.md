# Lab Experiment 01 — Controlled QLoRA Specialization

## Goal

Experiment 01 validated CauseTune’s hands-on fine-tuning laboratory workflow: establish a frozen baseline, fine-tune once under controlled settings, inspect failures, and isolate a training-order cause with one variable. It was intentionally narrow and is not the final challenge task for the laboratory.

## Base capability

M4 created the realistic synthetic customer-support intent benchmark and evaluated untouched Qwen/Qwen3-4B under the frozen evaluation contract. Base accuracy was 88.0% validation, 83.6% ID, 76.8% HARD, and 79.6% OOD.

## First realistic fine-tune

M5 trained QLoRA on the 2,000-example train split with the verified Qwen3-4B NF4/BF16 configuration. The run used `shuffle=False`, preserving ten class-contiguous blocks of 200 examples.

## Failure

M5 accuracy fell to 26.8% validation, 24.8% ID, 23.6% HARD, and 20.0% OOD despite training loss collapsing. JSON/schema compliance stayed near 100%, so this was a semantic failure rather than a format-learning success.

## Root-cause audit

The audit found:

- `shuffle=False` preserved class-contiguous order.
- Microbatch `1` and gradient accumulation `8` made 250/250 optimizer windows single-class.
- The final 25 optimizer steps contained only `wrong_item` examples.
- Loss and predictions showed catastrophic recency/forgetting behavior.

This is why a balanced dataset did not produce balanced optimization.

## Controlled intervention

M6 changed exactly one training-affecting variable: deterministic seeded shuffle with seed `42`. Dataset contents, labels, model, quantization, LoRA, optimizer settings, masking, and frozen evaluation contract remained fixed.

## Result

M6 produced 0/250 single-class windows and 250/250 mixed-class windows, with no terminal `wrong_item` block. Accuracy recovered to 99.2% validation, 97.2% ID, 96.0% HARD, and 92.0% OOD. The key `wrong_item` collapse pairs fell from M5 to M6 as follows: `duplicate_charge` 100→0, `fraud_suspected` 100→1, `order_missing` 100→6, `refund` 99→1, and `cancel_order` 98→2. No replacement single-class collapse occurred.

## What we learned

Experiment 01 established practical lessons about QLoRA mechanics, assistant-only loss, quantized parameter accounting, frozen evaluation, optimizer-window composition, deterministic ordering, failure analysis, held-out evaluation, and fresh adapter reload. Its strongest lesson is that train loss alone is insufficient evidence of useful specialization. Failed runs should be retained as evidence, and future tasks should begin with capability-gap screening.

This remains a learning experiment on synthetic data, not a production benchmark.

# Experiment 02A — Production Incident Diagnosis Capability Gap

## Why this task

Experiment 02A begins the next gap-first specialization candidate: a controlled production-incident diagnosis task aligned with production AI/ML systems engineering. The objective is to measure how weak an untouched Qwen3-4B is before deciding whether fine-tuning is worth the cost.

## Task

Each incident packet contains topology, recent changes, metrics, logs/events, and dependency health. The model must return a strict JSON diagnosis with one culprit component, one failure family, one operational action, and supporting evidence IDs. Ground truth is stored separately from the model input.

## Benchmark

The frozen benchmark contains 144 cases:

- 72 STANDARD cases
- 48 HARD cases
- 24 TRANSFER cases
- 12 failure families, with 12 cases per family

HARD cases include controlled red herrings and downstream symptoms. TRANSFER changes topology and surface wording without changing the taxonomy. This is a controlled synthetic benchmark, not a claim of production SRE readiness.

## Metrics

The primary metric is diagnosis exact match: both `culprit_service` and `failure_mode` must be correct. A correct generic failure family with the wrong culprit is not a successful diagnosis, and the reverse is also insufficient. Resolution exact match, culprit accuracy, failure-mode accuracy/macro-F1, action accuracy, evidence precision/recall/F1, and strict JSON compliance are secondary metrics.

## Integrity

The benchmark and evaluation contract are frozen before the base run. The baseline uses untouched Qwen/Qwen3-4B, greedy deterministic decoding, `enable_thinking=False`, no adapter, no training, no few-shot examples, no retrieval, no external tools, no output repair, and no LLM judge. Experiment 01’s dataset and evaluation fingerprints are not reused.

Benchmark fingerprint: `5e9a6d74ba881af7aa1146a23f4b837e139a4e8e5a77dba7fe40c3bb2c9c0a94`<br>
Experiment 02A evaluation-contract fingerprint: `d2dea87fd26a3a8c78a2a6a1b4b6fc583b27070ed2de7e691b53c17e59231cf8`

## Baseline

The frozen base was evaluated once after the benchmark commit. The model was untouched Qwen/Qwen3-4B with greedy decoding, `enable_thinking=False`, no adapter, no few-shot examples, no retrieval, and no output repair. No training data was generated in 02A.

Diagnosis exact match is the primary metric:

| Slice | Diagnosis exact match | Resolution exact match | Culprit accuracy | Failure-mode accuracy | Failure-mode macro F1 | Action accuracy | Evidence F1 | Strict JSON |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ALL (144) | 65.28% (94) | 34.72% (50) | 77.78% (112) | 71.53% (103) | 71.84% | 40.97% (59) | 80.88% | 88.89% (128) |
| STANDARD (72) | 66.67% (48) | 37.50% (27) | 80.56% (58) | 72.22% (52) | 71.75% | 44.44% (32) | 81.74% | 91.67% (66) |
| HARD (48) | 64.58% (31) | 35.42% (17) | 75.00% (36) | 75.00% (36) | 74.39% | 41.67% (20) | 80.56% | 89.58% (43) |
| TRANSFER (24) | 62.50% (15) | 25.00% (6) | 75.00% (18) | 62.50% (15) | 65.00% | 29.17% (7) | 78.83% | 79.17% (19) |

JSON parsing was 100% valid, while strict schema compliance was 128/144. The 16 strict failures were 13 unknown culprit values and 3 invalid evidence references. The first raw generation was retained; a parser correctness fix accepted any evidence ID present in the packet, then re-aggregated the persisted outputs without another model run. The evaluation-contract fingerprint did not change.

The largest mechanically observed failure patterns were 12 recent-change/deploy biases, 17 distractor selections on HARD cases, 18 correct culprit/wrong family cases, 9 correct family/wrong culprit cases, and 44 correct diagnoses with the wrong action. The largest failure-mode confusions were `configuration_regression` → `db_connection_pool_exhaustion` (9), `downstream_dependency_timeout` → `dns_resolution_failure` (5), and `kafka_consumer_lag` → `downstream_dependency_timeout` (4). `cache_stampede`, `configuration_regression`, and `downstream_dependency_timeout` were weakest at 0/12 diagnosis exact; `memory_leak` was strongest at 12/12.

## Decision

The base leaves substantial headroom: diagnosis exact match is 65.28% overall and 62.50% on TRANSFER. HARD is 64.58%, only 2.08 percentage points below STANDARD, so the slice does not strongly separate difficulty yet; TRANSFER exposes a larger surface-generalization drop of 4.17 points. The deterministic evaluator is trustworthy enough to support a controlled next phase, with strict-schema limitations reported separately. The benchmark will not be changed to manufacture a larger gap, and no tuned result is claimed here.

## Experiment 02B — Specialization

### Experiment 02B.1 — Training foundation

Status: training foundation complete; GPU training has not started.

The 02A baseline establishes a meaningful capability gap without using its
per-family weaknesses to shape the training distribution. The independent
02B.1 dataset contains 2,400 balanced training incidents and 288 balanced
validation incidents: 200 and 24 per frozen failure family respectively.
Training and validation use separate deterministic namespaces and new
topology/surface families. The 144-case 02A benchmark is not copied,
paraphrased, or used as a training source.

The training foundation uses the same production-incident system instruction,
Qwen chat formatting, and assistant-only supervision as the frozen task
contract. Token inspection selected a 768-token context because it is the
smallest configured practical context covering every generated conversation;
there are zero truncations and zero decoded-target mismatches.

Checkpoint selection is validation-only: primary metric
`diagnosis_exact_match`, followed by resolution exact match, failure-mode
macro F1, teacher-forced loss, and earlier step. Validation is planned every
25 optimizer steps including step 0. The transparent early-stop policy uses
patience 3, `min_delta=0.005`, and a 50-step warm-up floor. The frozen 02A
benchmark remains sealed until a future selected adapter is evaluated.

### Experiment 02B.2 — QLoRA specialization — complete

The first real semantic run used the resolved 02B.1 manifest exactly once:
Qwen/Qwen3-4B with NF4, BF16 compute, double quantization, rank-16 LoRA with
alpha 32, zero dropout, all seven attention/MLP projection targets,
assistant-only supervision, 768-token context, microbatch 1, accumulation 8,
learning rate `2e-4`, two epochs, and a 600-step maximum. Training executed 100
optimizer steps and stopped by the frozen validation policy after three
post-warm-up evaluations without a meaningful diagnosis improvement. The best
validation checkpoint was step 100; step 25 was the earliest checkpoint within
1 percentage point of the best diagnosis score.

Validation progression:

| Step | Diagnosis | Resolution | Macro F1 | Action | Evidence F1 | Loss |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 58.33% (168/288) | 25.35% | 59.04% | 28.82% | 66.04% | 1.47637 |
| 25 | 100.00% (288/288) | 97.92% | 100.00% | 97.92% | 100.00% | 0.001167 |
| 50 | 100.00% (288/288) | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000369 |
| 75 | 100.00% (288/288) | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000200 |
| 100 | 100.00% (288/288) | 100.00% | 100.00% | 100.00% | 100.00% | 0.0000144 |

Frozen benchmark comparison:

| Metric | Base | Tuned | Delta |
| --- | ---: | ---: | ---: |
| Diagnosis exact | 65.28% (94/144) | 99.31% (143/144) | +34.03 pp |
| Resolution exact | 34.72% | 99.31% (143/144) | +64.58 pp |
| Culprit accuracy | 77.78% | 100.00% | +22.22 pp |
| Failure-mode accuracy | 71.53% | 99.31% | +27.78 pp |
| Failure-mode macro F1 | 71.84% | 99.30% | +27.47 pp |
| Action accuracy | 40.97% | 100.00% | +59.03 pp |
| Evidence F1 | 80.88% | 100.00% | +19.12 pp |
| Strict JSON | 88.89% | 100.00% | +11.11 pp |

Diagnosis exact improved from 66.67% to 100.00% on STANDARD (+33.33 pp),
64.58% to 97.92% on HARD (+33.33 pp), and 62.50% to 100.00% on TRANSFER
(+37.50 pp). TRANSFER is reported as TRANSFER, not true OOD. The previously
weak frozen families were not oversampled during training.

Diagnosis transitions were 49 base-wrong → tuned-correct, 1 base-wrong →
tuned-wrong, 94 persistent correct, and 0 persistent wrong. Resolution had 93
improvements, 1 regression, 50 persistent correct, and 0 persistent wrong.
Actions had 85 improvements and 59 persistent correct, with no regressions.

Per-family diagnosis exact results were: `db_connection_pool_exhaustion`
58.33% → 100.00%, `db_query_regression` 75.00% → 100.00%, `memory_leak`
100.00% → 100.00%, `downstream_dependency_timeout` 33.33% → 91.67%,
`cache_stampede` 100.00% → 100.00%, `kafka_consumer_lag` 58.33% → 100.00%,
`thread_pool_exhaustion` 83.33% → 100.00%, `disk_io_saturation` 25.00% →
100.00%, `dns_resolution_failure` 100.00% → 100.00%,
`tls_certificate_expiration` 91.67% → 100.00%,
`rate_limit_misconfiguration` 58.33% → 100.00%, and
`configuration_regression` 0.00% → 100.00%. The remaining error is one HARD
`downstream_dependency_timeout` → `dns_resolution_failure` family confusion;
no new diagnosis confusion pair was introduced.

Mechanically observed failures fell from 12 recent-change/deploy biases to 0,
17 HARD distractor selections to 1, 18 correct-culprit/wrong-family cases to
1, 9 correct-family/wrong-culprit cases to 0, and 44 correct-diagnosis/wrong-
action cases to 0. Base invalid culprit values (13) and invalid evidence
references (3) fell to 0; strict-schema failures fell from 16 to 0. No LLM
judge or output repair was used.

Training took 10,263.71 seconds, validation 4,455.54 seconds, and measured
wall time from preflight artifact creation through prediction persistence was
10,806.54 seconds. Throughput was 43.83 input tokens/sec and 3.07 supervised
tokens/sec; peak VRAM was 5.312 GiB allocated and 9.342 GiB reserved. The
100-step stop avoided 500 of 600 configured updates (83.33%); step 25 was the
earliest near-best checkpoint for efficiency analysis.

Integrity: all four frozen fingerprints matched; training ran once; selection
was validation-only; fresh base plus adapter reload passed; exactly one tuned
benchmark generation produced 144 raw outputs; and recovery scored those
persisted outputs offline. Initial post-generation analysis failed with
`KeyError: 'all'` because the synthetic aggregate was incorrectly looked up as
a real dataset split. The narrow aggregate handling fix and regression test
changed no inputs, predictions, prompts, or scoring rules.

Conclusion: STRONG SPECIALIZATION. The gain survived STANDARD, HARD, and
TRANSFER, with one remaining HARD family error and one diagnosis regression
against the base.
