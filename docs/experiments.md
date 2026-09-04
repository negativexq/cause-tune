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
