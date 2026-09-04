# Methodology

## Baseline-first specialization

CauseTune starts by measuring an untouched base model on a frozen challenge benchmark. A task is interesting only when the base has a meaningful, measurable capability gap. Fine-tuning is not performed first and justified afterward by a favorable score.

## Capability-gap screening

Future candidate domains should receive small frozen challenge sets before full dataset construction. Tasks where Qwen3-4B is already strong should be rejected as weak specialization experiments. The selected task should make the base → tuned change measurable on the intended metric.

## Train/validation/test isolation

Training uses only train data. Validation supports progression and checkpoint decisions. ID, HARD, and OOD remain sealed until final evaluation and must not become model-selection sets through repeated reuse.

## Frozen evaluation contracts

The evaluation contract fixes the instruction, chat-template arguments, thinking mode, generation settings, parser, and schema rules. The same fingerprinted contract is used before and after tuning so interface compliance is not confused with semantic improvement.

## Assistant-only supervision

Qwen chat formatting is used for training, but loss is applied only to the assistant JSON completion. System and user tokens are ignored. Every example must retain supervised assistant tokens.

## Determinism and controlled changes

Seeds, preprocessing, dataset fingerprints, exact training order, optimizer-window composition, and resolved configuration are persisted. Causal experiments change one training-affecting variable at a time. M6 changed only M5’s data order: deterministic shuffle with seed 42.

## Failed runs and optimizer-step analysis

A falling training loss is not sufficient evidence of useful adaptation. CauseTune records validation progression, class composition per optimizer window, confusion matrices, failure transitions, throughput, and VRAM. M5’s failure showed why balanced class counts can still produce pathological sequential optimization.

## Checkpoint selection and early stopping

The final checkpoint is not automatically the best checkpoint. Future selection should use validation-only checkpoints, patience, `min_delta`, and an explicit quality tolerance. ID, HARD, and OOD must remain sealed until the selected checkpoint is evaluated once.

## Quality versus efficiency

An experiment should report both specialization quality and the cost of obtaining it: optimizer steps, wall-clock time, input/supervised token throughput, peak VRAM, and trainable parameter count. Future claims should state how much of the capability gap was closed and how much training was required.

## Synthetic-data caveats

The current benchmark is synthetic and controlled; it is not production traffic. Once ID/HARD/OOD have been observed, they should not be repeatedly reused for hyperparameter or model selection. Future work needs fresh untouched benchmarks and, eventually, real domain data.

## Scope boundary

CauseTune owns offline model specialization and evidence generation. It is not a serving system, orchestration platform, registry, or production MLOps control plane.
