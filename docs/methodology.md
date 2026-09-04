# Methodology

## Frozen evaluation contracts

An evaluation contract fixes the task instruction, chat-template arguments, thinking mode, generation settings, parser, and schema rules. The same fingerprinted contract is used for the frozen base and tuned adapters. This prevents output-interface changes from being mistaken for semantic improvement.

## Assistant-only supervision

Training formats each record with the Qwen chat template, but labels only the assistant JSON completion. System and user tokens use the ignore index. This measures learning of the required intent response rather than copying the prompt. Every preprocessed example must retain at least one supervised token.

## Dataset separation

Training uses only the train split. Validation is used for deterministic progression checks. ID, HARD, and OOD remain held out until the prescribed final evaluation. Stable example IDs, normalized-content checks, scenario-family tracking, and OOD family exclusion make accidental leakage visible.

## ID, HARD, and OOD

ID samples follow the intended surface distribution while holding out examples. HARD emphasizes confusable intents, ambiguity rules, multi-issue requests, and noise. OOD holds out wording and surface families while preserving the same ten-label taxonomy. These are controlled benchmark definitions, not claims about real traffic.

## Synthetic-data caveats

The benchmark is generated without an external LLM API and is useful for controlled causal experiments. Its language distribution is not production traffic. Repeatedly selecting models or hyperparameters against observed ID/HARD/OOD sets would turn them into development data, so future work should create a fresh untouched benchmark.

## Reproducibility and experiment diffs

Configs, seeds, fingerprints, exact training orders, optimizer-window composition, loss histories, and runtime metadata are persisted. M6 records a minimal diff from M5: deterministic shuffle seed 42, plus experiment output identity. No other training-affecting setting changed.

## PEFT parameter accounting

For PEFT models, structural checks use `model.get_nb_trainable_parameters()` when available. The naive `sum(p.numel())` value is reported separately as packed physical parameter numel because bitsandbytes 4-bit storage does not represent the logical dense parameter count. Treating the packed count as the logical Qwen3-4B total would produce a false preflight failure.

## Failed runs as evidence

M5 is retained because its failure exposed a causal training-order bug. A near-zero training loss is not sufficient evidence of useful adaptation; held-out semantic metrics, class recall, confusion matrices, and failure transitions are required. M6 therefore changed one variable and sealed test evaluation during training.
