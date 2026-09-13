# Experiment 04C — learning-rate sensitivity protocol

## Status

This protocol is frozen before any E04-C GPU run. The study asks which
learning rate preserves specialization quality under the E04-selected data
fraction and LoRA capacity.

## Intervention

The only intervention is learning rate:

| Variant | Learning rate |
| --- | ---: |
| lr1e-4 | 0.0001 |
| lr2e-4 | 0.0002 |
| lr4e-4 | 0.0004 |

All variants use the E04-A selected 25% subset (600 examples), subset hash
`eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56`, and the
E04-B selected rank-8 adapter with fixed alpha 32.

## Frozen controls

The model, pinned revision, training subset, validation corpus, seed, LoRA
rank and alpha, target modules, optimizer, quantization, compute dtype,
sequence length, microbatch, gradient accumulation, shuffle, prompt,
assistant-only supervision, checkpoint-selection policy, and early-stopping
policy are copied from the selected E04 contracts. Learning rate is the sole
primary intervention.

Checkpoint selection remains validation-only. E03 and all blind evidence
remain sealed and excluded from training, checkpoint selection, and learning-
rate selection. Technical retries, negative results, raw predictions, and
lightweight provenance remain separate and preserved.

## Selection rule

Selection uses validation only. First, variants must satisfy the same frozen
quality boundary used in E04-A and E04-B: diagnosis exact, resolution exact,
and failure-mode macro F1 within 1.0 percentage point of the best variant; no
failure-family metric may regress by more than 5.0 percentage points; and no
critical schema-validity regression may occur.

Among eligible variants, selection priority is diagnosis exact, failure-mode
macro F1, resolution exact, action accuracy, evidence F1, no critical family
regression, checkpoint stability, earlier stable convergence, then training
cost. The deterministic implementation records the complete comparison and
tie-break basis. E03 and future blind evidence are not selection inputs.
