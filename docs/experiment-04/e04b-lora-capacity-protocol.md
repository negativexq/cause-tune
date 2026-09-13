# Experiment 04B — LoRA-capacity protocol

## Status

This protocol is frozen before any E04-B GPU run. The study asks what LoRA
rank is required after the E04-A minimum sufficient data fraction has been
selected.

## Intervention

The only intervention is LoRA rank:

| Variant | Rank |
| --- | ---: |
| r8 | 8 |
| r16 | 16 |
| r32 | 32 |

All variants use the E04-A selected 25% subset: 600 examples with subset hash
`eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56`.

## Frozen controls

The model, pinned revision, subset, validation corpus, seed, learning rate,
optimizer, quantization, compute dtype, sequence length, microbatch,
gradient accumulation, shuffle, prompt, assistant-only supervision,
checkpoint-selection policy, and early-stopping policy are copied from the
selected E04-A contract.

LoRA alpha is fixed at `32` for every rank. This is the existing CauseTune
alpha used by the rank-16 E02 and E04-A methodology; fixing alpha isolates rank
without changing the intervention into an alpha/rank scaling study. Dropout,
target modules, and all other adapter settings remain unchanged.

Checkpoint selection is validation-only. E03 and all blind evidence remain
sealed and excluded from training, checkpoint selection, and rank selection.
Raw predictions, lightweight provenance, negative results, and technical
retry records are retained.

## Selection rule

After all three semantic runs verify, select the smallest rank satisfying the
same predeclared validation rule used by E04-A: diagnosis exact, resolution
exact, and failure-mode macro F1 must each be within 1.0 percentage point of
the best variant; no failure-family metric may regress by more than 5.0
percentage points; and no critical schema-validity regression may occur.
Ties are resolved by trainable parameter count, then peak VRAM, then wall
time. E03 is not eligible as a selection input.
