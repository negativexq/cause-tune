# Experiment 04 — selected recipe and final gate

## G04 final gate

`G04 PASS`. G04A, G04B, and G04C each passed with validation-only selection;
all ten valid E04 semantic bundles (four E04-A fractions, three E04-B ranks,
and three E04-C learning rates) are represented by persisted evidence. The
75% first attempt remains a
separate `TECHNICAL_FAILURE` record and is excluded from quality comparison.
No negative semantic result was discarded, and E03 was not used to tune any
E04 choice.

## Selected recipe

The machine-readable recipe is
`results/experiment_04/selected_recipe.json`:

- data fraction: 25% / 600 examples;
- LoRA rank: 8, alpha: 32, with the frozen seven-module target policy;
- learning rate: `1e-4`;
- AdamW, microbatch 1, gradient accumulation 8, effective batch 8;
- NF4 with double quantization and BF16 compute;
- sequence length 768, deterministic seed `20260941`;
- validation-only checkpoint selection and the frozen early-stopping policy.

This recipe was selected under the predeclared E04 controlled study. It is
not a claim of global or universal optimality. Adapter weights remain local-
only; lightweight manifests, metrics, predictions, hashes, comparisons, and
selection provenance are persisted.
