# Experiment 08 — Training-Efficiency Frontier Protocol

Status: `FROZEN` before frontier classification.

## Scientific question

Was the E04 training-efficiency gain worth the E05 fresh-blind generalization regression?

E08 is an analysis of already completed, valid measurements. It does not launch a hyperparameter search or rerun semantic evaluations.

## Predeclared candidates

Only these measured recipes are candidates:

1. `E02 original recipe`: 2,400-example training corpus, rank 16, learning rate `2e-4`, selected E02 checkpoint.
2. `E04 selected recipe`: 600-example E04-A subset, rank 8, learning rate `1e-4`, selected E04 checkpoint.

If two rows are identical on a metric, both rows remain; no synthetic candidate is created.

## Quality reference and tolerance

The E02 original recipe is the quality reference because E08 asks whether the E04 efficiency change preserved the original specialization quality.

A lower-cost recipe is quality-preserving only if all conditions hold:

- diagnosis exact is within `-1.0 percentage point` of the reference;
- failure-mode macro F1 is within `-1.0 percentage point` of the reference;
- no critical failure-family regression exceeds `5.0 percentage points`;
- no major schema-validity regression exists.

The E05 fresh-blind metrics are used only because E02 and E04 were both evaluated exactly once on the same frozen 120-case benchmark. The E04 E05 diagnosis result (`92.5%`) is not changed or reinterpreted as an E04 selection criterion.

## Cost provenance

Cost fields are reported with their exact provenance and labels:

- unique training-corpus examples;
- examples processed by the recorded run;
- supervised tokens processed;
- optimizer steps executed;
- selected checkpoint step;
- wall-clock training seconds;
- peak allocated VRAM;
- peak reserved VRAM;
- trainable parameters.

Allocated and reserved VRAM are not conflated with physical GPU capacity.

## Gate

G08 requires predeclared candidates, no automatic search, provenance-backed cost fields, frozen quality tolerance, deterministic Pareto classification, and retention of dominated/negative candidates.
