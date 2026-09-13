# Experiment 07 — Failure Boundary and Abstention Protocol

Status: `FROZEN` before model evaluation.

## Scientific question

Does specialization cause confident incorrect diagnoses when evidence is insufficient or ambiguous?

E07 is evaluation-only. It does not change any training recipe or checkpoint.

## Frozen challenge

The challenge contains 60 cases in seven explicit categories:

- 12 sufficient-evidence cases
- 12 insufficient-evidence cases
- 8 contradictory-evidence cases
- 8 multiple-plausible-culprit cases
- 6 missing-topology cases
- 6 missing-metrics cases
- 8 out-of-taxonomy cases

The benchmark fingerprint is `c797b3a8378a2cbb2cb1e2a27e05a561087a62ed3f9adc74aa050d9f3f42a800`.

The challenge was generated with a separate E07 namespace and seed. Exact and normalized packet overlap against the E02 benchmark, E02B training/validation data, E05, and E06 final challenge is zero. The benchmark reuses the frozen incident taxonomy and packet sections; this is disclosed rather than presented as a new taxonomy.

Inputs, truths, prompt, scorer, decoding, and model set are frozen in `results/experiment_07/protocol.json` before evaluation.

## Output contract

The versioned contract adds `status` and `required_evidence` while retaining the diagnostic fields. `status` is one of:

- `diagnose`: enough non-contradictory evidence for one defensible taxonomy diagnosis
- `insufficient_evidence`: required causal evidence is missing
- `ambiguous_evidence`: evidence supports multiple plausible or contradictory explanations
- `out_of_taxonomy`: no frozen failure family matches the evidence

For non-diagnosis statuses, diagnosis fields must be null/empty and the response must explain what evidence is missing or conflicting through `required_evidence`. Scoring is deterministic and uses no LLM judge.

## Compared systems

Exactly one frozen benchmark evaluation is planned for each:

1. untouched Qwen base
2. original E02 adapter
3. E04-selected adapter

All systems use the same prompt, greedy decoding, scorer, and benchmark. E07 is excluded from training and checkpoint selection.

## Gate

G07 requires frozen inputs and semantics, exclusion from training, deterministic scoring, raw predictions, offline reproduction, and explicit false-confidence analysis on insufficient/ambiguous evidence.
