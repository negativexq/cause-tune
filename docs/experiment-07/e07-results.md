# Experiment 07 — Failure Boundary Results

Status: `G07 PASS`.

The frozen 60-case boundary challenge was evaluated exactly once for each of:

- untouched Qwen3-4B base
- original E02 adapter
- E04-selected adapter

The benchmark fingerprint is `c797b3a8378a2cbb2cb1e2a27e05a561087a62ed3f9adc74aa050d9f3f42a800`. Raw outputs were persisted before scoring. All three system evaluations have deterministic offline reproduction `PASS` and artifact hashes in `results/experiment_07/evaluations/`.

## Aggregate results

| system | abstention precision | abstention recall | false confident diagnosis | false abstention | sufficient-case accuracy | schema validity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 100.0% | 54.2% | 5/48 (10.4%) | 5/12 (41.7%) | 58.3% | 63.3% |
| E02 | 100.0% | 37.5% | 15/48 (31.3%) | 6/12 (50.0%) | 50.0% | 65.0% |
| E04 | 100.0% | 54.2% | 9/48 (18.8%) | 0/12 (0.0%) | 100.0% | 78.3% |

The primary safety question has an explicit answer: E04 produced confident diagnosis outputs on 9 of 48 insufficient, contradictory, ambiguous, or out-of-taxonomy cases (18.8%). This is lower than E02's 15/48 (31.3%), but it is not zero. Specialization therefore did not eliminate confident errors at the failure boundary.

On the 12 sufficient-evidence cases, E04 preserved 100% complete diagnosis accuracy and had no false abstentions. This does not override the non-diagnosis boundary result.

## Important scoring provenance

The first deterministic scorer revision omitted invalid outputs from the sufficient-case false-abstention denominator. The semantic generations were already complete and were not rerun. The persisted raw predictions were rescored offline with `e07-boundary-scorer-v2`; `scoring_correction.json` preserves the prior aggregate metrics and records the correction.

No LLM judge was used. E07 was evaluation-only and was excluded from training and checkpoint selection. The result does not claim real-world production safety or external-benchmark performance.
