# CauseTune v0.2.0 — Laboratory Hardening

This release freezes the laboratory hardening boundary:

```text
G8  Experiment Contract       PASS
G9  Doctor                    PASS
G10 Evidence Provenance       PASS
G11 Offline Verification      PASS
G12 Real Training CI          PASS
G13 Unified CLI               PASS
```

The release includes deterministic resolved contracts, CPU-safe preflight,
versioned run evidence, offline tamper verification, a real Trainer/PEFT model
smoke and the thin `causetune` CLI. Existing E01/E02 evidence is not rewritten.
Experiment 03 was subsequently run against a newly frozen independent blind
benchmark and its raw predictions are persisted under
`results/experiment_03_blind/`.

E04 remains a predeclared focused-study matrix, not automatic hyperparameter
search. EX-LV remains a research admission track with no streaming backend
admitted. No serving, orchestration, export platform, recipe zoo or trainer
autopilot is part of this release.
