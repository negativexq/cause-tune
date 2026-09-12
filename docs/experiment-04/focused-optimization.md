# Experiment 04 — Focused Optimization

Experiment 04 is a sequence of one-question studies. Each study freezes one
primary intervention and controls all other training variables:

| Study | Intervention | Levels |
| --- | --- | --- |
| E04-A | data fraction | 25%, 50%, 75%, 100% |
| E04-B | LoRA rank | 8, 16, 32 |
| E04-C | learning rate | 1e-4, 2e-4, 4e-4 |

The contracts are generated before any result is read:

```bash
python scripts/prepare_e04_studies.py
```

Every contract pins the blind benchmark fingerprint, validation-only
checkpoint selection, raw-prediction persistence, evidence verification and
retention of negative results. The blind benchmark is never a tuning input.
The current repository contains the predeclared study machinery and contracts;
scientific GPU executions remain explicit study runs rather than an automatic
hyperparameter search.
