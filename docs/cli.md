# Unified laboratory CLI

The supported command surface is intentionally thin:

```bash
causetune doctor --config configs/experiment.json
causetune prepare --config configs/experiment.json
causetune train --config configs/experiment.json
causetune train --config configs/experiment.json --prepare-only
causetune evaluate --predictions runs/x/predictions.jsonl --output runs/x/evaluation.json
causetune compare --base runs/base/evaluation.json --tuned runs/tuned/evaluation.json
causetune verify runs/x --offline
```

Commands route to package-level application APIs. The CLI does not implement
training, scoring, checkpoint selection or artifact hashing. `train` currently
reserves actual training for a package-level runner that is not yet wired. It
fails explicitly instead of claiming that training occurred. `prepare` is the
canonical preparation command; `train --prepare-only` is an explicit
backward-compatible alias that initializes the M10 evidence bundle with
`execution: not_started`. No scientific GPU run is started by M13. The first
authorized runner is reserved for E03.

`evaluate` consumes persisted predictions and writes aggregate metrics before
`compare` reads them. `compare` never generates model output. Structured JSON is
available with `--json`. Exit codes are stable: `0` success/ready, `1`
general failure, `2` invalid contract, `3` preflight failure and `4`
verification failure.
