# Offline evidence verification

Verify a completed run without model loading or inference:

```bash
causetune verify runs/<run_id> --offline
```

The verifier checks the versioned manifest, resolved configuration hash, model
and data identities, persisted evaluation contract, validation-only checkpoint
record and every artifact hash. If `predictions.jsonl` and `evaluation.json`
are present together, the deterministic `causetune-exact-match-v1` scorer
recomputes count, correct and exact-match metrics and compares them with the
persisted evaluation.

Optional artifacts that are intentionally absent are reported as
`NOT_APPLICABLE`, never as PASS. Required evidence files or any one-byte
configuration, data, prediction, evaluation or artifact mutation produce a
FAIL and exit code `4`. Verification is read-only: it never repairs files,
regenerates predictions or loads model weights.
