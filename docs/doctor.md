# CauseTune Doctor

`causetune doctor` is the preflight boundary before a training run. The
default mode is CPU-safe: it resolves and validates the experiment contract,
reads dataset files, computes fingerprints, checks role isolation and
supervised training content, and records the local package environment. It
does not load model weights or allocate CUDA memory.

```bash
causetune doctor --config configs/experiment.json
causetune doctor --config configs/experiment.json --json
causetune doctor --config configs/experiment.json --hardware
```

Checks are grouped into Contract, Data, Training, Evaluation, Environment and
optional Hardware layers. Every check is `PASS`, `WARN` or `FAIL`:

- `PASS` means the check completed successfully.
- `WARN` is non-blocking and is never rewritten as PASS or FAIL.
- `FAIL` is blocking and returns exit code `3`.

A ready report returns exit code `0` for either PASS or WARN. Invalid command
usage is handled by the CLI layer; the doctor engine itself exposes the
structured report and `doctor_exit_code()` for embedding in other entrypoints.

The machine-readable report is serialized with stable key ordering. Hardware
inspection is explicitly opt-in so a normal preflight cannot accidentally
initialize a GPU runtime. Dataset fingerprints and train/validation/benchmark
role checks use the same resolved contract that the training workflow will
receive.
