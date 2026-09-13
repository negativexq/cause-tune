# Experiment 06 — Capability-gap screen

Status: `TECHNICAL_FAILURE` at the pre-training capability-gap screen.

The selected second family was `microsoft/Phi-4-mini-instruct`, pinned to
revision `cfbefacb99257ffa30c83adab238a50856ac3083`. The repository was
selected because it was publicly accessible in this environment, MIT licensed,
and documented a Transformers causal-LM loading path. Gemma 3 and Llama 3.2
were not usable candidates in this environment because their model-file
requests returned HTTP 401 access failures.

Before any model evaluation, a separate 48-case capability-gap screen was
frozen using benchmark fingerprint
`677d6e569c09031c2256161e01dcf48747794678c371948564db97b3e1328505`.
The screen reuses the disclosed E05 generator namespace with a new seed and is
not final E06 evidence. The rejection rule was frozen: reject only if diagnosis
exact, resolution exact, failure-mode macro F1, and strict JSON each reach 90%.

The screen did not reach semantic generation. At model load, the pinned
official dynamic `modeling_phi3.py` raised:

```text
ImportError: cannot import name 'LossKwargs' from 'transformers.utils'
```

No screen prediction, training run, adapter, or final E06 benchmark exists.
The failure is preserved in
`results/experiment_06/capability_gap/technical_failure.json`, including the
git SHA and benchmark hashes. G06 is therefore not passed, and the roadmap
must not proceed to E06 training or final-benchmark evaluation from this state.
This is a compatibility failure, not evidence of no capability gap.

## Compatibility recovery

The permitted single recovery attempt succeeded without changing the model,
revision, benchmark, prompt, scorer, or decoding. Native Transformers loading
with `trust_remote_code=false` loaded the tokenizer and quantized model, and a
non-benchmark prompt generated `OK.` The recovery record is persisted at
`results/experiment_06/compatibility_recovery.json`; attempt 0 remains the
original technical failure above. The 48-case capability screen is now
authorized to run exactly once under the recovered runtime.
