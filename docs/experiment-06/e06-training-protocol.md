# Experiment 06 — Cross-model training protocol

Status: `FROZEN_BEFORE_TRAINING`

The frozen 48-case capability screen completed under the recovered native Phi
runtime and found a capability gap. Training is therefore scientifically
justified. This protocol applies the selected E04 recipe methodology to
`microsoft/Phi-4-mini-instruct`, pinned to revision
`cfbefacb99257ffa30c83adab238a50856ac3083`.

The canonical contract is
`configs/experiment_06/e06-training.json`. Its data fingerprints and model
revision are immutable for this run.

## Primary intervention

There is no new optimization intervention in E06. The intervention is model
family: the non-Qwen Phi model is evaluated under the selected E04 data,
capacity, learning-rate, quantization, and validation methodology.

## Frozen controls

| Field | Frozen value |
| --- | --- |
| training data | E04-selected 25% subset, 600 examples |
| subset hash | `eaecc635921cb82219f4e6efc05a1387d76ae52695430f333c640b8f87728f56` |
| validation | E04 validation corpus, 288 examples |
| learning rate | `1e-4`, AdamW, no scheduler |
| LoRA rank / alpha | `8 / 32` |
| LoRA dropout | `0.0` |
| sequence / batch | 768 / microbatch 1 / accumulation 8 |
| effective batch | 8 |
| epochs / step ceiling | 2 / 150 optimizer steps |
| seed / shuffle | `20260941` / deterministic seeded shuffle |
| quantization | NF4, 4-bit, BF16 compute, double quantization |
| checkpoint selection | validation-only, diagnosis exact match, every 25 steps |
| early stopping | patience 3, min delta 0.005, validation every 25 steps |
| benchmark | `data/incident_diagnosis`, reporting-only; excluded from decisions |
| prompt / scorer | frozen `incident-evaluation-v1` / `incident-scorer-v1` |

The E04 alpha policy remains fixed alpha 32. No E05, E03, or future blind
benchmark is used for training or checkpoint selection.

## Architecture-specific LoRA scope

Native Transformers exposes these Phi linear projection suffixes in every
decoder layer:

```text
self_attn.qkv_proj
self_attn.o_proj
mlp.gate_up_proj
mlp.down_proj
```

The E06 target module policy is therefore exactly:
`qkv_proj`, `o_proj`, `gate_up_proj`, and `down_proj`. The fused QKV and fused
gate/up projections are explicit architecture differences from Qwen's
separate `q_proj`, `k_proj`, `v_proj`, `gate_proj`, and `up_proj` modules; this
is required for equivalent attention/MLP projection coverage and is not a
scientific variable.

The recovered runtime uses the native Transformers implementation with
`trust_remote_code=false`. Model identity, revision, weights, data, prompt,
scorer, decoding, and evidence boundaries remain unchanged from the frozen
E06 selection and capability-screen contracts.

## Gate before training

Training may start only after:

```text
capability-gap screen valid                 PASS
selected model and revision frozen          PASS
selected data and validation fingerprints   PASS
architecture target inspection              PASS
native model/LoRA attachment smoke          PASS
doctor with hardware                        PASS
benchmark excluded from decisions           PASS
```
