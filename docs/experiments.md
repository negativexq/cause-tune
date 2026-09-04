# Lab Experiment 01 — Controlled QLoRA Specialization

## Goal

Experiment 01 validated CauseTune’s hands-on fine-tuning laboratory workflow: establish a frozen baseline, fine-tune once under controlled settings, inspect failures, and isolate a training-order cause with one variable. It was intentionally narrow and is not the final challenge task for the laboratory.

## Base capability

M4 created the realistic synthetic customer-support intent benchmark and evaluated untouched Qwen/Qwen3-4B under the frozen evaluation contract. Base accuracy was 88.0% validation, 83.6% ID, 76.8% HARD, and 79.6% OOD.

## First realistic fine-tune

M5 trained QLoRA on the 2,000-example train split with the verified Qwen3-4B NF4/BF16 configuration. The run used `shuffle=False`, preserving ten class-contiguous blocks of 200 examples.

## Failure

M5 accuracy fell to 26.8% validation, 24.8% ID, 23.6% HARD, and 20.0% OOD despite training loss collapsing. JSON/schema compliance stayed near 100%, so this was a semantic failure rather than a format-learning success.

## Root-cause audit

The audit found:

- `shuffle=False` preserved class-contiguous order.
- Microbatch `1` and gradient accumulation `8` made 250/250 optimizer windows single-class.
- The final 25 optimizer steps contained only `wrong_item` examples.
- Loss and predictions showed catastrophic recency/forgetting behavior.

This is why a balanced dataset did not produce balanced optimization.

## Controlled intervention

M6 changed exactly one training-affecting variable: deterministic seeded shuffle with seed `42`. Dataset contents, labels, model, quantization, LoRA, optimizer settings, masking, and frozen evaluation contract remained fixed.

## Result

M6 produced 0/250 single-class windows and 250/250 mixed-class windows, with no terminal `wrong_item` block. Accuracy recovered to 99.2% validation, 97.2% ID, 96.0% HARD, and 92.0% OOD. The key `wrong_item` collapse pairs fell from M5 to M6 as follows: `duplicate_charge` 100→0, `fraud_suspected` 100→1, `order_missing` 100→6, `refund` 99→1, and `cancel_order` 98→2. No replacement single-class collapse occurred.

## What we learned

Experiment 01 established practical lessons about QLoRA mechanics, assistant-only loss, quantized parameter accounting, frozen evaluation, optimizer-window composition, deterministic ordering, failure analysis, held-out evaluation, and fresh adapter reload. Its strongest lesson is that train loss alone is insufficient evidence of useful specialization. Failed runs should be retained as evidence, and future tasks should begin with capability-gap screening.

This remains a learning experiment on synthetic data, not a production benchmark.
