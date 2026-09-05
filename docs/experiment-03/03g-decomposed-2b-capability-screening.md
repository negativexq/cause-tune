# Experiment 03G — decomposed untouched Qwen3.5-2B capability screening

Status: complete. This is a base-model screen, not a training run.

## Scientific purpose

03E.2 showed that untouched Qwen3.5-2B was `BASE_TOO_WEAK` for the full
agentic task: tool selection, evidence acquisition, stopping, and joint
category/root-cause/object diagnosis. 03G asks a narrower question: does the
same model show a measurable, non-saturated signal on bounded diagnosis heads
when tool policy and termination are removed?

This is a new formulation, not a reinterpretation of 03E.2. The screen uses
the immutable Cloud-OpsBench revision
`03c415e5709297432282fbbfd499f1bca0f8c347`, the frozen 03D split, the 03F
decomposition fingerprint
`249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9`, and the
03F.1 evidence representation fingerprint
`b676b10a17d690f3796e237e942c340f2722a1aa5bcd97af1b43de1c0e972215`.
Only the 57-case `VALIDATION_SCREEN` was model-facing. `VALIDATION_SELECTION`
and `TEST` were not used.

## Frozen model and runtime

The only model was `Qwen/Qwen3.5-2B`, revision
`15852e8c16360a2fea060d615a32b45270f8a8fc`. Inference used local NF4 4-bit
quantization, BF16 compute, double quantization, greedy decoding,
`enable_thinking=False`, and `max_new_tokens=256`. Prompts, schemas,
vocabularies, evidence inputs, parser, and generation contract were frozen
before model loading. There were 228 generations: 57 each for Tasks A, B,
C Stage 2, and D; Task C Stage 1 reused the persisted Task A outputs.

The runtime was an NVIDIA GeForce RTX 5070 Laptop GPU with 8,546,484,224
reported physical VRAM bytes. Peak allocated/reserved VRAM was
3,552,660,992 / 4,766,826,496 bytes. Model loading took 6.02 seconds;
total generation plus load time was 391.65 seconds. There were zero OOMs,
zero context overflows, and zero truncations. Software versions were torch
2.14.0+cu130, transformers 5.16.1, bitsandbytes 0.50.2, and CUDA 13.0.

## Frozen task contracts

* **Task A — category classification:** evidence to exactly one of the eight
  native categories, output `{ "fault_category": ... }`.
* **Task B — `ORACLE_CATEGORY_ROOT_CAUSE`:** evidence plus the authoritative
  category to a root cause from that category's native candidate set, output
  `{ "root_cause": ... }`. This is explicitly not end-to-end RCA.
* **Task C — hierarchical self-predicted RCA:** persisted Task A category,
  followed by a new Stage 2 generation using only that predicted category's
  candidate set. The authoritative category was never substituted.
* **Task D — fault-object resolution:** evidence to an open resource identity,
  with strict and previously frozen deterministic object normalization.

No tools, process labels, target labels, golden answers, few-shot examples,
or case-specific repair instructions were exposed. No output was repaired by
another model.

## Baselines and results

The screen has one case per root cause. Category counts have a tie: both
`Runtime_Fault` and `Scheduling_Fault` occur 12/57, so the majority baseline
is 21.05%; the uniform category expectation is 12.50%. The frozen
Task-B uniform-within-category expectation is 14.04%. The frozen hierarchical
random joint expectation is 1.75%.

| Task | Strict result | Schema / enum | Frozen comparison |
|---|---:|---:|---:|
| A category | 14/57 (24.56%) | 57/57, 57/57 | above 12.50% uniform, only marginally above 21.05% majority |
| B oracle-category root cause | 15/57 (26.32%) | 57/57, 57/57 | above 14.04% uniform-within-category |
| C Stage 1 category | 14/57 (24.56%) | 57/57, 57/57 | same category signal as A |
| C hierarchical joint | 4/57 (7.02%) | 57/57 schema, 55/57 enum | above 1.75% random joint, but not a reliable composed protocol |
| D object, strict | 0/57 | 57/57, 57/57 | no fixed-vocabulary baseline |
| D object, normalized | 0/57 | 57/57, 57/57 | no normalized successes |

For Task C, 43 failures were caused by a wrong Stage 1 category and 10 were
within-category discrimination failures despite a correct category. Among the
14 correct Stage 1 categories, Stage 2 was exact for 4 (28.57%). Two Stage 2
outputs were outside the predicted category's candidate set. This makes the
category bottleneck, rather than the oracle-category root-cause head, the
limiting component of the composed pipeline.

Task B descriptive accuracy by candidate-set size was: size 4, 2/8; size 5,
1/5; size 6, 5/12; size 8, 1/8; size 12, 6/24. These are descriptive slices
of a one-case-per-root-cause screen, not stable per-label estimates.

## Evidence and context slices

The 03F.1 FULL/PARTIAL/ZERO labels are analysis slices only. They do not mean
that a ZERO case contains no diagnostic information: 03F.1 found that most
original zero matches arose from process labels without mechanically
matchable patterns. Results were:

| Slice | Cases | A exact | B exact | C joint | D normalized |
|---|---:|---:|---:|---:|---:|
| FULL | 22 | 2/22 | 6/22 | 1/22 | 0/22 |
| PARTIAL | 11 | 4/11 | 5/11 | 2/11 | 0/11 |
| ZERO | 24 | 8/24 | 4/24 | 1/24 | 0/24 |

The selected R2 package remained well within the model context. Exact local
tokenizer distributions (chat-template input) were:

| Input | min | median | p75 | p90 | p95 | p99 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 716 | 2,784 | 4,698 | 8,404 | 9,874 | 11,949 | 13,321 |
| B | 721 | 2,844 | 4,705 | 8,427 | 9,934 | 12,009 | 13,381 |
| C Stage 1 | 715 | 2,783 | 4,697 | 8,403 | 9,873 | 11,948 | 13,320 |
| C Stage 2 | 733 | 2,855 | 4,716 | 8,438 | 9,908 | 11,983 | 13,339 |
| D | 667 | 2,735 | 4,649 | 8,355 | 9,825 | 11,900 | 13,272 |

No truncation was used.

## TRAIN support relationship

TRAIN support for the 57 native labels ranges from 1 to 40, with median 8.
The ten labels below the frozen low-support threshold (TRAIN count < 3) are
retained as descriptive context rather than augmented or oversampled:

`code_memory_leak`, `namespace_storage_quota_exceeded`, `node_network_delay`,
`node_network_packet_loss`, `pv_binding_occupied`,
`pvc_access_mode_mismatch`, `pvc_capacity_mismatch`, `pvc_selector_mismatch`,
`pvc_storage_class_mismatch`, and `volume_mount_permission_denied`.

The complete case-level join, including each label's TRAIN count and Task B/C
result, is in `train_support_cross_analysis.json`. The screen is too small for
statistical claims about individual labels.

## Selection decision

The frozen 03F policy requires a candidate to be materially above its trivial
baseline, non-saturated, protocol-reliable, locally feasible, and free of an
environment confound. Applying that qualitative policy yields:

**`SELECT_TASK_B_FOR_QLORA`**

Task B is the first credible specialization target: it has a clean 57/57
schema/enum protocol, a clear oracle-category within-class signal, substantial
remaining error, and feasible local execution. Task A is not selected because
its result is only marginally above the tied majority baseline. Task C is not
preferred because category errors dominate the composed result and two Stage 2
outputs violated the predicted-category candidate set. Task D has no exact
object successes and remains secondary.

This selection does not start QLoRA. It only makes Task B a future 03H
candidate. No model was trained, no adapter was created, and no weights were
modified in 03G.

The historical conclusion remains unchanged: 03E.2 measured the full
agentic RCA task and found `BASE_TOO_WEAK`. 03G shows that decomposition
exposes a narrower, potentially learnable oracle-category root-cause signal;
it does not claim that the full agentic system is solved.

## Limitations and next boundary

Task B supplies the category oracle and therefore must not be reported as
end-to-end RCA. Task C's weak composition still leaves category recognition,
candidate restriction, and root-cause discrimination coupled. Fault-object
resolution is not yet a suitable primary specialization target. The
VALIDATION_SCREEN has one case per root cause, so per-label and slice results
are descriptive. 03G does not evaluate generalization to TEST or to RCAEval.

The next possible experiment is a separately versioned 03H QLoRA run for the
frozen Task B contract. It was not started automatically and remains subject
to the project’s training authorization and a new immutable experiment
namespace.
