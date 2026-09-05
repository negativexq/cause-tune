# Experiment 03F.1 — Evidence Package Sufficiency and Diagnostic Input Integrity Gate

Status: `READY_FOR_03G`

03F remains immutable. Its decomposition fingerprint is
`249254d440008c3d674171aa023b27efa83a89f598a021afac3c28655533dda9`.
03F.1 creates a separate evidence-representation audit and does not replace
the 03F contract or its 244 FULL / 79 PARTIAL / 310 ZERO process-coverage
counts.

## Question and scope

The 80.55% aggregate retention in 03F was not sufficient evidence that the
diagnosis inputs were valid: case-level coverage was 244 FULL, 79 PARTIAL,
and 310 ZERO.  This audit asks whether ZERO means packaging failure, matcher
failure, path omission, unavailable replay, or simply that the process-label
annotation has no matchable evidence pattern.

No model was loaded. No model was trained. No GPU or provider was used. No
QLoRA was performed. No TEST inference occurred.

## ZERO and PARTIAL findings

All 310 original ZERO cases were audited individually. The deterministic
reason counts are:

| reason | cases |
|---|---:|
| `PROCESS_LABEL_NO_MATCHABLE_PATTERNS` | 301 |
| `MATCHER_SCHEMA_GAP` | 9 |

The 301 cases have process milestones/admissible tools but no evidence
patterns that the matcher can evaluate. They are annotation-semantic gaps,
not evidence packaging loss. The nine remaining cases contain a Kubernetes
Pending observation, but the native regex ends in an impossible word boundary
after `>` (`<none>\b`). The audit records that as a matcher/schema defect and
does not use a silent repair for scoring.

The 79 PARTIAL cases divide into:

| reason | cases |
|---|---:|
| `PROCESS_EVIDENCE_ON_ALTERNATE_PATH` | 28 |
| `SOURCE_ARTIFACT_AVAILABLE_BUT_NOT_PACKAGED` | 45 |
| `PACKAGING_OR_BUDGET_LOSS` | 6 |

Thus the aggregate retention problem is not one cause. Path selection and
replay boundaries account for recoverable omissions, while the large ZERO
population is primarily an upstream process-label coverage limitation.

## Matcher and path audit

The matcher was checked against immutable process patterns, both golden trace
outputs, source modality files, and the final package. Allowed canonicalization
is limited to Unicode NFKC, case folding, and whitespace collapse for literal
and code-snippet comparisons. There is no fuzzy matching, synonym table,
embedding, or LLM judge. Across 2,154 patterns, 2,080 had a strict source-text
match and nine exposed the impossible-regex-boundary defect.

The frozen 03F R1 view is path1-preferred and has 1,735 / 2,154 retained
patterns (80.55%): 244 FULL, 79 PARTIAL, 310 ZERO. Offline path comparison:

| view | FULL | PARTIAL | ZERO | retained | retention |
|---|---:|---:|---:|---:|---:|
| PATH1_ONLY | 248 | 75 | 310 | 1,755 | 81.48% |
| PATH2_ONLY | 223 | 100 | 310 | 1,735 | 80.55% |
| DETERMINISTIC_UNION | 249 | 74 | 310 | 1,910 | 88.67% |

The union is target-blind: path1 observations come first, exact `(tool,
observation)` duplicates are removed, and unique path2 observations follow.
It removes 2,216 duplicate observations and adds 589 observations over the
path1 view. It does not use targets, process labels, or future model scores.

## Representation candidates and selection

Three candidates were frozen before any future model execution:

* R1: existing 03F single-path package.
* R2: deterministic executable path1 + path2 union.
* R3: R2 plus fixed alert, Kubernetes-state, logs, metrics, and code source
  channels with fixed per-channel caps.

At the 30,000-character budget, R1 is 244 / 79 / 310 with 1,735 retained
patterns; R2 is 245 / 78 / 310 with 1,889 retained patterns after package
budgeting; R3 is 241 / 82 / 310 with 1,868 retained patterns. R3 nearly fills
the budget on most cases and is retained as a future ablation rather than
selected. R2 is the simplest evidence-justified correction to the observed
alternate-path omission.

The selected representation is `R2_DETERMINISTIC_PATH_UNION` at 30,000
characters. All 633 TRAIN/VALIDATION packages are non-empty. The selected
package size distribution is chars: min 1,027, median 8,167, p75 16,613,
p90 27,951, p95 29,668, p99 30,000, max 30,000. Only 27 packages are
truncated by the deterministic equal observation cap.

This is smaller than the previous 03D.1 full-history view: the median is
8,167 versus 10,524 characters and the maximum is 30,000 versus 150,042.
The audit records both character and byte distributions; exact tokenizer
counts were intentionally not loaded because this gate forbids loading Qwen.

Budget sensitivity was measured at 15K, 20K, 30K, and 40K. R2 retains 310
ZERO cases at 30K and 40K; the remaining ZERO cases are therefore not
budget-driven. The detailed distributions are in `budget_analysis.json`.

## Category and screen integrity

Coverage is structurally uneven as a process-label proxy. For example,
Scheduling has 130 FULL / 6 PARTIAL / 0 ZERO, while Performance has 0 FULL /
0 PARTIAL / 65 ZERO because its process labels contain no matchable patterns.
Trainticket has 3 FULL / 8 PARTIAL / 164 ZERO, versus boutique 242 / 70 / 146.
These are reported as annotation coverage bias, not automatically as
diagnosis-input insufficiency.

The 57-case VALIDATION_SCREEN membership is unchanged. Under R2 at 30K it has
22 FULL, 11 PARTIAL, and 24 ZERO process-pattern cases. Every screen case has
non-empty executable source evidence and there are zero clear package
failures. The 24 ZERO cases are 23 no-pattern annotations and one malformed
regex. Process-label coverage is a proxy designed for agentic process
evaluation; it is not a complete definition of static diagnosis sufficiency.

## Selected 03G boundary

The final 03F.1 gate is `READY_FOR_03G`: the ZERO causes are classified, the
matcher is valid apart from explicitly recorded upstream regex defects, R2 is
deterministic and replayable, the screen is unchanged and non-empty, and no
target-guided packaging was used. This does not claim that a future model will
succeed; it means an untouched decomposed screen can now distinguish model
behavior from the previously identified package/path defects.

03G remains a future task and is not implemented here. TEST remains sealed.
