# Experiment 08 — Training-Efficiency Frontier Results

Status: `G08 PASS`.

E08 used only the two predeclared, already measured recipes. No semantic evaluation, checkpoint switching, or automatic search was launched.

| candidate | unique train examples | examples processed | supervised tokens | optimizer steps | wall-clock training | trainable params | E05 blind diagnosis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E02 original | 2,400 | 800 | 31,551 | 100 | 10,263.71 s | 33,030,144 | 98.3% |
| E04 selected | 600 | 800 | 31,551 | 100 | 3,441.39 s | 16,515,072 | 92.5% |

The E04 selected recipe reduces the unique training corpus by 75.0%, wall-clock training by 66.5%, and trainable parameters by 50.0%. The recorded run processed the same number of examples/tokens and optimizer steps, so those fields are not presented as reduced for E04.

Against the E02 reference on the same frozen E05 blind benchmark:

- diagnosis exact delta: `-5.83 pp`;
- failure-mode macro F1 delta: `-4.63 pp`;
- schema-validity delta: `-3.33 pp`;
- maximum failure-family diagnosis loss: recorded in `results/experiment_08/frontier.json`.

The predeclared quality tolerance was `-1 pp` for diagnosis exact and macro F1, `-5 pp` for any critical family, and `-5 pp` for major schema validity. E04 therefore fails the quality-preservation rule on fresh generalization. It is a cheaper negative trade-off, not a quality-preserving replacement for E02. The negative result remains visible rather than being absorbed into a revised tolerance.

E08 does not retroactively alter E04 selection, E05 scoring, or any completed training run.
