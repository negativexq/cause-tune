# CI compatibility gates

CauseTune keeps two intentionally separate CI jobs:

`cpu-safe` compiles the package and runs the complete dependency-light test
suite. It must remain safe on ordinary CPU runners.

`training-smoke` uses the current Transformers/PEFT/Accelerate stack with the
public `hf-internal-testing/tiny-random-GPT2` asset. It runs on CPU, performs
two real Trainer optimizer steps with assistant-only labels, saves a LoRA
adapter, reloads it into a fresh base model and generates output. The model
asset is cached by the workflow, but the job remains isolated from scientific
GPU training and does not claim quality, speed, VRAM or hardware support.
