# CauseTune scope boundary

Laboratory Hardening v1 does not add:

- Web UI, serving, vLLM, model routing or an inference gateway;
- MCP, cloud orchestration or SSH management;
- GGUF/AWQ/GPTQ export platforms or an export matrix;
- a recipe zoo, automatic hyperparameter search or autopilot;
- automatic distillation or a trainer checklist for DPO/GRPO/KTO.

DPO/GRPO/KTO may be future scientific interventions, but CauseTune does not
aim to support many trainer families as a product feature. The repository owns
controlled offline specialization, evaluation isolation and evidence bundles.
