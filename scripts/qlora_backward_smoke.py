import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

MODEL_ID = "Qwen/Qwen3-4B"

# --------------------------------------------------
# 1. Quantized frozen base model
# --------------------------------------------------

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
)

model.config.use_cache = False

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

# --------------------------------------------------
# 2. LoRA adapters
# --------------------------------------------------

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
)

model = get_peft_model(model, lora_config)
model.train()

trainable, total = model.get_nb_trainable_parameters()

print("\n--- PARAMS ---")
print(f"Trainable: {trainable:,}")
print(f"Logical total: {total:,}")
print(f"Trainable %: {100 * trainable / total:.4f}%")

# --------------------------------------------------
# 3. One training example
# --------------------------------------------------

messages = [
    {
        "role": "user",
        "content": (
            "Customer says: I was charged twice for order 83120. "
            "Classify the intent."
        ),
    },
    {
        "role": "assistant",
        "content": '{"intent":"duplicate_charge"}',
    },
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

batch = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=512,
)

batch = {k: v.to("cuda") for k, v in batch.items()}

# Smoke test only:
# for now loss is calculated on the whole sequence.
# Final SFT pipeline will mask user/system tokens.
labels = batch["input_ids"].clone()

# --------------------------------------------------
# 4. Optimizer contains ONLY trainable LoRA params
# --------------------------------------------------

optimizer = torch.optim.AdamW(
    (p for p in model.parameters() if p.requires_grad),
    lr=2e-4,
)

# Pick lora_B deliberately.
# Standard LoRA initializes B to zero, so B should
# receive a useful gradient on the first step.
tracked_name = None
tracked_param = None

for name, param in model.named_parameters():
    if "lora_B" in name and param.requires_grad:
        tracked_name = name
        tracked_param = param
        break

assert tracked_param is not None

before = tracked_param.detach().clone()

# Measure training memory separately from model loading.
baseline_vram = torch.cuda.memory_allocated()
torch.cuda.reset_peak_memory_stats()

# --------------------------------------------------
# 5. REAL TRAINING STEP
# --------------------------------------------------

optimizer.zero_grad(set_to_none=True)

outputs = model(
    **batch,
    labels=labels,
)

loss = outputs.loss

print("\n--- FORWARD ---")
print("Loss:", float(loss.detach()))

loss.backward()

grad = tracked_param.grad

assert grad is not None

grad_norm = float(grad.float().norm())

print("\n--- BACKWARD ---")
print("Tracked parameter:", tracked_name)
print("Gradient norm:", grad_norm)

optimizer.step()

torch.cuda.synchronize()

after = tracked_param.detach()

max_delta = float(
    (after.float() - before.float()).abs().max()
)

peak_vram = torch.cuda.max_memory_allocated()

# --------------------------------------------------
# 6. Results
# --------------------------------------------------

print("\n--- OPTIMIZER STEP ---")
print("Max LoRA weight delta:", max_delta)

print("\n--- MEMORY ---")
print(
    "Baseline VRAM:",
    round(baseline_vram / 1024**3, 2),
    "GB",
)
print(
    "Training peak:",
    round(peak_vram / 1024**3, 2),
    "GB",
)
print(
    "Training extra:",
    round((peak_vram - baseline_vram) / 1024**3, 2),
    "GB",
)

assert torch.isfinite(loss)
assert grad_norm > 0
assert max_delta > 0

print("\nQLORA BACKWARD SMOKE: PASS")
