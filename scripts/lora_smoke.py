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

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"Loading {MODEL_ID} in NF4...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=quant_config,
    device_map={"": 0},
)

print("\nPreparing model for k-bit training...")

model = prepare_model_for_kbit_training(
    model,
    use_gradient_checkpointing=True,
)

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

print("\n--- TRAINABLE PARAMETERS ---")
model.print_trainable_parameters()

trainable = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

total = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Trainable: {trainable:,}")
print(f"Total:     {total:,}")
print(f"Trainable %: {100 * trainable / total:.4f}%")

print("\n--- MEMORY ---")
print(
    "Allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)
print(
    "Reserved:",
    round(torch.cuda.memory_reserved() / 1024**3, 2),
    "GB"
)
print(
    "Peak:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB"
)

print("\n--- LORA MODULE SAMPLE ---")

count = 0
for name, module in model.named_modules():
    if "lora_A" in name or "lora_B" in name:
        print(name)
        count += 1
        if count >= 10:
            break

print("\nLORA SMOKE: PASS")
