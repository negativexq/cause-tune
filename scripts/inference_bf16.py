import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-0.6B"

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

print(f"Loading {MODEL_ID} in BF16...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    device_map={"": 0},
)

messages = [
    {
        "role": "user",
        "content": "Explain LoRA in three short sentences."
    }
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

inputs = tokenizer(text, return_tensors="pt").to("cuda")

torch.cuda.synchronize()
start = time.perf_counter()

with torch.inference_mode():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

torch.cuda.synchronize()
elapsed = time.perf_counter() - start

generated = outputs[0][inputs.input_ids.shape[1]:]

print("\n--- RESPONSE ---")
print(tokenizer.decode(generated, skip_special_tokens=True))

print("\n--- METRICS ---")
print("Generated tokens:", generated.numel())
print("Generation time:", round(elapsed, 2), "s")
print("Tokens/sec:", round(generated.numel() / elapsed, 2))
print(
    "Peak VRAM:",
    round(torch.cuda.max_memory_allocated() / 1024**3, 2),
    "GB",
)
print("BF16 INFERENCE: PASS")
