
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare([])
# ---------------------------------------------------------------------------------------

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/OLMoE-1B-7B-0125-Instruct"

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("Using device:", device)
print("Loading OLMoE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float32,
    attn_implementation="eager",
).to(device).eval()

messages = [
    {
        "role": "user",
        "content": "Explain mixture-of-experts in two sentences."
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

start = time.perf_counter()

with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=40,
        do_sample=False
    )

elapsed = time.perf_counter() - start

new_tokens = output[0][inputs["input_ids"].shape[-1]:]

print("\nMODEL RESPONSE:")
print(tokenizer.decode(new_tokens, skip_special_tokens=True))
print(f"\nGeneration time: {elapsed:.2f} seconds")

