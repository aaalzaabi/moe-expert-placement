
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/expert_routing.csv'], measured=True)
# ---------------------------------------------------------------------------------------

import time
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/OLMoE-1B-7B-0125-Instruct"
DEVICE = "mps"

print("Loading OLMoE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    attn_implementation="eager",
).to(DEVICE).eval()

prompt = "Explain how GPU expert placement works."

inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

print("Running router profiling...")
start = time.perf_counter()

with torch.inference_mode():
    outputs = model(
        **inputs,
        output_router_logits=True,
        use_cache=False,
    )

torch.mps.synchronize()
elapsed = time.perf_counter() - start

records = []

for layer_id, logits in enumerate(outputs.router_logits):
    if logits is None:
        continue

    top_experts = torch.topk(
        logits.float(),
        k=model.config.num_experts_per_tok,
        dim=-1
    ).indices

    counts = torch.bincount(
        top_experts.reshape(-1).cpu(),
        minlength=model.config.num_experts
    )

    for expert_id, count in enumerate(counts.tolist()):
        records.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "token_assignments": count
        })

df = pd.DataFrame(records)
df.to_csv("outputs/expert_routing.csv", index=False)

print("\nProfiling completed")
print(f"Forward-pass time: {elapsed:.2f} seconds")
print(f"Layers profiled: {df.layer_id.nunique()}")
print("Routing saved to outputs/expert_routing.csv")

print("\nTop 10 most selected layer-specific experts:")
print(
    df.sort_values("token_assignments", ascending=False)
      .head(10)
      .to_string(index=False)
)

