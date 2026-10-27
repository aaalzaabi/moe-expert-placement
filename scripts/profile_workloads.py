
# --- added: run from project root; refuse to overwrite saved results without --force ---
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from moeplace.legacy import prepare as _prepare  # noqa: E402
_prepare(['outputs/workload_routing.csv', 'outputs/workload_timings.csv'], measured=True)
# ---------------------------------------------------------------------------------------

import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "models/OLMoE-1B-7B-0125-Instruct"
DEVICE = "mps"

# Start with six requests. Expand later.
workloads = [
    ("ai", "Explain mixture-of-experts models."),
    ("ai", "How does GPU parallelism work?"),
    ("coding", "Write a Python sorting function."),
    ("coding", "Explain how to optimize a database query."),
    ("general", "Why is the sky blue?"),
    ("general", "How do airplanes navigate?"),
]

Path("outputs").mkdir(exist_ok=True)

print("Loading OLMoE...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.float32,
    attn_implementation="eager",
).to(DEVICE).eval()

records = []
timings = []

for request_id, (category, prompt) in enumerate(workloads):
    print(f"\nProcessing {request_id + 1}/{len(workloads)}: {category}")

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(DEVICE)

    if DEVICE == "mps":
        torch.mps.synchronize()

    start = time.perf_counter()

    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_router_logits=True,
            use_cache=False,
        )

    if DEVICE == "mps":
        torch.mps.synchronize()

    elapsed = time.perf_counter() - start
    token_count = inputs["input_ids"].shape[1]

    timings.append({
        "request_id": request_id,
        "category": category,
        "input_tokens": token_count,
        "forward_seconds": elapsed,
    })

    assert len(outputs.router_logits) == model.config.num_hidden_layers

    for layer_id, logits in enumerate(outputs.router_logits):
        selected = torch.topk(
            logits.float(),
            k=model.config.num_experts_per_tok,
            dim=-1,
        ).indices

        counts = torch.bincount(
            selected.reshape(-1).cpu(),
            minlength=model.config.num_experts,
        )

        # Check each layer selected exactly 8 experts per token.
        assert int(counts.sum()) == (
            token_count * model.config.num_experts_per_tok
        )

        for expert_id, count in enumerate(counts.tolist()):
            records.append({
                "request_id": request_id,
                "category": category,
                "layer_id": layer_id,
                "expert_id": expert_id,
                "token_assignments": count,
                "input_tokens": token_count,
            })

    print(f"Input tokens: {token_count}")
    print(f"Forward time: {elapsed:.2f} seconds")

    # Append completed requests immediately.
    # This preserves progress if execution is interrupted.
    pd.DataFrame(records).to_csv(
        "outputs/workload_routing.csv",
        index=False,
    )
    pd.DataFrame(timings).to_csv(
        "outputs/workload_timings.csv",
        index=False,
    )

print("\nProfiling complete!")
print("Routing: outputs/workload_routing.csv")
print("Timings: outputs/workload_timings.csv")

