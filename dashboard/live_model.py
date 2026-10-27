"""Run the real OLMoE model on one user prompt and record its routing (used by app_pages/my_prompt.py).

This is the only part of the dashboard that loads the model, and it does so only when the user presses
the button on that page. torch and transformers are imported inside the functions, so importing this
module stays cheap and the rest of the dashboard never touches them. Nothing is written to disk.

The routing table has the same schema as outputs/workload_routing.csv, so the placement and visual
simulator code can use it unchanged.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from moeplace import config as C
from moeplace.paths import MODEL_DIR

PROMPT_ID = 100  # request_id given to the user's prompt (the saved questions are 0-5)
MAX_TOKENS = 256


@dataclass
class Loaded:
    model: object
    tokenizer: object
    device: str
    dtype: str
    load_seconds: float


@dataclass
class Routed:
    prompt: str
    tokens: list[str]
    routing: pd.DataFrame          # request_id, category, layer_id, expert_id, token_assignments, input_tokens
    choices: np.ndarray            # [layer, token, k] expert ids picked by the router
    forward_seconds: float
    device: str
    dtype: str


def checkpoint_present() -> bool:
    return (MODEL_DIR / "config.json").exists()


def load(dtype: str = "bfloat16") -> Loaded:
    """Load the local checkpoint (never downloads). BF16 halves memory versus the FP32 used in Phase 2."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not checkpoint_present():
        raise FileNotFoundError(f"Checkpoint not found at {MODEL_DIR}. The dashboard never downloads the model.")
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=getattr(torch, dtype), attn_implementation="eager",
                                                 local_files_only=True).to(device).eval()
    cfg = model.config
    if (cfg.num_experts_per_tok, cfg.num_experts, cfg.num_hidden_layers) != (C.TOP_K, C.EXPERTS_PER_LAYER, C.NUM_LAYERS):
        raise RuntimeError("Unexpected model configuration")
    return Loaded(model, tok, device, dtype, time.perf_counter() - t0)


def route(m: Loaded, prompt: str) -> Routed:
    """One prefill forward pass; returns which 8 of 64 experts each token used in each layer."""
    import torch

    text = prompt.strip()
    if not text:
        raise ValueError("The prompt is empty.")
    inputs = m.tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS).to(m.device)
    n_tok = int(inputs["input_ids"].shape[1])

    def sync():
        if m.device == "mps":
            torch.mps.synchronize()
        elif m.device == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = m.model(**inputs, output_router_logits=True, use_cache=False)
    sync()
    seconds = time.perf_counter() - t0

    if len(out.router_logits) != C.NUM_LAYERS:
        raise RuntimeError(f"Expected {C.NUM_LAYERS} router outputs, got {len(out.router_logits)}")
    choices = np.stack([torch.topk(lg.float(), k=C.TOP_K, dim=-1).indices.reshape(n_tok, C.TOP_K).cpu().numpy()
                        for lg in out.router_logits])
    rows = []
    for layer_id in range(C.NUM_LAYERS):
        counts = np.bincount(choices[layer_id].reshape(-1), minlength=C.EXPERTS_PER_LAYER)
        if int(counts.sum()) != n_tok * C.TOP_K:
            raise RuntimeError(f"Layer {layer_id}: top-{C.TOP_K} invariant violated")
        rows += [{"request_id": PROMPT_ID, "category": "your prompt", "layer_id": layer_id, "expert_id": e,
                  "token_assignments": int(c), "input_tokens": n_tok} for e, c in enumerate(counts)]
    tokens = [m.tokenizer.decode([t]) for t in inputs["input_ids"][0].tolist()]
    return Routed(text, tokens, pd.DataFrame(rows), choices, seconds, m.device, m.dtype)


def unload():
    """Free accelerator memory after the cached model has been dropped."""
    import gc

    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 - best effort only
        pass
