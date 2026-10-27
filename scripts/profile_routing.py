"""Collect NEW real OLMoE routing measurements without touching the original files.

This is the explicit, expensive step: it loads the 13 GB checkpoint and runs one
forward pass per prompt. It is never triggered by the dashboard.

Output: outputs/measurements/<name>-<UTC timestamp>/
    workload_routing.csv  (same schema as outputs/workload_routing.csv)
    workload_timings.csv  (same schema + warm-up flag)
    prompts.json, metadata.json

Examples
  python scripts/profile_routing.py --dry-run                      # validate setup, load nothing
  python scripts/profile_routing.py --prompt-set original          # re-measure the 6 Phase 2 prompts
  python scripts/profile_routing.py --prompt-set extended --warmup 1
  python scripts/profile_routing.py --prompts my_prompts.json      # [{"category": "...", "prompt": "..."}]
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moeplace import config as C  # noqa: E402
from moeplace.paths import MEASUREMENTS, MODEL_DIR  # noqa: E402

ORIGINAL = [
    ("ai", "Explain mixture-of-experts models."),
    ("ai", "How does GPU parallelism work?"),
    ("coding", "Write a Python sorting function."),
    ("coding", "Explain how to optimize a database query."),
    ("general", "Why is the sky blue?"),
    ("general", "How do airplanes navigate?"),
]

EXTENDED = ORIGINAL + [
    ("ai", "What is the difference between pretraining and fine-tuning a language model?"),
    ("ai", "Describe how attention works in a transformer."),
    ("ai", "Why do large models need tensor parallelism?"),
    ("ai", "Explain what a learning rate schedule does."),
    ("ai", "How does quantization reduce inference memory?"),
    ("ai", "What problems does a router solve in a sparse model?"),
    ("coding", "Write a function that checks whether a string is a palindrome."),
    ("coding", "How do I read a CSV file with pandas and drop missing values?"),
    ("coding", "Explain the difference between a list and a tuple in Python."),
    ("coding", "Write a SQL query that returns the top five customers by revenue."),
    ("coding", "How do I handle exceptions in JavaScript async functions?"),
    ("coding", "Implement binary search in C++."),
    ("general", "What causes the seasons on Earth?"),
    ("general", "Give three tips for writing a clear email."),
    ("general", "How do vaccines train the immune system?"),
    ("general", "Why do cities have higher temperatures than rural areas?"),
    ("general", "Summarize the water cycle in simple terms."),
    ("general", "What should I pack for a weekend hiking trip?"),
]


def load_prompts(args) -> list[tuple[str, str]]:
    if args.prompts:
        items = json.loads(Path(args.prompts).read_text())
        prompts = [(str(i["category"]), str(i["prompt"])) for i in items]
    else:
        prompts = ORIGINAL if args.prompt_set == "original" else EXTENDED
    if not prompts:
        sys.exit("No prompts to profile.")
    for cat, text in prompts:
        if not text.strip() or not cat.strip():
            sys.exit(f"Empty prompt/category in {cat!r}: {text!r}")
    return prompts


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--prompt-set", choices=["original", "extended"], default="extended")
    p.add_argument("--prompts", help="JSON file: list of {category, prompt}")
    p.add_argument("--name", default="routing")
    p.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                   help="float32 matches the original Phase 2 measurements")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap prompts with the chat template (Phase 2 did NOT)")
    p.add_argument("--warmup", type=int, default=1, help="unrecorded warm-up passes before timing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dry-run", action="store_true", help="validate prompts and paths only; load nothing")
    args = p.parse_args()

    prompts = load_prompts(args)
    if not (MODEL_DIR / "config.json").exists():
        sys.exit(f"Checkpoint not found at {MODEL_DIR}. This script never downloads the model.")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = MEASUREMENTS / f"{args.name}-{stamp}"
    print(f"{len(prompts)} prompts; output -> {out}")
    if args.dry_run:
        print("Dry run: prompts and checkpoint path OK. Nothing loaded, nothing written.")
        return 0

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)

    def sync():
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()

    out.mkdir(parents=True, exist_ok=False)
    (out / "prompts.json").write_text(json.dumps([{"request_id": i, "category": c, "prompt": t}
                                                  for i, (c, t) in enumerate(prompts)], indent=2))
    print(f"Loading OLMoE on {device} ({args.dtype})...")
    t_load = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=dtype, attn_implementation="eager").to(device).eval()
    load_s = time.perf_counter() - t_load
    k, n_exp, n_layers = model.config.num_experts_per_tok, model.config.num_experts, model.config.num_hidden_layers
    assert (k, n_exp, n_layers) == (C.TOP_K, C.EXPERTS_PER_LAYER, C.NUM_LAYERS), "Unexpected model configuration"

    def encode(text):
        if args.chat_template:
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
        return tok(text, return_tensors="pt").to(device)

    def forward(inputs):
        with torch.inference_mode():
            return model(**inputs, output_router_logits=True, use_cache=False)

    for i in range(args.warmup):
        print(f"Warm-up {i + 1}/{args.warmup}")
        forward(encode(prompts[0][1]))
        sync()

    records, timings = [], []
    for rid, (category, text) in enumerate(prompts):
        inputs = encode(text)
        sync()
        start = time.perf_counter()
        outputs = forward(inputs)
        sync()
        elapsed = time.perf_counter() - start
        n_tok = int(inputs["input_ids"].shape[1])
        if len(outputs.router_logits) != n_layers:
            raise RuntimeError(f"Expected {n_layers} router outputs, got {len(outputs.router_logits)}")
        for layer_id, logits in enumerate(outputs.router_logits):
            sel = torch.topk(logits.float(), k=k, dim=-1).indices
            counts = torch.bincount(sel.reshape(-1).cpu(), minlength=n_exp)
            if int(counts.sum()) != n_tok * k:
                raise RuntimeError(f"Request {rid} layer {layer_id}: top-{k} invariant violated")
            for expert_id, count in enumerate(counts.tolist()):
                records.append({"request_id": rid, "category": category, "layer_id": layer_id,
                                "expert_id": expert_id, "token_assignments": count, "input_tokens": n_tok})
        timings.append({"request_id": rid, "category": category, "input_tokens": n_tok,
                        "forward_seconds": elapsed, "after_warmup": args.warmup > 0})
        print(f"[{rid + 1}/{len(prompts)}] {category:8s} tokens={n_tok:3d} forward={elapsed:.2f}s")
        pd.DataFrame(records).to_csv(out / "workload_routing.csv", index=False)  # incremental save
        pd.DataFrame(timings).to_csv(out / "workload_timings.csv", index=False)

    (out / "metadata.json").write_text(json.dumps({
        "kind": "measured", "model": C.MODEL_ID, "device": device, "dtype": args.dtype,
        "chat_template": args.chat_template, "warmup_passes": args.warmup, "seed": args.seed,
        "model_load_seconds": load_s, "prompts": len(prompts), "created_utc": stamp,
        "torch": torch.__version__, "platform": platform.platform(),
        "note": "Prefill-only routing (no generated tokens). Timing is Apple-silicon single-device, not multi-GPU.",
    }, indent=2))
    print(f"Done. Saved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
