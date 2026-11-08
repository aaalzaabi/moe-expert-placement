"""REAL expert-parallel OLMoE inference across several GPUs, following the project's placement plans.

This is the explicit, expensive step that the simulator only models. For each placement strategy it:
  1. physically moves every one of the 1,024 experts (layer, expert) to the device the plan assigns;
  2. runs the real prompts: router on the gateway GPU, each token's hidden state sent once per layer to every GPU
     that holds one of its 8 chosen experts, expert MLPs computed there, results sent back and combined;
  3. measures wall-clock latency per prompt, per-GPU expert compute time (CUDA events), tokens and bytes moved;
  4. between strategies, times the real re-layout (copying experts between GPUs = a real migration).

Correctness: --check compares the final logits with the unmodified single-device model on the same prompts.

Nothing is downloaded; the local checkpoint is required. Original results are never touched:
output goes to outputs/real_runs/<name>-<UTC timestamp>/ (runs.csv, device_stats.csv, relayout.csv,
plan.csv, check.json, metadata.json).

Examples
  python scripts/run_real_expert_parallel.py --dry-run                 # plan only, loads nothing
  python scripts/run_real_expert_parallel.py --check                   # all visible CUDA GPUs
  python scripts/run_real_expert_parallel.py --devices cuda:0,cuda:1 --speeds equal
  python scripts/run_real_expert_parallel.py --devices mps,cpu --repeats 1   # local smoke test (not a benchmark)

Limitations (also written to metadata.json): prefill only, batch size 1, a simple dispatch loop (asynchronous
launches let GPUs overlap, but there are no custom streams or fused all-to-all), attention and router always on
the gateway GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from moeplace import config as C  # noqa: E402
from moeplace import data as D  # noqa: E402
from moeplace import placement as P  # noqa: E402
from moeplace.paths import MODEL_DIR, OUTPUTS  # noqa: E402

PROMPTS = [
    ("ai", "Explain mixture-of-experts models."),
    ("ai", "How does GPU parallelism work?"),
    ("coding", "Write a Python sorting function."),
    ("coding", "Explain how to optimize a database query."),
    ("general", "Why is the sky blue?"),
    ("general", "How do airplanes navigate?"),
]
LIMITATIONS = ("Prefill only, batch size 1. Simple per-layer dispatch loop: kernel launches are asynchronous so GPUs "
               "can overlap, but there are no custom streams or fused all-to-all. Attention, router, embeddings and "
               "LM head stay on the gateway GPU. Speeds used by the planner are from a short micro-benchmark or "
               "assumed equal, as recorded below.")


# ------------------------------------------------------------------ planning (no torch)
def make_plan(n_devices: int, speeds: list[float], gateway: int):
    cfg = C.DEFAULT.with_(gpu_speeds=tuple(speeds), placement_capacity_slots=math.ceil(C.TOTAL_EXPERT_SLOTS / n_devices),
                          staging_capacity_slots=2 * math.ceil(C.TOTAL_EXPERT_SLOTS / n_devices), gateway_gpu=gateway)
    r = D.load_routing()
    pl = P.plan(D.demand(r, cfg.train_ids), cfg, eval_demand=D.demand(r, cfg.test_ids))
    return cfg, pl


# ------------------------------------------------------------------ torch helpers
def sync_all(torch, devices):
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)
        elif d.type == "mps":
            torch.mps.synchronize()


def measure_speeds(torch, devices, dtype, hidden=2048, inter=1024, tokens=64, iters=30):
    """Relative expert-MLP throughput per device (1.0 = slowest). A short micro-benchmark, not a full profile."""
    times = []
    for d in devices:
        g = torch.Generator(device="cpu").manual_seed(0)
        w1, w2, w3 = (torch.randn(s, generator=g).to(d, dtype) * 0.02 for s in ((inter, hidden), (inter, hidden),
                                                                               (hidden, inter)))
        x = torch.randn(tokens, hidden, generator=g).to(d, dtype)
        f = lambda: (torch.nn.functional.silu(x @ w1.T) * (x @ w2.T)) @ w3.T  # noqa: E731
        for _ in range(5):
            f()
        sync_all(torch, [d])
        t0 = time.perf_counter()
        for _ in range(iters):
            f()
        sync_all(torch, [d])
        times.append((time.perf_counter() - t0) / iters)
    slowest = max(times)
    return [slowest / t for t in times], times


def place_experts(model, layout, devices):
    """Move each expert module to its planned device. Returns the number of experts moved and bytes copied."""
    moved = nbytes = 0
    for li, layer in enumerate(model.model.layers):
        for e, expert in enumerate(layer.mlp.experts):
            dev = devices[layout[(li, e)]]
            p = next(expert.parameters())
            if p.device != dev:
                nbytes += sum(q.numel() * q.element_size() for q in expert.parameters())
                expert.to(dev)
                moved += 1
    return moved, nbytes


def move_rest_to(model, gateway):
    for p in model.parameters():
        if p.device.type == "cpu" and gateway.type != "cpu":
            p.data = p.data.to(gateway)
    for name, b in model.named_buffers():
        if b.device.type == "cpu" and gateway.type != "cpu":
            mod = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
            setattr(mod, name.rsplit(".", 1)[-1], b.to(gateway))


class Recorder:
    """Per-forward bookkeeping: tokens/bytes per (layer, device) and CUDA-event compute windows."""

    def __init__(self, torch, devices):
        self.torch, self.devices, self.rows, self.events = torch, devices, [], []

    def reset(self):
        self.rows, self.events = [], []

    def compute_ms(self):
        out = {}
        for (layer, di), (start, end) in self.events:
            out[(layer, di)] = out.get((layer, di), 0.0) + start.elapsed_time(end)
        return out


def install_expert_parallel(model, devices, gateway_idx, rec):
    """Replace each MoE block's forward with a dispatch/compute/combine across devices.

    Numerically the same computation as transformers' OlmoeSparseMoeBlock.forward (softmax, top-k, optional
    renormalisation, weighted sum), only the expert MLPs run on their own devices.
    """
    import torch
    import torch.nn.functional as F

    gateway = devices[gateway_idx]

    for li, layer in enumerate(model.model.layers):
        block = layer.mlp

        def forward(hidden_states, block=block, li=li):
            b, s, h = hidden_states.shape
            x = hidden_states.view(-1, h)
            router_logits = block.gate(x)
            w = F.softmax(router_logits, dim=1, dtype=torch.float)
            w, sel = torch.topk(w, block.top_k, dim=-1)
            if block.norm_topk_prob:
                w /= w.sum(dim=-1, keepdim=True)
            w = w.to(x.dtype)
            out = torch.zeros_like(x)
            by_dev = {}
            for e in torch.unique(sel).tolist():
                di = devices.index(next(block.experts[e].parameters()).device)
                by_dev.setdefault(di, []).append(e)
            for di, experts in by_dev.items():
                dev = devices[di]
                needed = torch.unique(torch.cat([torch.where(sel == e)[0] for e in experts]))
                xs = x[needed].to(dev, non_blocking=True) if dev != gateway else x[needed]
                pos = torch.full((x.shape[0],), -1, dtype=torch.long, device=x.device)
                pos[needed] = torch.arange(len(needed), device=x.device)
                if dev.type == "cuda":
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    with torch.cuda.device(dev):
                        start.record()
                acc = torch.zeros_like(xs)
                calls = 0
                for e in experts:
                    tok, slot = torch.where(sel == e)
                    local = pos[tok].to(dev) if dev != gateway else pos[tok]
                    y = block.experts[e](xs[local]) * w[tok, slot, None].to(dev)
                    acc.index_add_(0, local, y.to(acc.dtype))
                    calls += 1
                if dev.type == "cuda":
                    with torch.cuda.device(dev):
                        end.record()
                    rec.events.append(((li, di), (start, end)))
                out.index_add_(0, needed, acc.to(gateway, non_blocking=True) if dev != gateway else acc)
                moved = 0 if dev == gateway else len(needed) * h * x.element_size()
                rec.rows.append({"layer_id": li, "device": di, "tokens": int(len(needed)), "expert_calls": calls,
                                 "bytes_out": moved, "bytes_back": moved})
            return out.view(b, s, h), router_logits

        block.forward = forward


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", help="comma list, e.g. cuda:0,cuda:1 (default: all visible CUDA GPUs)")
    ap.add_argument("--gateway", type=int, default=0, help="index in --devices for attention/router (default 0)")
    ap.add_argument("--strategies", nargs="+", default=list(C.STRATEGIES), choices=list(C.STRATEGIES))
    ap.add_argument("--speeds", default="measure",
                    help="'measure' (micro-benchmark), 'equal', or comma list of relative speeds per device")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--check", action="store_true", help="compare logits with the unmodified single-device model")
    ap.add_argument("--checkpoint", type=Path, default=MODEL_DIR)
    ap.add_argument("--name", default="expert_parallel")
    ap.add_argument("--dry-run", action="store_true", help="print the plan for the given devices; load nothing")
    args = ap.parse_args()

    if args.dry_run:
        n = len(args.devices.split(",")) if args.devices else 4
        speeds = [1.0] * n if args.speeds in ("measure", "equal") else [float(s) for s in args.speeds.split(",")]
        cfg, pl = make_plan(n, speeds, args.gateway)
        print(f"Plan for {n} devices, speeds {speeds} (dry run: speeds 'measure' shown as equal)")
        print(P.summary(pl, "test_demand", cfg).to_string(index=False))
        print(pl.groupby(["strategy", "gpu_id"]).size().unstack().to_string())
        return 0

    if not (args.checkpoint / "config.json").exists():
        sys.exit(f"Checkpoint not found at {args.checkpoint}. This script never downloads the model.")

    import pandas as pd
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def norm(d):  # tensors report "mps:0"/"cuda:0"; make the device list compare equal to them
        d = torch.device(d)
        return torch.device(d.type, 0) if d.type in ("mps", "cuda") and d.index is None else d

    if args.devices:
        devices = [norm(d.strip()) for d in args.devices.split(",")]
    else:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    if len(devices) < 1:
        sys.exit("No devices: no CUDA GPU is visible and --devices was not given.")
    if len(set(devices)) != len(devices):
        sys.exit("--devices must not repeat a device.")
    gateway = devices[args.gateway]
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = OUTPUTS / "real_runs" / f"{args.name}-{stamp}"
    out.mkdir(parents=True, exist_ok=False)
    print(f"Devices: {[str(d) for d in devices]} (gateway {gateway}); output -> {out}")

    # speeds -> plan
    bench = None
    if args.speeds == "measure":
        speeds, bench = measure_speeds(torch, devices, dtype)
    elif args.speeds == "equal":
        speeds = [1.0] * len(devices)
    else:
        speeds = [float(s) for s in args.speeds.split(",")]
        if len(speeds) != len(devices):
            sys.exit("--speeds must list one value per device.")
    print("Relative speeds used by the planner:", [round(s, 3) for s in speeds])
    cfg, pl = make_plan(len(devices), speeds, args.gateway)
    pl.to_csv(out / "plan.csv", index=False)
    predicted = P.summary(pl, "test_demand", cfg)

    print(f"Loading OLMoE ({args.dtype}) on CPU, then distributing…")
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, dtype=dtype, attn_implementation="eager",
                                                 local_files_only=True).eval()
    load_s = time.perf_counter() - t0
    if not isinstance(model.model.layers[0].mlp.experts, torch.nn.ModuleList):
        sys.exit("This transformers version stores experts as fused tensors; install transformers==4.57.6 "
                 "(the version used by this project), where each expert is its own module.")
    mc = model.config
    assert (mc.num_experts_per_tok, mc.num_experts, mc.num_hidden_layers) == (C.TOP_K, C.EXPERTS_PER_LAYER,
                                                                             C.NUM_LAYERS)
    encoded = [tok(p, return_tensors="pt") for _, p in PROMPTS]

    # optional reference: unmodified model on one device
    check = {"done": False}
    if args.check:
        ref_dev = gateway
        try:
            model.to(ref_dev)
            with torch.inference_mode():
                ref = [model(**{k: v.to(ref_dev) for k, v in e.items()}, use_cache=False).logits.float().cpu()
                       for e in encoded]
            check.update(done=True, reference_device=str(ref_dev))
        except RuntimeError as exc:  # usually out of memory: reference needs the whole model on one device
            print(f"Reference run skipped on {ref_dev}: {exc}")
            check.update(skipped=str(exc)[:300])
            ref = None
        model.to("cpu")
        if gateway.type == "cuda":
            torch.cuda.empty_cache()

    # distribute: experts per plan, everything else on the gateway
    rec = Recorder(torch, devices)
    install_expert_parallel(model, devices, args.gateway, rec)
    first = args.strategies[0]
    t0 = time.perf_counter()
    place_experts(model, P.layout(pl, first), devices)
    move_rest_to(model, gateway)
    sync_all(torch, devices)
    print(f"Initial layout '{first}' ready in {time.perf_counter() - t0:.1f} s")

    runs, stats, relayout = [], [], []
    for si, strategy in enumerate(args.strategies):
        if si:
            sync_all(torch, devices)
            t0 = time.perf_counter()
            moved, nbytes = place_experts(model, P.layout(pl, strategy), devices)
            sync_all(torch, devices)
            secs = time.perf_counter() - t0
            relayout.append({"from": args.strategies[si - 1], "to": strategy, "experts_moved": moved,
                             "gb_copied": nbytes / C.GB, "seconds": secs,
                             "effective_gb_per_s": nbytes / C.GB / secs if secs else None})
            print(f"Re-layout {args.strategies[si - 1]} -> {strategy}: {moved} experts, {nbytes / C.GB:.2f} GB "
                  f"in {secs:.2f} s")
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(**{k: v.to(gateway) for k, v in encoded[0].items()}, use_cache=False)
            sync_all(torch, devices)
            for rid, ((cat, _), enc) in enumerate(zip(PROMPTS, encoded)):
                inputs = {k: v.to(gateway) for k, v in enc.items()}
                for rep in range(args.repeats):
                    rec.reset()
                    sync_all(torch, devices)
                    t0 = time.perf_counter()
                    res = model(**inputs, use_cache=False)
                    sync_all(torch, devices)
                    ms = 1000 * (time.perf_counter() - t0)
                    runs.append({"strategy": strategy, "request_id": rid, "category": cat,
                                 "split": "train" if rid in cfg.train_ids else "held_out",
                                 "tokens": int(enc["input_ids"].shape[1]), "repeat": rep, "latency_ms": ms})
                    cms = rec.compute_ms()
                    for row in rec.rows:
                        stats.append({"strategy": strategy, "request_id": rid, "repeat": rep, **row,
                                      "compute_ms": cms.get((row["layer_id"], row["device"]))})
                    if rep == 0 and args.check and check.get("done") and si == 0:
                        got = res.logits.float().cpu()
                        diff = (got - ref[rid]).abs().max().item()
                        agree = (got.argmax(-1) == ref[rid].argmax(-1)).float().mean().item()
                        check.setdefault("prompts", []).append({"request_id": rid, "max_abs_logit_diff": diff,
                                                                "next_token_agreement": agree})
        med = statistics.median(r["latency_ms"] for r in runs if r["strategy"] == strategy)
        print(f"{strategy:13s} median prompt latency {med:8.1f} ms")

    runs_df, stats_df = pd.DataFrame(runs), pd.DataFrame(stats)
    runs_df.to_csv(out / "runs.csv", index=False)
    stats_df.to_csv(out / "device_stats.csv", index=False)
    pd.DataFrame(relayout).to_csv(out / "relayout.csv", index=False)
    (out / "check.json").write_text(json.dumps(check, indent=2))

    # headline: measured busiest-GPU expert time vs the planner's predicted busiest load
    print("\nStrategy        median ms   busiest-GPU expert ms (held-out, median rep)   predicted peak load")
    for s in args.strategies:
        r = runs_df[(runs_df.strategy == s) & (runs_df.split == "held_out")]
        st_ = stats_df[(stats_df.strategy == s) & stats_df.request_id.isin(cfg.test_ids)]
        busiest = (st_.groupby(["request_id", "repeat", "device"]).compute_ms.sum().groupby(["request_id", "repeat"])
                   .max().groupby("request_id").median().sum()) if st_.compute_ms.notna().any() else float("nan")
        peak = predicted.set_index("strategy").loc[s, "peak_load"]
        print(f"{s:13s} {r.latency_ms.median():10.1f}   {busiest:12.2f}   {peak:12.1f}")
    if check.get("prompts"):
        worst = max(p["max_abs_logit_diff"] for p in check["prompts"])
        agree = min(p["next_token_agreement"] for p in check["prompts"])
        print(f"\nCorrectness vs single-device model: max |logit diff| {worst:.4f}, next-token agreement >= {agree:.3f}")

    gpus = []
    for d in devices:
        if d.type == "cuda":
            pr = torch.cuda.get_device_properties(d)
            gpus.append({"device": str(d), "name": pr.name, "memory_gb": pr.total_memory / C.GB,
                         "compute_capability": f"{pr.major}.{pr.minor}"})
        else:
            gpus.append({"device": str(d), "name": d.type})
    (out / "metadata.json").write_text(json.dumps({
        "kind": "measured", "what": "real expert-parallel execution of OLMoE across devices", "model": C.MODEL_ID,
        "devices": gpus, "gateway": str(gateway), "dtype": args.dtype, "speeds_used": speeds,
        "speed_benchmark_seconds": bench, "strategies": args.strategies, "warmup": args.warmup,
        "repeats": args.repeats, "prompts": [p for _, p in PROMPTS], "train_ids": list(cfg.train_ids),
        "test_ids": list(cfg.test_ids), "model_load_seconds": load_s, "created_utc": stamp,
        "torch": torch.__version__, "cuda": torch.version.cuda, "platform": platform.platform(),
        "command": " ".join(sys.argv), "limitations": LIMITATIONS,
    }, indent=2))
    print(f"\nSaved to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
