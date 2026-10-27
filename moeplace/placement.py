"""Expert placement algorithms (generalised from optimize_placement.py / evaluate_placement.py).

Loads here are token-assignment counts divided by a hypothetical relative GPU
speed. They are a LOAD PROXY, not GPU utilisation and not latency.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def assign(demand: np.ndarray, strategy: str, speeds, capacity: int) -> np.ndarray:
    """Return gpu_id for each expert (experts in canonical (layer, expert) order).

    static        round-robin by expert index (i mod n), skipping full GPUs
    balanced      greedy: highest-demand expert -> GPU with lowest raw load (ignores speed)
    demand_aware  greedy: highest-demand expert -> GPU minimising (load + demand) / speed
    Ties break toward the lower GPU id; demand ties keep canonical order (stable sort).
    """
    demand = np.asarray(demand, dtype=float)
    speeds = np.asarray(speeds, dtype=float)
    n = len(speeds)
    if capacity * n < len(demand):
        raise ValueError(f"Insufficient capacity: {n} GPUs x {capacity} slots < {len(demand)} experts")
    loads = np.zeros(n)
    counts = np.zeros(n, dtype=int)
    out = np.empty(len(demand), dtype=int)

    if strategy == "static":
        order = range(len(demand))
    elif strategy in ("balanced", "demand_aware"):
        order = sorted(range(len(demand)), key=lambda i: -demand[i])
    else:
        raise ValueError(f"Unknown strategy {strategy!r}; expected one of {C.STRATEGIES}")

    for i in order:
        d = demand[i]
        feasible = [g for g in range(n) if counts[g] < capacity]
        if strategy == "static":
            gpu = next((g % n for g in range(i, i + n) if counts[g % n] < capacity), feasible[0])
        elif strategy == "balanced":
            gpu = min(feasible, key=lambda g: (loads[g], g))
        else:
            gpu = min(feasible, key=lambda g: ((loads[g] + d) / speeds[g], g))
        out[i] = gpu
        loads[gpu] += d
        counts[gpu] += 1
    return out


def plan(train_demand: pd.Series, cfg: C.SimConfig, strategies=C.STRATEGIES, eval_demand: pd.Series | None = None) -> pd.DataFrame:
    """Placements for all strategies in the schema of heterogeneous_placements.csv."""
    base = train_demand.rename("train_demand").reset_index()
    if eval_demand is not None:
        base["test_demand"] = eval_demand.reindex(train_demand.index, fill_value=0).to_numpy()
    frames = []
    for s in strategies:
        gpu = assign(base.train_demand.to_numpy(), s, cfg.gpu_speeds, cfg.capacity_slots)
        frames.append(base.assign(gpu_id=gpu, strategy=s))
    return pd.concat(frames, ignore_index=True)


def layout(placements: pd.DataFrame, strategy: str) -> dict[tuple[int, int], int]:
    rows = placements[placements.strategy == strategy]
    if rows.empty:
        raise ValueError(f"No placement rows for strategy {strategy!r}")
    if rows.duplicated(["layer_id", "expert_id"]).any():
        raise ValueError(f"Duplicate expert in {strategy} placement")
    return {(int(r.layer_id), int(r.expert_id)): int(r.gpu_id) for r in rows.itertuples()}


def gpu_metrics(placements: pd.DataFrame, strategy: str, demand_col: str, cfg: C.SimConfig) -> pd.DataFrame:
    """Per-GPU load proxy, expert count, weight residency and feasibility."""
    rows = placements[placements.strategy == strategy]
    n = cfg.num_gpus
    load = rows.groupby("gpu_id")[demand_col].sum().reindex(range(n), fill_value=0).to_numpy(float)
    count = rows.groupby("gpu_id").size().reindex(range(n), fill_value=0).to_numpy()
    speeds = np.asarray(cfg.gpu_speeds, float)
    weight_gb = count * cfg.expert_bytes / C.GB
    out = pd.DataFrame({
        "gpu_id": range(n),
        "speed": speeds,
        "experts": count,
        "assignments": load,
        "normalized_load": load / speeds,
        "weight_gb": weight_gb,
        "slot_capacity": cfg.capacity_slots,
    })
    out["slots_ok"] = out.experts <= cfg.capacity_slots
    if cfg.gpu_memory_gb is not None:
        out["memory_gb"] = cfg.gpu_memory_gb
        out["memory_ok"] = out.weight_gb <= cfg.gpu_memory_gb
    else:
        out["memory_ok"] = True
    out["feasible"] = out.slots_ok & out.memory_ok
    return out


def summary(placements: pd.DataFrame, demand_col: str, cfg: C.SimConfig, strategies=C.STRATEGIES) -> pd.DataFrame:
    """Peak/mean normalized load and imbalance (same definitions as heldout_comparison.csv)."""
    rows = []
    for s in strategies:
        m = gpu_metrics(placements, s, demand_col, cfg)
        mean = m.normalized_load.mean()
        rows.append({
            "strategy": s,
            "peak_load": m.normalized_load.max(),
            "mean_load": mean,
            "imbalance": m.normalized_load.max() / mean if mean > 0 else 1.0,
            "max_experts_per_gpu": int(m.experts.max()),
            "max_weight_gb": float(m.weight_gb.max()),
            "feasible": bool(m.feasible.all()),
        })
    return pd.DataFrame(rows)


def moved_experts(placements: pd.DataFrame, src: str, dst: str) -> pd.DataFrame:
    a, b = layout(placements, src), layout(placements, dst)
    rows = [{"layer_id": k[0], "expert_id": k[1], "src_gpu": a[k], "dst_gpu": b[k]} for k in sorted(a) if a[k] != b[k]]
    return pd.DataFrame(rows, columns=["layer_id", "expert_id", "src_gpu", "dst_gpu"])


def random_baseline(eval_demand, cfg: C.SimConfig, n: int = 10_000, seed: int = 0) -> np.ndarray:
    """Peak normalized load of n RANDOM placements that respect the same slot capacity.

    Used to answer "is the planner better than guessing?": a placement method is only meaningful if it beats
    (almost) all random placements on held-out demand."""
    eval_demand = np.asarray(eval_demand, float)
    speeds = np.asarray(cfg.gpu_speeds, float)
    slots = np.repeat(np.arange(cfg.num_gpus), cfg.capacity_slots)  # every free slot, shuffled
    rng = np.random.default_rng(seed)
    peaks = np.empty(n)
    for i in range(n):
        gpu = rng.permutation(slots)[: len(eval_demand)]
        peaks[i] = (np.bincount(gpu, weights=eval_demand, minlength=cfg.num_gpus) / speeds).max()
    return peaks


def lower_bound(eval_demand, cfg: C.SimConfig) -> float:
    """Best possible peak normalized load: work split exactly in proportion to GPU speed (ignores slot limits,
    so no real placement can do better)."""
    return float(np.sum(eval_demand) / np.sum(cfg.gpu_speeds))


def validate_vs_random(placements: pd.DataFrame, cfg: C.SimConfig, demand_col: str = "test_demand",
                       n: int = 10_000, seed: int = 0) -> dict:
    """Compare each strategy's held-out peak load with random placements, the theoretical best and hindsight."""
    base = placements[placements.strategy == placements.strategy.iloc[0]].sort_values(["layer_id", "expert_id"])
    demand = base[demand_col].to_numpy(float)
    rand = random_baseline(demand, cfg, n, seed)
    lb = lower_bound(demand, cfg)
    hindsight = (np.bincount(assign(demand, "demand_aware", cfg.gpu_speeds, cfg.capacity_slots), weights=demand,
                             minlength=cfg.num_gpus) / np.asarray(cfg.gpu_speeds)).max()
    rows = []
    for s in placements.strategy.unique():
        peak = gpu_metrics(placements, s, demand_col, cfg).normalized_load.max()
        rows.append({"strategy": s, "peak_load": peak, "beats_random_pct": 100 * float((rand > peak).mean()),
                     "gap_to_best_pct": 100 * (peak - lb) / lb})
    return {"table": pd.DataFrame(rows), "random": rand, "lower_bound": lb, "hindsight": float(hindsight),
            "random_best": float(rand.min()), "random_median": float(np.median(rand)), "n": n, "seed": seed}
