"""Planner scalability experiment (proposal feedback 10).

Two metric families are kept strictly apart:

COMPUTATIONAL performance of the planner (measured, wall clock on the machine that runs the experiment)
    planner_ms          median wall time of plan_sites() over `repeats` runs (time.perf_counter)
    planner_p95_ms      95th percentile of those runs
    candidates          number of (expert, device) candidate evaluations the planner performed
    ms_per_1k_candidates = planner_ms / (candidates / 1000)

PREDICTED inference performance of the resulting plan (analytic model, HYPOTHETICAL hardware, multisite.py)
    J                   bottleneck utilisation (max over devices and inter-site links)
    latency_ms          added cross-site latency per token
    remote_share        share of assignments served in another site

Baseline assumptions: CostModel defaults (15-min window, 5e4 assignments/s per speed-1.0 device, 1.25 GB/s and
60 ms RTT between sites); device speeds cycle (1.0, 1.3, 0.8, 1.6); demand per window = 50 % of total cluster
compute capacity, popularity resampled from the measured OLMoE per-expert distribution; locality mode with a 0.7
utilisation cap. Each factor is varied one at a time around the base point below; growth exponents are fitted on
log-log scale (time ~ x^k).
"""
from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd

from . import multisite as M
from . import workload as WL

BASE = {"experts": 1024, "sites": 3, "devices_per_site": 4, "replica_frac": 0.25, "constraints": "capacity+memory+residency"}
GRID = {
    "experts": [16, 64, 256, 1024, 4096, 16384],
    "devices_per_site": [1, 2, 4, 8, 16, 32],
    "sites": [1, 2, 3, 5, 8],
    "replica_frac": [0.0, 0.1, 0.25, 0.5, 1.0, 2.0],
    "constraints": ["none", "capacity", "capacity+memory", "capacity+memory+residency"],
}
PROPOSAL_POINT = {"experts": 16, "sites": 3, "devices_per_site": 1, "replica_frac": 0.25, "constraints": "capacity+memory+residency"}
SPEEDS = (1.0, 1.3, 0.8, 1.6)


def _constraints(label: str) -> M.Constraints:
    on = set(label.split("+")) if label != "none" else set()
    return M.Constraints(capacity="capacity" in on, memory="memory" in on, residency="residency" in on)


def build_instance(experts: int, sites: int, devices_per_site: int, replica_frac: float, constraints: str,
                   seed: int = 0, routing: pd.DataFrame | None = None, load: float = 0.5):
    cons = _constraints(constraints)
    G = sites * devices_per_site
    copies = experts * (1 + replica_frac)
    slots = math.ceil(copies * 1.3 / G)
    if cons.residency:
        slots = max(slots, math.ceil(experts / devices_per_site))
    cost = M.CostModel()
    mem_gb = slots * cost.expert_bytes / 1e9 if cons.memory else None
    site_list = [M.Site(f"s{i}", residency=(cons.residency and i == 0 and sites > 1)) for i in range(sites)]
    topo = M.make_topology(site_list, devices_per_site=devices_per_site, speeds=SPEEDS, slots=slots,
                           memory_gb=mem_gb, cost=cost)
    total_cap = topo.speeds.sum() * cost.assignments_per_s * cost.window_s
    Y = WL.synthetic_demand(experts, sites, total_per_window=load * total_cap, seed=seed, routing=routing)
    return Y, topo, cons


def time_planner(Y, topo, cons, replica_frac, repeats=5, util_cap=0.7) -> dict:
    times, plan = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        plan = M.plan_sites(Y, topo, replica_frac=replica_frac, constraints=cons, util_cap=util_cap)
        times.append((time.perf_counter() - t0) * 1e3)
    t1 = time.perf_counter()
    ev = M.evaluate(plan, Y, topo, util_cap)
    eval_ms = (time.perf_counter() - t1) * 1e3
    cand = plan.info["candidates_evaluated"]
    med = float(np.median(times))
    return {"planner_ms": med, "planner_p95_ms": float(np.percentile(times, 95)), "candidates": cand,
            "ms_per_1k_candidates": med / max(cand / 1000, 1e-9), "evaluate_ms": eval_ms,
            "replicas_added": plan.info["replicas_added"], "copies": plan.copies,
            "J": ev["J"], "latency_ms": ev["added_latency_ms_per_token"], "remote_share": ev["remote_share"],
            "residency_violations": ev["residency_violation_assignments"]}


def run(routing: pd.DataFrame | None = None, repeats: int = 5, seeds=(0, 1, 2), grid: dict | None = None,
        log=print) -> pd.DataFrame:
    grid = grid or GRID
    rows = []
    points = [("proposal_point", PROPOSAL_POINT)]
    for factor, values in grid.items():
        for v in values:
            points.append((factor, {**BASE, factor: v}))
    for factor, p in points:
        for seed in seeds:
            try:
                Y, topo, cons = build_instance(seed=seed, routing=routing, **p)
                res = time_planner(Y, topo, cons, p["replica_frac"], repeats=repeats)
                status = "ok"
            except M.InfeasiblePlan as exc:
                res, status = {}, f"infeasible: {exc}"
            rows.append({"factor": factor, **p, "devices": p["sites"] * p["devices_per_site"], "seed": seed,
                         "status": status, **res})
        log(f"[{factor}] {p} -> {rows[-1].get('planner_ms', float('nan')):.1f} ms")
    return pd.DataFrame(rows)


def growth_fits(df: pd.DataFrame) -> pd.DataFrame:
    """Fit log(planner_ms) = k log(x) + c per factor (median over seeds)."""
    out = []
    xcol = {"experts": "experts", "devices_per_site": "devices", "sites": "sites", "replica_frac": "replica_frac"}
    ok = df[df.status == "ok"]
    for factor, col in xcol.items():
        g = ok[ok.factor == factor].groupby(col).planner_ms.median().reset_index()
        g = g[g[col] > 0]
        if len(g) < 3:
            continue
        x, y = np.log(g[col].to_numpy(float)), np.log(g.planner_ms.to_numpy(float))
        k, c = np.polyfit(x, y, 1)
        r2 = 1 - ((y - (k * x + c)) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-12)
        out.append({"factor": factor, "x": col, "exponent_k": k, "r2": r2,
                    "range": f"{g[col].min():g}..{g[col].max():g}",
                    "ms_at_min": float(g.planner_ms.iloc[0]), "ms_at_max": float(g.planner_ms.iloc[-1])})
    return pd.DataFrame(out)


def optimality_gap(n_instances: int = 20, experts: int = 7, devices: int = 3, seed: int = 0,
                   routing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Greedy vs exhaustive optimum of J on tiny single-copy instances (balance mode, capacity only)."""
    rows = []
    for i in range(n_instances):
        sites = [M.Site(f"s{j}") for j in range(devices)]
        topo = M.make_topology(sites, devices_per_site=1, speeds=SPEEDS, slots=math.ceil(experts / devices) + 1)
        total_cap = topo.speeds.sum() * topo.cost.assignments_per_s * topo.cost.window_s
        Y = WL.synthetic_demand(experts, devices, total_per_window=0.5 * total_cap, seed=seed + i, routing=routing)
        cons = M.Constraints(capacity=True, memory=False, residency=False)
        g = M.plan_sites(Y, topo, constraints=cons)
        o = M.exact_small(Y, topo, cons)
        jg, jo = M.J_of(g, Y, topo), M.J_of(o, Y, topo)
        rows.append({"instance": i, "experts": experts, "devices": devices, "J_greedy": jg, "J_optimal": jo,
                     "gap_rel": (jg - jo) / jo})
    return pd.DataFrame(rows)
