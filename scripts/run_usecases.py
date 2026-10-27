"""Run the six use cases: site-aware forecasting, placement regret, drift analysis and the operations agent.

Simulation only (never loads OLMoE). Demand = synthetic site traffic over MEASURED OLMoE routing profiles;
all hardware is HYPOTHETICAL. Writes outputs/runs/usecases-<hash>/ with one sub-folder per use case.

Examples
  python scripts/run_usecases.py                       # all six use cases, 2 evaluation days each (~3-5 min)
  python scripts/run_usecases.py --only uc4_workload_shift --eval-days 1
  python scripts/run_usecases.py --list

Each use case's result is cached in outputs/runs/.partial-usecases/ so a long run can be done in pieces
(e.g. `--only uc1_multi_region --no-save`, then a final call without --only combines everything). --fresh
ignores the cache.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from moeplace import agent as A  # noqa: E402
from moeplace import data as D  # noqa: E402
from moeplace import experiment as E  # noqa: E402
from moeplace import forecast as F  # noqa: E402
from moeplace import multisite as M  # noqa: E402
from moeplace import usecases as U  # noqa: E402

ASSUMPTIONS = ("Site traffic, device speeds, bandwidth, RTT, throughput and events are HYPOTHETICAL. Expert-routing "
               "profiles per category are MEASURED on OLMoE. Metrics come from an analytic utilisation/latency model, "
               "not from real multi-site GPU serving.")


def run_one(uc: U.UseCase, routing: pd.DataFrame, eval_days: int, log=print) -> tuple[dict, dict]:
    t0 = time.perf_counter()
    Y, cats = uc.demand(routing)
    spec = U.spec()
    end = min(uc.T, uc.train_end + eval_days * U.DAY)
    kw = uc.plan_kw()
    rows = F.rolling_evaluation(Y, uc.topology, spec, uc.train_end, eval_end=end, **kw)
    summ = F.summarize(rows)
    drift = F.drift_table(rows)
    agent = A.PlacementAgent(Y, uc.topology, spec, A.AgentConfig(consolidation=uc.consolidation), plan_kw=kw)
    decisions, trace = agent.run(uc.train_end, end, events=uc.events)
    audit = pd.DataFrame(agent.audit)
    # baseline comparison: constraint cost for the residency use case
    site_volume = pd.DataFrame(Y.sum(axis=2), columns=[s.name for s in uc.topology.sites]).assign(window=range(uc.T))
    best = summ.iloc[0]
    d = decisions
    out = {
        "use_case": uc.key, "title": uc.title, "users": uc.users,
        "sites": uc.topology.n_sites, "devices": uc.topology.n_devices, "mode": "balance" if uc.util_cap is None else f"locality (cap {uc.util_cap})",
        "best_forecaster": best.model, "best_mean_wape": best.mean_wape, "best_mean_regret_abs": best.mean_regret_abs,
        "best_aggregate_regret_rel": best.aggregate_regret_rel,
        "site_ewma_aggregate_regret_rel": float(summ.set_index("model").loc["site_ewma", "aggregate_regret_rel"]),
        "global_ewma_aggregate_regret_rel": float(summ.set_index("model").loc["global_ewma", "aggregate_regret_rel"]),
        "site_ewma_popularity_error": float(summ.set_index("model").loc["site_ewma", "mean_popularity_error"]),
        "global_ewma_popularity_error": float(summ.set_index("model").loc["global_ewma", "mean_popularity_error"]),
        "agent_decisions": len(d), "agent_replans": int((d.action != "keep").sum()),
        "agent_refits": int(d.refit.sum()), "agent_human_approvals": int(d.get("needs_human", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()),
        "agent_migration_gb": float(d.loc[d.action != "keep", "migration_gb"].sum()),
        "objective_agent": float(d.objective_actual_agent.mean()), "objective_static": float(d.objective_actual_static.mean()),
        "objective_oracle": float(d.objective_actual_oracle.mean()),
        "util_agent_max": float(d.util_actual_agent.max()), "util_static_max": float(d.util_actual_static.max()),
        "latency_ms_agent": float(d.latency_ms_actual_agent.mean()), "latency_ms_static": float(d.latency_ms_actual_static.mean()),
        "residency_violations": float(d.residency_violations_actual.sum()),
        "gpu_hours_released": float(d.gpu_hours_released.sum()),
        "runtime_s": time.perf_counter() - t0,
    }
    frames = {f"{uc.key}/forecast_points.csv": rows, f"{uc.key}/forecast_summary.csv": summ,
              f"{uc.key}/drift.csv": drift, f"{uc.key}/agent_decisions.csv": decisions,
              f"{uc.key}/agent_tool_trace.csv": trace, f"{uc.key}/agent_audit.csv": audit,
              f"{uc.key}/site_volume.csv": site_volume}
    log(f"[{uc.key}] best={best.model} regret={best.aggregate_regret_rel:.4f} agent={out['objective_agent']:.3f} "
        f"static={out['objective_static']:.3f} oracle={out['objective_oracle']:.3f} ({out['runtime_s']:.0f} s)")
    return out, frames


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=sorted(U.USE_CASES))
    p.add_argument("--eval-days", type=int, default=2)
    p.add_argument("--routing", help="routing CSV or measurement-run directory (default: outputs/workload_routing.csv)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--no-save", action="store_true", help="only compute/cache the selected use cases")
    p.add_argument("--fresh", action="store_true", help="recompute even if a cached partial result exists")
    a = p.parse_args(argv)
    if a.list:
        for k, uc in U.USE_CASES.items():
            print(f"{k:28s} {uc.title}\n{'':28s} users: {uc.users}")
        return 0
    routing = D.load_routing(a.routing)
    keys = a.only or list(U.USE_CASES)
    summary, frames = [], {}
    cache = E.RUNS / ".partial-usecases"
    cache.mkdir(parents=True, exist_ok=True)
    tag = E.file_sha256(E.routing_path(a.routing))[:10]
    for k in keys:
        f = cache / f"{k}-eval{a.eval_days}-{tag}.pkl"
        if f.exists() and not a.fresh:
            out, fr = pickle.loads(f.read_bytes())
            print(f"[{k}] loaded cached result")
        else:
            out, fr = run_one(U.USE_CASES[k], routing, a.eval_days)
            f.write_bytes(pickle.dumps((out, fr)))
        summary.append(out)
        frames.update(fr)
    if a.no_save:
        return 0
    frames["summary.csv"] = pd.DataFrame(summary)
    E.save_frames("usecases", {"use_cases": keys, "eval_days": a.eval_days}, frames, routing_source=a.routing,
                  assumptions=ASSUMPTIONS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
