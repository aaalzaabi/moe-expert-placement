"""Reusable experiment runner (simulation only; never loads OLMoE).

Examples (run from anywhere):
  python scripts/run_experiment.py                                   # defaults: 12.583 MB experts, 12 GB/s
  python scripts/run_experiment.py --preset phase4e_legacy            # regenerate the saved Phase 4E scenario
  python scripts/run_experiment.py --bandwidth 3 12 25 100            # sweep -> outputs/runs/<sweep>/summary.csv
  python scripts/run_experiment.py --bandwidth 6 12 --expert-mb 8 12.582912 --arrival-gap-ms 100 200
  python scripts/run_experiment.py --gpu-speeds 1 1 1 1 --link-model shared_port --policy cost_aware
  python scripts/run_experiment.py --arrival-process poisson --seed 1 2 3

Outputs go to outputs/runs/ with metadata.json beside every CSV. Bandwidth is GB/s (gigaBYTES/s).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from moeplace import config as C  # noqa: E402
from moeplace import experiment as E  # noqa: E402


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default=None, help="run name prefix (default: preset name or 'scenario')")
    p.add_argument("--preset", choices=sorted(C.PRESETS), help="start from a saved-experiment preset")
    p.add_argument("--bandwidth", type=float, nargs="+", help="link bandwidth in GB/s (several values = sweep)")
    p.add_argument("--expert-mb", type=float, nargs="+", help="expert weight size in MB (1e6 bytes); checkpoint = 12.582912")
    p.add_argument("--arrival-gap-ms", type=float, nargs="+", help="mean gap between request arrivals (ms)")
    p.add_argument("--arrival-rate-rps", type=float, help="alternative to --arrival-gap-ms: requests per simulated second")
    p.add_argument("--arrival-process", choices=C.ARRIVAL_PROCESSES)
    p.add_argument("--seed", type=int, nargs="+", help="random seed(s) for poisson arrivals")
    p.add_argument("--gpu-speeds", type=float, nargs="+", help="relative speed per simulated GPU, e.g. 1 1.3 0.8 1.6")
    p.add_argument("--capacity-slots", type=int, help="experts per GPU for placement (default 256, or ceil(1024/n))")
    p.add_argument("--staging-slots", type=int, help="resident + staged experts allowed during migration")
    p.add_argument("--gpu-memory-gb", type=float, help="optional per-GPU byte budget for expert weights")
    p.add_argument("--link-model", choices=C.LINK_MODELS)
    p.add_argument("--migration-start-ms", type=float)
    p.add_argument("--min-gain-pct", type=float, help="controller: minimum forecast makespan gain")
    p.add_argument("--max-p95-regression-pct", type=float, help="controller: maximum forecast p95 increase")
    p.add_argument("--policy", choices=C.POLICIES + ("all",), default="all",
                   help="which policy rows to print (all are always computed and saved)")
    p.add_argument("--routing", help="routing CSV or measurement-run directory (default: outputs/workload_routing.csv)")
    p.add_argument("--fail-after-copies", type=int, help="fault injection: fail after N successful copies")
    p.add_argument("--overwrite", action="store_true", help="recompute even if the run id already exists")
    p.add_argument("--list", action="store_true", help="list existing runs and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse(argv)
    if a.list:
        print(E.list_runs().to_string(index=False) or "No runs yet.")
        return 0
    base = C.PRESETS[a.preset] if a.preset else C.DEFAULT
    fixed = {}
    if a.gpu_speeds:
        fixed["gpu_speeds"] = tuple(a.gpu_speeds)
        if a.capacity_slots is None and len(a.gpu_speeds) != base.num_gpus:
            fixed["placement_capacity_slots"] = None
            fixed["staging_capacity_slots"] = None
    for arg, key in [("capacity_slots", "placement_capacity_slots"), ("staging_slots", "staging_capacity_slots"),
                     ("gpu_memory_gb", "gpu_memory_gb"), ("link_model", "link_model"),
                     ("arrival_process", "arrival_process"), ("migration_start_ms", "migration_start_ms"),
                     ("min_gain_pct", "min_makespan_gain_pct"), ("max_p95_regression_pct", "max_p95_regression_pct"),
                     ("fail_after_copies", "fail_after_copies")]:
        if getattr(a, arg) is not None:
            fixed[key] = getattr(a, arg)
    if a.arrival_rate_rps:
        if a.arrival_gap_ms:
            sys.exit("Use either --arrival-gap-ms or --arrival-rate-rps, not both.")
        a.arrival_gap_ms = [1000.0 / a.arrival_rate_rps]

    try:
        base = base.with_(**fixed)
    except C.ConfigError as exc:
        sys.exit(str(exc))

    grid = {}
    if a.bandwidth:
        grid["bandwidth_gb_per_s"] = a.bandwidth
    if a.expert_mb:
        grid["expert_bytes"] = [int(round(mb * C.MB)) for mb in a.expert_mb]
    if a.arrival_gap_ms:
        grid["arrival_gap_ms"] = a.arrival_gap_ms
    if a.seed:
        grid["seed"] = a.seed

    name = a.name or a.preset or "scenario"
    print("Simulation only. Hardware parameters are hypothetical; bandwidth unit = GB/s.")
    if sum(len(v) for v in grid.values()) > len(grid):  # at least one axis has >1 value
        out = E.sweep(base, grid, name=name, routing_source=a.routing, overwrite=a.overwrite)
        summary = pd.read_csv(out / "summary.csv")
        print(summary.drop(columns=["run_id"], errors="ignore").round(2).to_string(index=False))
        return 0 if (summary.status == "ok").all() else 1

    cfg = base.with_(**{k: v[0] for k, v in grid.items()})
    run_dir = E.run_scenario(cfg, name=name, routing_source=a.routing, overwrite=a.overwrite)
    pol = pd.read_csv(run_dir / "policy_comparison.csv")
    if a.policy != "all":
        pol = pol[pol.policy == a.policy]
    dec = pd.read_csv(run_dir / "policy_decision.csv").iloc[0]
    print(f"\nRun: {run_dir}")
    print(f"Controller decision (train ids {dec.forecast_request_ids}): {dec.decision.upper()}")
    for reason in str(dec.reasons).split(" | "):
        print("  " + reason)
    cols = ["policy", "migration_state", "p50_latency_ms", "p95_latency_ms", "makespan_ms",
            "migration_mb", "migration_activation_ms"]
    print(f"\nHeld-out evaluation (test ids {dec.evaluation_request_ids}, simulated):")
    print(pol[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
