"""Planner scalability sweep (experts, devices, sites, replicas, constraints) + tiny-instance optimality gap.

Reports the planner's COMPUTATIONAL cost (wall-clock ms, candidate evaluations) separately from the PREDICTED
inference metrics of its plans (utilisation J, cross-site latency). Writes outputs/runs/scalability-<hash>/.

Examples
  python scripts/run_scalability.py                    # full grid, 3 seeds x 5 repeats (~1-2 min)
  python scripts/run_scalability.py --quick            # 1 seed x 2 repeats
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from moeplace import data as D  # noqa: E402
from moeplace import experiment as E  # noqa: E402
from moeplace import scalability as SC  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = p.parse_args(argv)
    repeats, seeds = (2, [0]) if a.quick else (a.repeats, a.seeds)
    routing = D.load_routing()
    df = SC.run(routing, repeats=repeats, seeds=seeds)
    fits = SC.growth_fits(df)
    gap = SC.optimality_gap(20, 7, 3, routing=routing)
    print(fits.to_string(index=False))
    print(f"Greedy vs optimum on 20 tiny instances: mean gap {gap.gap_rel.mean():.4f}, max {gap.gap_rel.max():.4f}")
    E.save_frames("scalability", {"repeats": repeats, "seeds": seeds, "grid": SC.GRID, "base": SC.BASE,
                                   "machine": platform.platform(), "processor": platform.processor()},
                  {"scalability.csv": df, "growth_fits.csv": fits, "optimality_gap.csv": gap},
                  result_kind="measured planner run time + simulated plan quality",
                  assumptions="Planner wall time is MEASURED on this machine. Demand is synthetic (resampled from the "
                              "measured OLMoE expert distribution); J and latency come from the analytic model with "
                              "HYPOTHETICAL hardware parameters.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
