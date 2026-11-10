"""Re-run the held-out placement evaluation (Phase 3B) on the 24-prompt routing measurement.

The 24-prompt run (`profile_routing.py --prompt-set extended`) re-profiles the original six prompts as request ids
0-5 and adds 18 new ones (6-23), 8 per category in total. This script answers: does demand-aware placement still
beat static and load-balanced placement on prompts it never saw, and how stable is the gain across splits?

Evaluations (all on the same simulated 4-GPU cluster as the original: speeds 1.0/1.3/0.8/1.6, 256 slots each):
  sanity      original split (train 0,2,4 / test 1,3,5) on the new file; must reproduce the published numbers
  unseen      train on the original 6 prompts, test on the 18 new prompts (no prompt overlap at all)
  splits      N random 50/50 splits of all 24 prompts, stratified by category (4 train / 4 test per category)
  curve       k = 1..6 training prompts per category, test on the remaining prompts, N random draws each

Peak load = max over GPUs of (held-out token assignments / GPU speed); lower is better. Routing is MEASURED;
GPU speeds and slot limits are the project's HYPOTHETICAL cluster, so this is a load proxy, not a latency.

Usage:  python scripts/evaluate_extended_prompts.py [--routing DIR] [--splits 1000] [--seed 0]
Output: outputs/runs/extended_prompts-<hash>/ (CSV files + metadata.json)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from moeplace import config as C  # noqa: E402
from moeplace import data as D  # noqa: E402
from moeplace import experiment as E  # noqa: E402
from moeplace import placement as P  # noqa: E402

DEFAULT_ROUTING = Path(__file__).resolve().parents[1] / "outputs" / "measurements" / "routing-20261110T123026Z"
ORIGINAL_IDS = tuple(range(6))


def peaks(routing: pd.DataFrame, train_ids, test_ids, cfg: C.SimConfig) -> dict[str, float]:
    """Held-out peak normalised load of each strategy, planned on train_ids only."""
    if set(train_ids) & set(test_ids):
        raise ValueError("train and test prompts overlap (evaluation leakage)")
    pl = P.plan(D.demand(routing, train_ids), cfg, eval_demand=D.demand(routing, test_ids))
    return {s: float(P.gpu_metrics(pl, s, "test_demand", cfg).normalized_load.max()) for s in C.STRATEGIES}


def row(name: str, train_ids, test_ids, pk: dict[str, float], **extra) -> dict:
    r = {"evaluation": name, "train_prompts": len(train_ids), "test_prompts": len(test_ids), **extra}
    for s, v in pk.items():
        r[f"peak_{s}"] = v
    r["demand_aware_vs_static_pct"] = 100 * (pk["demand_aware"] - pk["static"]) / pk["static"]
    r["demand_aware_vs_balanced_pct"] = 100 * (pk["demand_aware"] - pk["balanced"]) / pk["balanced"]
    return r


def stratified(cats: pd.Series, k: int, rng: np.random.Generator) -> tuple[list[int], list[int]]:
    train, test = [], []
    for _, ids in cats.groupby(cats).groups.items():
        ids = sorted(int(i) for i in ids)
        pick = set(rng.choice(ids, size=k, replace=False).tolist())
        train += [i for i in ids if i in pick]
        test += [i for i in ids if i not in pick]
    return sorted(train), sorted(test)


def summarise(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    def agg(g):
        x, b = g.demand_aware_vs_static_pct, g.demand_aware_vs_balanced_pct
        return pd.Series({
            "draws": len(g),
            "mean_vs_static_pct": x.mean(), "p2.5_vs_static_pct": x.quantile(0.025),
            "p97.5_vs_static_pct": x.quantile(0.975),
            "share_beats_static_pct": 100 * (x < 0).mean(),
            "mean_vs_balanced_pct": b.mean(), "share_beats_balanced_pct": 100 * (b < 0).mean(),
        })
    return df.groupby(by).apply(agg, include_groups=False).reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--routing", type=Path, default=DEFAULT_ROUTING, help="measurement run directory")
    ap.add_argument("--splits", type=int, default=1000, help="random draws for 'splits' and per point of 'curve'")
    ap.add_argument("--random-layouts", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = C.SimConfig()
    routing = D.load_routing(a.routing)  # validates schema and the top-8 invariant
    cats = routing.groupby("request_id").category.first()
    ids = sorted(int(i) for i in cats.index)
    new_ids = [i for i in ids if i not in ORIGINAL_IDS]
    rng = np.random.default_rng(a.seed)

    # 1. sanity: the original split on the re-profiled prompts must match the published Phase 3B result
    rows = [row("sanity: original split", cfg.train_ids, cfg.test_ids, peaks(routing, cfg.train_ids, cfg.test_ids, cfg))]

    # 2. unseen prompts: plan on the original 6, evaluate on the 18 new ones, with the random-layout baseline
    pl = P.plan(D.demand(routing, ORIGINAL_IDS), cfg, eval_demand=D.demand(routing, new_ids))
    v = P.validate_vs_random(pl, cfg, n=a.random_layouts, seed=a.seed)
    t = v["table"].set_index("strategy")
    rows.append(row("unseen: train original 6, test 18 new", ORIGINAL_IDS, new_ids,
                    {s: float(t.loc[s, "peak_load"]) for s in C.STRATEGIES},
                    demand_aware_beats_random_pct=float(t.loc["demand_aware", "beats_random_pct"]),
                    demand_aware_gap_to_lower_bound_pct=float(t.loc["demand_aware", "gap_to_best_pct"]),
                    random_median_peak=v["random_median"], lower_bound_peak=v["lower_bound"]))
    headline = pd.DataFrame(rows)

    # 3. repeated stratified 50/50 splits of all 24 prompts
    per_cat = int(cats.value_counts().min())
    split_rows = []
    for d in range(a.splits):
        tr, te = stratified(cats, per_cat // 2, rng)
        split_rows.append(row("splits", tr, te, peaks(routing, tr, te, cfg), draw=d))
    splits = pd.DataFrame(split_rows)

    # 4. learning curve: more training prompts per category, test on the rest
    curve_rows = []
    for k in range(1, per_cat - 1):
        for d in range(a.splits // 5):
            tr, te = stratified(cats, k, rng)
            curve_rows.append(row("curve", tr, te, peaks(routing, tr, te, cfg), train_per_category=k, draw=d))
    curve = pd.DataFrame(curve_rows)

    split_summary = summarise(splits, ["evaluation"])
    curve_summary = summarise(curve, ["train_per_category"])

    params = {"routing_dir": str(a.routing), "splits": a.splits, "random_layouts": a.random_layouts, "seed": a.seed,
              "gpu_speeds": list(cfg.gpu_speeds), "capacity_slots": cfg.capacity_slots,
              "original_ids": list(ORIGINAL_IDS), "new_ids": new_ids}
    out = E.save_frames("extended_prompts", params,
                        {"headline.csv": headline, "splits.csv": splits, "splits_summary.csv": split_summary,
                         "curve.csv": curve, "curve_summary.csv": curve_summary},
                        result_kind="measured routing; simulated hardware", routing_source=a.routing)

    pd.set_option("display.width", 200)
    print(headline.drop(columns=["evaluation"]).assign(evaluation=headline.evaluation).to_string(index=False))
    print(split_summary.to_string(index=False))
    print(curve_summary.to_string(index=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
